from functools import partial
import gymnasium as gym
from jaxtyping import Float
import numpy as np
import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Union, Tuple
from typing_extensions import Self
from tqdm import tqdm
import wandb

from src.utils import MovingAverage
from src.construct_pole_balancing_env import ConstructPoleBalancingEnv

# device = T.device("mps" if T.backends.mps.is_available() else "cpu")
device = T.device("cpu")  # right now the network is too small to benefit from GPUs


class Agent(nn.Module):
    def __init__(
        self,
        d_obs: int,
        n_act: int,
        hidden_dims: list[int],
        device: Optional[T.device] = None,
    ) -> None:
        super().__init__()

        # create the shared network (common to both actor and critic)
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(nn.Linear(d_obs, hidden_dim, device=device))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dim, device=device))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

        # the actor and critic are just affine transformations of the
        # shared network's outputs
        self.actor = nn.Linear(
            hidden_dims[-1], n_act, device=device
        )  # outputs logits corresponding to Q-values
        self.critic = nn.Linear(
            hidden_dims[-1], 1, device=device
        )  # outputs the value of the state

        self.device = device

    def get_value(self, x: Float[Tensor, "batch d_obs"]) -> Float[Tensor, "batch"]:
        # return just the value of the state
        return self.critic(self.network(x))

    def forward(
        self, x: Float[Tensor, "batch d_obs"]
    ) -> tuple[Float[Tensor, "batch n_act"], Float[Tensor, "batch"]]:
        # return the action probabilities and the value of the state
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, self.critic(hidden)

    def to(self, device: T.device) -> Self:
        super().to(device)
        self.network.to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.device = device
        return self


class ReplayBuffer:
    def __init__(
        self, size: int, d_obs: int, device: Optional[T.device] = None
    ) -> None:
        self.size = size
        self.obs = T.zeros((size, d_obs), device=device)
        self.act = T.zeros(size, dtype=T.int, device=device)
        self.rew = T.zeros(size, device=device)
        self.val = T.zeros(size, device=device)
        self.act_prob = T.zeros(size, device=device)
        self.dones = T.zeros(size, dtype=T.int, device=device)
        self.device = device

    def fill_with_samples(
        self, env: gym.Env, agent: Agent, max_episode_len: Optional[int] = None
    ):
        """
        Fill the buffer with a batch of trajectory data

        Args:
            env (gym.Env): the environment to sample from
            agent (Agent): the agent to use to sample
            max_episode_len (int): the maximum length of an episode
        """
        with T.no_grad():
            obs = T.tensor(env.reset()[0], device=agent.device)
            done = 0
            current_episode_len = 0
            for step in range(self.size):
                self.obs[step] = obs
                self.dones[step] = done

                # given the observation, figure out what the agent should do
                action_probs, values = agent(obs)
                self.val[step] = values.squeeze()

                if T.isnan(action_probs).any() or T.isinf(action_probs).any():
                    raise ValueError("action probabilities are NaN")

                action = T.distributions.Categorical(action_probs).sample()
                self.act[step] = action
                self.act_prob[step] = action_probs.squeeze()[action]

                # take a step in the env
                obs, rew, done, _, _ = env.step(action.item())
                obs = T.tensor(obs, dtype=T.float32, device=agent.device)
                self.rew[step] = rew
                current_episode_len += 1

                # reset the env if we're done
                if done or (
                    max_episode_len is not None
                    and current_episode_len >= max_episode_len
                ):
                    obs = T.tensor(env.reset()[0], device=agent.device)
                    done = 1
                    current_episode_len = 0

    def to(self, device: T.device) -> Self:
        self.obs = self.obs.to(device)
        self.act = self.act.to(device)
        self.rew = self.rew.to(device)
        self.val = self.val.to(device)
        self.act_prob = self.act_prob.to(device)
        self.dones = self.dones.to(device)
        self.device = device
        return self


def get_advantages(
    buffer: ReplayBuffer, agent: Agent, discount: float, gae_lambda: float
) -> Float[Tensor, "batch"]:
    """
    Estimate the empirical advantages using GAE
    """

    with T.no_grad():
        advantages = T.zeros(buffer.size, device=buffer.device)
        # bootstrap the value of the last state
        if buffer.dones[-1] == 0:  # last state was not terminal
            next_value = agent.get_value(buffer.obs[-1])
        else:  # last state was terminal
            next_value = 0

        for t in reversed(range(buffer.size)):
            # loop in reverse order, compute TD and then compute advantages "recursively"
            td_error = (
                buffer.rew[t]
                + discount * next_value * (1 - buffer.dones[t])
                - buffer.val[t]
            )
            if t == buffer.size - 1:  # base case (ie. no more "recursion")
                advantages[t] = td_error
            else:  # "recursive" case
                advantages[t] = (
                    td_error
                    + discount * gae_lambda * (1 - buffer.dones[t]) * advantages[t + 1]
                )
            next_value = buffer.val[t]

    return advantages


def compute_loss(
    probs_hat: Float[Tensor, "batch n_act"],
    vals_hat: Float[Tensor, "batch"],
    mb_actions: Float[Tensor, "batch"],
    mb_act_prob: Float[Tensor, "batch"],
    mb_advantages: Float[Tensor, "batch"],
    mb_returns: Float[Tensor, "batch"],
    clip_eps: float = 0.2,
    critic_loss_coef: float = 0.5,
    entropy_loss_coef: float = 0.01,
    entropy_eps: float = 1e-6,
    return_tuple: bool = False,
    log_to_wandb: bool = False,
) -> Union[Float[Tensor, ""], Tuple[Float[Tensor, ""], float, float, float]]:
    """
    Compute the PPO loss

    Args:
        probs_hat (Tensor): action probabilities predicted by the actor
        vals_hat (Tensor): values predicted by the critic
        mb_actions (Tensor): actions taken in the minibatch
        mb_act_prob (Tensor): action probabilities of the actions taken
        mb_advantages (Tensor): empirical advantages
        mb_returns (Tensor): empirical returns
        clip_eps (float): PPO clipping parameter
        critic_loss_coef (float): how much to weight the critic loss
        entropy_loss_coef (float): how much to weight the entropy bonus
        entropy_eps (float): small constant to prevent log(0) errors in the entropy loss
        return_tuple (bool): whether to return a tuple of the loss and its components,
            or just the total loss
        log_to_wandb (bool): whether to log the loss to wandb inside this function
            (note that this would be logging minibatch losses, which may be noisy)

    """
    # compute policy ratios between the current policy and the old policy
    ratios = probs_hat[T.arange(probs_hat.size(0)), mb_actions] / mb_act_prob

    # clipped surrogate loss from the PPO paper
    actor_loss1 = ratios * mb_advantages
    actor_loss2 = T.clamp(ratios, min=1 - clip_eps, max=1 + clip_eps) * mb_advantages
    actor_loss = T.min(actor_loss1, actor_loss2).mean()

    # MSE between the vals our critic just returned and the empirical returns
    critic_loss = ((vals_hat[mb_actions] - mb_returns) ** 2).mean()

    # low entropy bonus (regularization)
    entropy_loss = (probs_hat * T.log(probs_hat + entropy_eps)).sum(-1).mean()

    if T.isnan(actor_loss).any() or T.isinf(actor_loss).any():
        raise ValueError("actor loss is NaN")
    if T.isnan(critic_loss).any() or T.isinf(critic_loss).any():
        raise ValueError("critic loss is NaN")
    if T.isnan(entropy_loss).any() or T.isinf(entropy_loss).any():
        raise ValueError("entropy loss is NaN")

    # total loss
    loss = (
        -actor_loss + critic_loss_coef * critic_loss - entropy_loss_coef * entropy_loss
    )

    # log the loss
    if log_to_wandb:
        wandb.log(
            {
                "actor_loss": -actor_loss.item(),  # flipping the sign to make this a loss
                "critic_loss": (critic_loss_coef * critic_loss).item(),
                "entropy_loss": -(entropy_loss_coef * entropy_loss).item(),
                "loss": loss.item(),
            }
        )

    if return_tuple:
        return (
            loss,
            -actor_loss.item(),
            (critic_loss_coef * critic_loss).item(),
            -(entropy_loss_coef * entropy_loss).item(),
        )
    return loss


def train_agent(
    env: gym.Env,
    discount=0.99,
    d_obs=4,
    n_act=2,
    max_episode_len=1024,
    hidden_dims=[32, 8],
    batch_size=4096,  # 2^12
    mini_batch_size=128,  # 2^7
    lr=0.001,
    use_wandb=False,
    # Needs to be a power of 2, otherwise the actual number would
    # depend on the batch size because of integer division, and then
    # the sweep comparisons wouldn't be fair
    # n_timesteps=8388608,  # 2^23,
    n_timesteps=2**16,  # faster for testing
    n_epochs_per_episode=8,  # 2^3
    gae_lambda=0.95,
    clip_eps=0.2,
    critic_loss_coef=0.01,
    entropy_loss_coef=0.1,
    entropy_eps: float = 1e-6,
    monitor_gym: bool = True,
    logging_ep_len_running_avg_len: int = 10,
    logging_ep_len_exp_avg_alpha: float = 0.02,
    logging_loss_running_avg_len: int = 20,
    logging_loss_exp_avg_alpha: float = 0.01,
) -> Agent:
    """
    Train an agent using PPO

    Args:
        env (gym.Env): the environment to train on
        discount (float): discount factor
        d_obs (int): observation dimension
        n_act (int): number of actions (assumes discrete actions)
        max_episode_len (int): maximum episode length
            (among other things, this is useful for making sure hparam
            sweep comparisons are fair — if you're doing a sweep, you
            should set this to be below your smallest batch_size option)
        hidden_dims (list[int]): hidden layer dimensions
        batch_size (int): how many steps to collect before updating
        mini_batch_size (int): how many samples to use in each update
        lr (float): learning rate
        n_timesteps (int): total number of steps to collect
        n_epochs_per_episode (int): how many update epochs to do at the
            end of each episode
        gae_lambda (float): Generalized Advantage Estimation lambda parameter
        clip_eps (float): PPO clipping parameter
        critic_loss_coef (float): how much to weight the critic loss
        entropy_loss_coef (float): how much to weight the entropy bonus
        entropy_eps (float): small constant to prevent log(0) errors in the entropy loss
        monitor_gym (bool): whether to monitor the gym environment with wandb
        logging_ep_len_running_avg_len (int): how many episodes to use in the
            running average of episode length
        logging_ep_len_exp_avg_alpha (float): alpha parameter for the exponential
            moving average of episode length
        logging_loss_running_avg_len (int): how many episodes to use in the
            running average of the loss
        logging_loss_exp_avg_alpha (float): alpha parameter for the exponential
            moving average of the loss

    Returns:
        Agent: the trained agent
    """
    if use_wandb:
        wandb.init(project="explainable-rl-iai", monitor_gym=monitor_gym)
        config = wandb.config
        overwrote_something_from_wandb = False
        if "lr" in config:
            lr = config.lr
            overwrote_something_from_wandb = True
        if "clip_eps" in config:
            clip_eps = config.clip_eps
            overwrote_something_from_wandb = True
        if "batch_size" in config:
            batch_size = config.batch_size
            overwrote_something_from_wandb = True
        if "mini_batch_size" in config:
            mini_batch_size = config.mini_batch_size
            overwrote_something_from_wandb = True
        if "n_epochs_per_episode" in config:
            n_epochs_per_episode = config.n_epochs_per_episode
            overwrote_something_from_wandb = True
            # adjust the number of timesteps to make sure
            # we're performing the same number of gradient updates
            # in each run within a hyperparameter sweep
            n_timesteps = n_timesteps // n_epochs_per_episode
        if overwrote_something_from_wandb:
            print("INFO: Overwrote some hyperparams from wandb config.")

    # the way we do logging (and other hparam sweep stuff) assumes
    # that the batch size is evenly divisible by the mini_batch size
    if batch_size % mini_batch_size != 0:
        print(
            "\n\nWARNING: Your batch_size is not divisible by your \
                mini_batch_size. If you are logging to wandb, the logged loss \
                values will likely be wrong. \
                Please consider adjusting these sizes.\n\n\n"
        )
    # ditto for the number of timesteps by batch size
    if n_timesteps % batch_size != 0:
        print(
            "\n\nWARNING: Your n_timesteps is not divisible by your \
                batch_size. If you are doing a hparam sweep, your results \
                will be skewed because you're comparing runs with different \
                lengths. Please consider adjusting these arguments.\n\n\n"
        )
    # the batch size should be at least as big as max_episode_len
    if max_episode_len > batch_size:
        print(
            "\n\nWARNING: Your max_episode_len is larger than your \
                batch_size. This means that batch_size will effectively \
                become your max_episode_len. If you're doing a hparam sweep, \
                different runs may have different maximum episode lengths, \
                so comparisons of the average episode length will be skewed.\n\n\n"
        )

    # initialize the agent, optimizer, and replay buffer
    agent = Agent(d_obs, n_act, hidden_dims, device=device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    buffer = ReplayBuffer(batch_size, d_obs, device=device)

    n_episodes = n_timesteps // batch_size
    episode_len_exp_avg = MovingAverage(
        ema_alpha=logging_ep_len_exp_avg_alpha,
        # size=logging_ep_len_running_avg_len,
        disable_simple_average=True,
    )
    loss_exp_avg = MovingAverage(
        ema_alpha=logging_loss_exp_avg_alpha,
        # size=logging_loss_running_avg_len,
        disable_simple_average=True,
    )
    for _ in tqdm(range(n_episodes)):  # episode loop
        # fill the buffer with a batch of trajectory data
        buffer.fill_with_samples(env, agent, max_episode_len)

        # log the average episode length and average reward
        episode_len = batch_size / max(1, buffer.dones.sum().item())
        episode_len_exp_avg.append(episode_len)
        if use_wandb:
            wandb.log(
                {
                    "episode_len": episode_len,
                    "episode_len_exp_avg": episode_len_exp_avg.exp_average(),
                    "action": buffer.act.float().mean().item(),
                    "reward": buffer.rew.mean().item(),
                }
            )

        # estimate the empirical advantages and returns using GAE
        advantages = get_advantages(buffer, agent, discount, gae_lambda)
        returns = advantages + buffer.val

        # for logging — we want to log the episodes' average reward to reduce noise
        n_updates = n_epochs_per_episode * (batch_size // mini_batch_size)
        losses = T.empty(n_updates, device=device)
        actor_losses = T.empty(n_updates, device=device)
        critic_losses = T.empty(n_updates, device=device)
        entropy_losses = T.empty(n_updates, device=device)

        # update the networks
        batch_indices = np.arange(batch_size)
        for i_epoch in range(n_epochs_per_episode):  # epoch loop
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, mini_batch_size):  # minibatch loop
                end = start + mini_batch_size
                mb_indices = batch_indices[start:end]

                # get the minibatch data
                mb_obs = buffer.obs[mb_indices]
                mb_actions = buffer.act[mb_indices]
                mb_act_prob = buffer.act_prob[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # run the actor and critic on the minibatch
                probs_hat, vals_hat = agent(mb_obs)

                # compute the loss
                loss, actor_loss, critic_loss, entropy_loss = compute_loss(
                    probs_hat,
                    vals_hat,
                    mb_actions,
                    mb_act_prob,
                    mb_advantages,
                    mb_returns,
                    clip_eps,
                    critic_loss_coef,
                    entropy_loss_coef,
                    entropy_eps,
                    return_tuple=True,
                    log_to_wandb=False,
                )

                # save for logging
                loss_idx = (
                    i_epoch * (batch_size // mini_batch_size) + start // mini_batch_size
                )
                losses[loss_idx] = loss.item()
                actor_losses[loss_idx] = actor_loss
                critic_losses[loss_idx] = critic_loss
                entropy_losses[loss_idx] = entropy_loss

                # update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # log the average loss
        mean_loss = losses.mean().item()
        loss_exp_avg.append(mean_loss)
        if use_wandb:
            wandb.log(
                {
                    "loss": mean_loss,
                    "loss_exp_avg": loss_exp_avg.exp_average(),
                    "actor_loss": actor_losses.mean().item(),
                    "critic_loss": critic_losses.mean().item(),
                    "entropy_loss": entropy_losses.mean().item(),
                }
            )

    return agent


if __name__ == "__main__":
    include_visuals_in_wandb = False
    use_wandb = True

    sweep_config = {
        # 'method': 'random',
        "method": "grid",
        "metric": {
            "name": "episode_len_exp_avg",
            "goal": "maximize",
        },
        "parameters": {
            "batch_size": {
                "values": [2048, 1024, 8192, 32768, 65536],
            },
            # 'mini_batch_size': {
            #     # 'values': [128, 32, 512, 2048],
            #     'values': [64, 128],
            # },
            # 'n_epochs_per_episode': {
            #     'values': [16, 4],
            # },
            "lr": {
                # 'min': 1e-4,
                # 'max': 1e-1,
                # 'distribution': 'log_uniform_values'
                # 'values': [1e-3, 1e-2, 1e-4, 1e-1]
                "values": [1e-3, 1e-4],
            },
            "clip_eps": {
                # 'min': 1e-4,
                # 'max': 1,
                # 'distribution': 'log_uniform_values'
                "values": [0.2, 0.1, 0.5, 1e-2, 1.0, 1e-4],
            },
        },
    }

    # env = ConstructPoleBalancingEnv(
    #     render_mode='rgb_array' if include_visuals_in_wandb else None)
    env = gym.make(
        "CartPole-v1", render_mode="rgb_array" if include_visuals_in_wandb else None
    )
    if include_visuals_in_wandb:
        env = gym.wrappers.RecordVideo(env, "./videos")
    sweep_id = wandb.sweep(sweep_config, project="explainable-rl-iai")
    wandb.agent(
        sweep_id, function=partial(train_agent, env, use_wandb=use_wandb), count=1024
    )

    # train_agent(env,
    #             hidden_dims=[16, 8, 2],
    #             monitor_gym=include_visuals_in_wandb)

    env.close()
