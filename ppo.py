import numpy as np
import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import wandb
from jaxtyping import Float

class Agent(nn.Module):
    def __init__(self, d_obs: int, n_act: int, hidden_dims: list[int]) -> None:
        super().__init__()

        # create the shared network (common to both actor and critic)
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(nn.Linear(d_obs, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

        # the actor and critic are just affine transformations of the
        # shared network's outputs
        self.actor = nn.Linear(hidden_dims[-1], n_act) # outputs logits corresponding to Q-values
        self.critic = nn.Linear(hidden_dims[-1], 1) # outputs the value of the state
    
    def get_value(self, x: Float[Tensor, "batch d_obs"]) -> Float[Tensor, "batch"]:
        # return just the value of the state
        return self.critic(self.network(x))
    
    def forward(self, x: Float[Tensor, "batch d_obs"]
                ) -> tuple[Float[Tensor, "batch n_act"], Float[Tensor, "batch"]]:
        # return the action probabilities and the value of the state
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, self.critic(hidden)

class ReplayBuffer():
    def __init__(self, size: int, d_obs: int) -> None:
        self.size = size
        self.obs = T.zeros((size, d_obs))
        self.act = T.zeros(size, dtype=T.int)
        self.rew = T.zeros(size)
        self.val = T.zeros(size)
        self.act_prob = T.zeros(size)
        self.dones = T.zeros(size, dtype=T.int)

    def fill_with_samples(self, env: gym.Env, agent: Agent):
        """
            Fill the buffer with a batch of trajectory data
        """
        with T.no_grad():
            obs = T.tensor(env.reset()[0])
            done = 0
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
                obs = T.tensor(obs)
                self.rew[step] = rew

                # reset the env if we're done
                if done:
                    obs = T.tensor(env.reset()[0])
                    done = 1

def get_advantages(buffer: ReplayBuffer, agent: Agent,
                   discount: float, gae_lambda: float) -> Float[Tensor, "batch"]:
    """
        Estimate the empirical advantages using GAE
    """

    with T.no_grad():
        advantages = T.zeros(buffer.size)
        # bootstrap the value of the last state
        if buffer.dones[-1] == 0: # last state was not terminal
            next_value = agent.get_value(buffer.obs[-1])
        else: # last state was terminal
            next_value = 0
        
        for t in reversed(range(buffer.size)):
            # loop in reverse order, compute TD and then compute advantages "recursively"
            td_error = buffer.rew[t] + discount * next_value * (1 - buffer.dones[t]) \
                        - buffer.val[t]
            if t == buffer.size - 1: # base case (ie. no more "recursion")
                advantages[t] = td_error
            else: # "recursive" case
                advantages[t] = td_error + discount * gae_lambda * \
                        (1 - buffer.dones[t]) * advantages[t+1]
            next_value = buffer.val[t]
    
    return advantages

def compute_loss(probs_hat: Float[Tensor, "batch n_act"],
                 vals_hat: Float[Tensor, "batch"],
                 mb_actions: Float[Tensor, "batch"],
                 mb_act_prob: Float[Tensor, "batch"],
                 mb_advantages: Float[Tensor, "batch"],
                 mb_returns: Float[Tensor, "batch"],
                 clip_eps: float = 0.2,
                 critic_loss_coef: float = 0.5,
                 entropy_loss_coef: float = 0.01,
                 entropy_eps: float = 1e-6) -> Float[Tensor, ""]:
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
    """
    # compute policy ratios between the current policy and the old policy
    ratios = probs_hat[T.arange(probs_hat.size(0)), mb_actions] / mb_act_prob

    # clipped surrogate loss from the PPO paper
    actor_loss1 = ratios * mb_advantages
    actor_loss2 = T.clamp(ratios, min=1-clip_eps, max=1+clip_eps) * mb_advantages
    actor_loss = T.min(actor_loss1, actor_loss2).mean()

    # MSE between the vals our critic just returned and the empirical returns
    critic_loss = ((vals_hat[mb_actions] - mb_returns)**2).mean()

    # low entropy bonus (regularization)
    entropy_loss = (probs_hat * T.log(probs_hat + entropy_eps)).sum(-1).mean()

    if T.isnan(actor_loss).any() or T.isinf(actor_loss).any():
        raise ValueError("actor loss is NaN")
    if T.isnan(critic_loss).any() or T.isinf(critic_loss).any():
        raise ValueError("critic loss is NaN")
    if T.isnan(entropy_loss).any() or T.isinf(entropy_loss).any():
        raise ValueError("entropy loss is NaN")

    # total loss
    loss = -actor_loss + critic_loss_coef * critic_loss \
        - entropy_loss_coef * entropy_loss

    # log the loss
    wandb.log({
        "actor_loss": -actor_loss.item(), # flipping the sign to make this a loss
        "critic_loss": (critic_loss_coef * critic_loss).item(),
        "entropy_loss": -(entropy_loss_coef * entropy_loss).item(),
        "total_loss": loss.item(),
    })

    return loss

def train_agent(
        env: gym.Env,
        discount = 0.99,
        d_obs = 4,
        n_act = 2,
        hidden_dims = [64, 128, 32],
        batch_size = 2048,
        mini_batch_size = 128,
        lr = 0.001,
        n_timesteps = 10**6,
        n_epochs_per_episode = 10,
        gae_lambda = 0.95,
        clip_eps = 0.2,
        critic_loss_coef = 0.01,
        entropy_loss_coef = 0.1,
        entropy_eps: float = 1e-6,
        monitor_gym: bool = True,
    ) -> Agent:
    """
        Train an agent using PPO

        Args:
            env (gym.Env): the environment to train on
            discount (float): discount factor
            d_obs (int): observation dimension
            n_act (int): number of actions (assumes discrete actions)
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

        Returns:
            Agent: the trained agent
    """

    # initialize the agent, optimizer, and replay buffer
    agent = Agent(d_obs, n_act, hidden_dims)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    buffer = ReplayBuffer(batch_size, d_obs)

    # initialize wandb
    run = wandb.init(
        project="explainable-rl-iai",
        config={
            "discount": discount,
            "d_obs": d_obs,
            "n_act": n_act,
            "hidden_dims": hidden_dims,
            "batch_size": batch_size,
            "mini_batch_size": mini_batch_size,
            "lr": lr,
            "n_timesteps": n_timesteps,
            "n_epochs_per_episode": n_epochs_per_episode,
            "gae_lambda": gae_lambda,
            "clip_eps": clip_eps,
            "critic_loss_coef": critic_loss_coef,
            "entropy_loss_coef": entropy_loss_coef, 
            "entropy_eps": entropy_eps,
        },
        monitor_gym=monitor_gym,
    )

    n_episodes = n_timesteps // batch_size
    for _ in range(n_episodes): # episode loop
        # fill the buffer with a batch of trajectory data
        buffer.fill_with_samples(env, agent)

        # log the average episode length and average reward
        wandb.log({
            "avg_episode_length": batch_size / max(1, buffer.dones.sum().item()),
            "avg_action": buffer.act.float().mean().item(),
        })
        
        # estimate the empirical advantages and returns using GAE
        advantages = get_advantages(buffer, agent, discount, gae_lambda)
        returns = advantages + buffer.val

        # update the networks
        batch_indices = np.arange(batch_size)
        for _ in range(n_epochs_per_episode): # epoch loop
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, mini_batch_size): # minibatch loop
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
                loss = compute_loss(probs_hat, vals_hat, mb_actions, mb_act_prob,
                                    mb_advantages, mb_returns, clip_eps,
                                    critic_loss_coef, entropy_loss_coef, entropy_eps)
                

                # update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    return agent

if __name__ == '__main__':
    include_visuals_in_wandb = False

    env = gym.make('CartPole-v1', render_mode='rgb_array' if include_visuals_in_wandb else None)
    if include_visuals_in_wandb:
        env = gym.wrappers.RecordVideo(env, "./videos")
    train_agent(env, monitor_gym=include_visuals_in_wandb)
