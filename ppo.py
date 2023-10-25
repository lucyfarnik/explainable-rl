import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

# TODO: TEST EVERYTHING
# TODO: make this all work with non-gym envs

class Agent(nn.Module):
    def __init__(self, d_obs: int, d_act: int, hidden_dims: list[int]) -> None:
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
        self.actor = nn.Linear(hidden_dims[1], d_act)
        self.critic = nn.Linear(hidden_dims[1], 1)
    
    def get_value(self, x: T.Tensor) -> T.Tensor:
        return self.critic(self.network(x))
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, self.critic(hidden)

class ReplayBuffer():
    def __init__(self, size: int, d_obs: int, d_act: int) -> None:
        self.size = size
        self.obs = T.zeros((size, d_obs))
        self.act = T.zeros((size, d_act)) #! FIXME this is assuming continuous actions
        self.rew = T.zeros(size)
        self.val = T.zeros(size)
        self.act_prob = T.zeros(size)
        self.dones = T.zeros(size)

    def fill_with_samples(self, env: gym.Env, agent: Agent):
        """
            Fill the buffer with a batch of trajectory data
        """
        with T.no_grad():
            obs = T.tensor(env.reset())
            done = T.zeros(1)
            for step in range(self.size):
                self.obs[step] = obs
                self.dones[step] = done

                # given the observation, figure out what the agent should do
                action_probs, values = agent(obs)
                self.val[step] = values.squeeze()
                action = T.distributions.Categorical(action_probs).sample()
                self.act[step] = action
                self.act_prob[step] = action_probs.squeeze()[action]

                # take a step in the env
                obs, rew, done, _ = env.step(action)
                self.rew[step] = rew

                if done:
                    obs = T.tensor(env.reset())
                    done = T.zeros(1)

def get_advantages(buffer: ReplayBuffer, agent: Agent,
                   discount: float, gae_lambda: float) -> T.Tensor:
    """
        Estimate the empirical advantages using GAE
    """

    with T.no_grad():
        advantages = T.zeros(buffer.size)
        # bootstrap the value of the last state
        if buffer.dones[-1] == 0: # last state was not terminal
            next_value = agent.get_value(buffer.obs[-1])[buffer.act[-1]]
        else: # last state was terminal
            next_value = 0
        
        # TODO vectorize this for loop to make GPU go brrr
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

def compute_loss(probs_hat: T.Tensor,
                 vals_hat: T.Tensor,
                 mb_actions: T.Tensor,
                 mb_act_prob: T.Tensor,
                 mb_advantages: T.Tensor,
                 mb_returns: T.Tensor,
                 clip_eps: float = 0.2,
                 critic_loss_coef: float = 0.5,
                 entropy_loss_coef: float = 0.01) -> T.Tensor:
    """
        Compute the PPO loss

        Args:
            probs_hat (T.Tensor): action probabilities predicted by the actor
            vals_hat (T.Tensor): values predicted by the critic
            mb_actions (T.Tensor): actions taken in the minibatch
            mb_act_prob (T.Tensor): action probabilities of the actions taken
            mb_advantages (T.Tensor): empirical advantages
            mb_returns (T.Tensor): empirical returns
            clip_eps (float): PPO clipping parameter
            critic_loss_coef (float): how much to weight the critic loss
            entropy_loss_coef (float): how much to weight the entropy bonus
    """
    # compute policy ratios between the current policy and the old policy
    ratios = probs_hat[mb_actions] / mb_act_prob

    # clipped surrogate loss from the PPO paper
    actor_loss1 = ratios * mb_advantages
    actor_loss2 = T.clamp(ratios, min=1-clip_eps, max=1+clip_eps) * mb_advantages
    actor_loss = T.min(actor_loss1, actor_loss2).mean()

    # MSE between the vals our critic just returned and the empirical returns
    critic_loss = ((vals_hat[mb_actions] - mb_returns)**2).mean()

    # low entropy bonus (regularization)
    entropy_loss = (probs_hat * T.log(probs_hat)).sum(-1).mean()

    # total loss
    loss = -actor_loss + critic_loss_coef * critic_loss \
        - entropy_loss_coef * entropy_loss

    return loss

def train_agent(
        env: gym.Env,
        discount = 0.99,
        d_obs = 4,
        d_act = 2,
        hidden_dims = [64, 64],
        batch_size = 512,
        mini_batch_size = 64,
        lr = 0.001,
        n_timesteps = 32000,
        n_epochs_per_episode = 10,
        gae_lambda = 0.95,
        clip_eps = 0.2,
        critic_loss_coef = 0.5,
        entropy_loss_coef = 0.01,
    ) -> Agent:
    """
        Train an agent using PPO

        Args:
            env (gym.Env): the environment to train on
            discount (float): discount factor
            d_obs (int): observation dimension
            d_act (int): action dimension
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

        Returns:
            Agent: the trained agent
    """

    # initialize the agent, optimizer, and replay buffer
    agent = Agent(d_obs, d_act, hidden_dims)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    buffer = ReplayBuffer(batch_size, d_obs, d_act)

    n_episodes = n_timesteps // batch_size
    for _ in range(n_episodes): # episode loop
        # fill the buffer with a batch of trajectory data
        buffer.fill_with_samples(env, agent)
            
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
                                    critic_loss_coef, entropy_loss_coef)

                # update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    return agent

if __name__ == '__main__':
    train_agent()
