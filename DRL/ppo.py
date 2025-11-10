import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# PPOhyperparameters


PPO_HYPERPARAMS = {
    "actor_hidden_dim": 128,
    "actor_depth": 2,
    "critic_hidden_dim": 64,
    "critic_depth": 2,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "clip_epsilon": 0.2,
    "num_epochs": 10,
    "mini_batch_size": 64,
    "rollout_steps": 2048,
    "value_loss_coef": 0.1,
    "entropy_coef": 0.01,
}   


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, depth=2):
        super(ActorNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() for _ in range(depth - 1)]
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        x = self.feature_extractor(state)
        x = self.classification_head(x)
        return x
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, depth=2):
        super(CriticNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() for _ in range(depth - 1)]
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        x = self.feature_extractor(state)
        x = self.classification_head(x)
        return x


class RolloutBuffer:
    def __init__(self, rollout_steps, gamma):
        self.rollout_steps = rollout_steps
        self.gamma = gamma

        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        self.advantages = None
        self.returns = None

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, next_value):
        """Calculates n-step returns and advantages, storing them in the buffer."""
        
        self.returns = torch.zeros(self.rollout_steps)
        values_tensor = torch.cat(self.values).detach().squeeze()

        R = next_value.item() 
        
        for t in reversed(range(self.rollout_steps)):
            R = self.rewards[t] + self.gamma * R * (1 - self.dones[t])
            self.returns[t] = R
            
        self.advantages = self.returns - values_tensor
        
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batch(self):
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_returns_and_advantages() before get_batch()")
            
        states_tensor = torch.cat(self.states)
        actions_tensor = torch.cat(self.actions)
        old_log_probs_tensor = torch.cat(self.old_log_probs).detach()
        
        return states_tensor, actions_tensor, old_log_probs_tensor, self.returns, self.advantages

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None

def train_ppo(actor_net, critic_net, actor_optimizer, critic_optimizer, env, 
              total_timesteps, rollout_steps=2048, num_update_epochs=10, 
              gamma=0.99, clip_epsilon=0.2, entropy_coeff=0.01):

    buffer = RolloutBuffer(rollout_steps, gamma)

    state = env.reset()
    current_timestep = 0

    while current_timestep < total_timesteps:
        
        for _ in range(rollout_steps):
            current_timestep += 1
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = actor_net(state_tensor)
                value = critic_net(state_tensor)
                m = Categorical(probs)
                action = m.sample()
                log_prob = m.log_prob(action)
            
            next_state, reward, done, _ = env.step(action.item())
            
            buffer.store(state_tensor, action, log_prob, reward, done, value)
            
            if done:
                state = env.reset()
            else:
                state = next_state
        
        
        with torch.no_grad():
            next_value = critic_net(torch.FloatTensor(state).unsqueeze(0))
        
        buffer.compute_returns_and_advantages(next_value)
        
        states_tensor, actions_tensor, old_log_probs_tensor, returns, advantages = buffer.get_batch()
        
        
        for _ in range(num_update_epochs):
            
            probs_new = actor_net(states_tensor)
            values_new = critic_net(states_tensor).squeeze(1)
            m_new = Categorical(probs_new)
            
            new_log_probs = m_new.log_prob(actions_tensor)
            entropy = m_new.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()
            
            # Critic loss uses the calculated n-step returns
            critic_loss = nn.MSELoss()(values_new, returns)
            
            # Optimization
            actor_optimizer.zero_grad()
            (actor_loss + entropy_coeff * entropy_loss).backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        
        buffer.clear()

    return actor_net, critic_net