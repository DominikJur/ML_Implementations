import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    
def train_a2c(actor_net, critic_net, actor_optimizer, critic_optimizer, env, num_episodes, gamma=0.99):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = actor_net(state_tensor)
            value = critic_net(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            log_probs.append(m.log_prob(action))
            values.append(value)
            rewards.append(reward)
            state = next_state

        # Compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        advantages = returns - values.detach()

        # Update actor
        actor_loss = -(log_probs * advantages).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update critic
        critic_loss = nn.MSELoss()(values, returns)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")
    return actor_net, critic_net