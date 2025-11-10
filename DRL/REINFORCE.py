import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import numpy as np

# REINFORCE
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
def select_action(policy_net, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_net(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def train_REINFORCE(policy_net, optimizer, env, num_episodes, gamma=0.99):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            action, log_prob = select_action(policy_net, state)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        # Update policy
        returns = torch.FloatTensor(returns)
        log_probs = torch.cat(log_probs)
        loss = -log_probs * returns
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")
    return policy_net