import torch
from torch import nn
import random


"""
   Trainer
"""

class Trainer(object):
    def __init__(self, net, target_net, optimizer, loss_func, replay_buffer, gamma, device):
        self.net = net
        self.target_net = target_net
        self.optimizer = optimizer 
        self.loss_func = loss_func
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.device = device
    
    def update(self, batch_size, beta):
        obs, action, reward, next_obs, done, indices, weights = self.replay_buffer.sample(batch_size, beta)
        obs, action, reward, next_obs, done, weights \
            = obs.float().to(self.device), action.to(self.device), reward.to(self.device), \
              next_obs.float().to(self.device), done.to(self.device), weights.to(self.device)

        q_values = self.net(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN
            greedy_action_next = torch.argmax(self.net(next_obs), dim=1)
            q_values_next = self.target_net(next_obs).gather(1, greedy_action_next.unsqueeze(1)).squeeze(1)

        target_q_values = reward + self.gamma * q_values_next * (1 - done)

        self.optimizer.zero_grad()
        loss = (weights * self.loss_func(q_values, target_q_values)).mean()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, (target_q_values - q_values).abs().detach().cpu().numpy())

        return loss.item()