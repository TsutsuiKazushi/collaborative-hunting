import torch
from torch import nn
import random

"""
    Dueling Network
"""
class DuelingNetwork(nn.Module):
    def __init__(self, num_state, num_action):
        super(DuelingNetwork, self).__init__()
        self.num_state = num_state
        self.num_action = num_action

        self.fc_common = nn.Sequential(
            nn.Linear(num_state, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.fc_state = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_action)
        )
    
    def forward(self, obs):
        feature = self.fc_common(obs)
        feature = feature.view(feature.size(0), -1)

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        action_values = state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_values

    def act(self, obs, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.num_action)
        else:
            with torch.no_grad():
                action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        return action