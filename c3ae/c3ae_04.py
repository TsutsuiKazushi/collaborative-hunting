import sys, os
sys.path.append(os.pardir)
import numpy as np
import time
import random
import torch
from torch import nn, optim
from common.network import DuelingNetwork
from common.replay import PrioritizedReplayBuffer
from common.trainer import Trainer
from common.hparameter import *

""" seed """
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

''' divice '''
device = torch.device('cpu')
print(device)

""" Network """
net_p1 = DuelingNetwork(26, 13).to(device)
target_net_p1 = DuelingNetwork(26, 13).to(device)
optimizer_p1 = optim.Adam(net_p1.parameters(), lr=learning_rate)

net_p2 = DuelingNetwork(26, 13).to(device)
target_net_p2 = DuelingNetwork(26, 13).to(device)
optimizer_p2 = optim.Adam(net_p2.parameters(), lr=learning_rate)

net_p3 = DuelingNetwork(26, 13).to(device)
target_net_p3 = DuelingNetwork(26, 13).to(device)
optimizer_p3 = optim.Adam(net_p3.parameters(), lr=learning_rate)

net_e = DuelingNetwork(26, 13).to(device)
target_net_e = DuelingNetwork(26, 13).to(device)
optimizer_e = optim.Adam(net_e.parameters(), lr=learning_rate)

loss_func = nn.SmoothL1Loss(reduction='none') # Huber loss

""" Replay buffer """
replay_buffer_p1 = PrioritizedReplayBuffer(buffer_size)
replay_buffer_p2 = PrioritizedReplayBuffer(buffer_size)
replay_buffer_p3 = PrioritizedReplayBuffer(buffer_size)
replay_buffer_e = PrioritizedReplayBuffer(buffer_size)
beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))

""" Epsilon """
epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

""" Trainer """
trainer_p1 = Trainer(net_p1, target_net_p1, optimizer_p1, loss_func, replay_buffer_p1, gamma, device)
trainer_p2 = Trainer(net_p2, target_net_p2, optimizer_p2, loss_func, replay_buffer_p2, gamma, device)
trainer_p3 = Trainer(net_p3, target_net_p3, optimizer_p3, loss_func, replay_buffer_p3, gamma, device)
trainer_e = Trainer(net_e, target_net_e, optimizer_e, loss_func, replay_buffer_e, gamma, device)

""" Environment """
from chase3_and_escape import Chase3AndEscape
speed_p = 1.8
speed_e = 3
env = Chase3AndEscape(speed_pursuer1=speed_p, speed_pursuer2=speed_p, speed_pursuer3=speed_p, speed_evader=speed_e, max_step=max_step_episode, reward_share=False)
filename_model_p1 = "../model/c3ae/reward_indiv/p1_04.pth"
filename_model_p2 = "../model/c3ae/reward_indiv/p2_04.pth"
filename_model_p3 = "../model/c3ae/reward_indiv/p3_04.pth"
filename_model_e = "../model/c3ae/reward_indiv/e_04.pth"


print('Start!')
start_time = time.time()

step = 0
for episode in range(num_episodes):
    step_episode = 0
    obs_p1, obs_p2, obs_p3, obs_e = env.reset()
    obs_p1, obs_p2, obs_p3, obs_e = torch.Tensor(obs_p1), torch.Tensor(obs_p2), torch.Tensor(obs_p3), torch.Tensor(obs_e)
    done = False
    total_reward_p1, total_reward_p2, total_reward_p3, total_reward_e = 0, 0, 0, 0
    
    while not done:      
        action_p1 = net_p1.act(obs_p1.float().to(device), epsilon_func(step))
        action_p2 = net_p2.act(obs_p2.float().to(device), epsilon_func(step))
        action_p3 = net_p3.act(obs_p3.float().to(device), epsilon_func(step))
        action_e = net_e.act(obs_e.float().to(device), epsilon_func(step))        
        
        next_obs_p1, next_obs_p2, next_obs_p3, next_obs_e, reward_p1, reward_p2, reward_p3, reward_e, done = env.step(action_p1, action_p2, action_p3, action_e, step_episode)
        next_obs_p1, next_obs_p2, next_obs_p3, next_obs_e = torch.Tensor(next_obs_p1), torch.Tensor(next_obs_p2), torch.Tensor(next_obs_p3), torch.Tensor(next_obs_e)
        
        total_reward_p1 += reward_p1
        total_reward_p2 += reward_p2
        total_reward_p3 += reward_p3
        total_reward_e += reward_e

        replay_buffer_p1.push([obs_p1, action_p1, reward_p1, next_obs_p1, done])
        replay_buffer_p2.push([obs_p2, action_p2, reward_p2, next_obs_p2, done])
        replay_buffer_p3.push([obs_p3, action_p3, reward_p3, next_obs_p3, done])
        replay_buffer_e.push([obs_e, action_e, reward_e, next_obs_e, done])
        
        obs_p1 = next_obs_p1
        obs_p2 = next_obs_p2
        obs_p3 = next_obs_p3
        obs_e = next_obs_e

        if len(replay_buffer_p1) >= initial_buffer_size:
            trainer_p1.update(batch_size, beta_func(step))
            trainer_p2.update(batch_size, beta_func(step))
            trainer_p3.update(batch_size, beta_func(step))
            trainer_e.update(batch_size, beta_func(step))

        if (step + 1) % target_update_interval == 0:
            target_net_p1.load_state_dict(net_p1.state_dict())
            target_net_p2.load_state_dict(net_p2.state_dict())
            target_net_p3.load_state_dict(net_p3.state_dict())
            target_net_e.load_state_dict(net_e.state_dict())
        
        step += 1
        step_episode += 1

    if (episode + 1) % print_interval_episode == 0:
        print('Episode: {},  Step: {}, Episode_step: {},  Reward_p1: {}, Reward_p2: {}, Reward_p3: {}, Reward_e: {}'.format(episode + 1, step + 1, step_episode + 1, round(total_reward_p1, 2), round(total_reward_p2, 2), round(total_reward_p3, 2), round(total_reward_e, 2)))

    if (episode + 1) % save_interval_episode == 0:
        torch.save(net_p1.state_dict(), filename_model_p1)
        torch.save(net_p2.state_dict(), filename_model_p2)
        torch.save(net_p3.state_dict(), filename_model_p3)
        torch.save(net_e.state_dict(), filename_model_e)
        
end_time = time.time()
diff_time = (end_time - start_time)/3600
print('Finish!', round(diff_time, 2), '[h]')