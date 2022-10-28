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
import argparse


def parse_option():
    """ argparse """
    parser = argparse.ArgumentParser()
    parser.add_argument('speed_p', choices=['4.2', '3.6', '3.0', '2.4', '1.8'])

    args = parser.parse_args()
    
    return args


def main():

    print('Start!')
    start_time = time.time()

    step = 0
    for episode in range(num_episodes):
        step_episode = 0
        obs_p, obs_e = env.reset()
        obs_p, obs_e = torch.Tensor(obs_p), torch.Tensor(obs_e)
        done = False
        total_reward_p = 0
        total_reward_e = 0
        
        while not done:      
            action_p = net_p.act(obs_p.float().to(device), epsilon_func(step))
            action_e = net_e.act(obs_e.float().to(device), epsilon_func(step))        
            
            next_obs_p, next_obs_e, reward_p, reward_e, done = env.step(action_p, action_e, step_episode)        
            next_obs_p, next_obs_e = torch.Tensor(next_obs_p), torch.Tensor(next_obs_e)
            
            total_reward_p += reward_p
            total_reward_e += reward_e

            replay_buffer_p.push([obs_p, action_p, reward_p, next_obs_p, done])
            replay_buffer_e.push([obs_e, action_e, reward_e, next_obs_e, done])
            
            obs_p = next_obs_p
            obs_e = next_obs_e

            if len(replay_buffer_p) >= initial_buffer_size:
                trainer_p.update(batch_size, beta_func(step))
                trainer_e.update(batch_size, beta_func(step))

            if (step + 1) % target_update_interval == 0:
                target_net_p.load_state_dict(net_p.state_dict())
                target_net_e.load_state_dict(net_e.state_dict())
            
            step += 1
            step_episode += 1

        if (episode + 1) % print_interval_episode == 0:
            print('Episode: {},  Step: {}, Episode_step: {},  Reward_p: {}, Reward_e: {}'.format(episode + 1, step + 1, step_episode + 1, total_reward_p, total_reward_e))
            
        if (episode + 1) % save_interval_episode == 0:
            torch.save(net_p.state_dict(), filename_model_p)
            torch.save(net_e.state_dict(), filename_model_e)

    end_time = time.time()
    diff_time = (end_time - start_time)/3600
    print('Finish!', round(diff_time, 2), '[h]')


if __name__ == '__main__':

    """ argparse """
    args = parse_option()

    """ seed """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    """ divice """
    device = torch.device('cpu')
    print(device)

    """ Network """
    net_p = DuelingNetwork(10, 13).to(device)
    target_net_p = DuelingNetwork(10, 13).to(device)
    optimizer_p = optim.Adam(net_p.parameters(), lr=learning_rate)

    net_e = DuelingNetwork(10, 13).to(device)
    target_net_e = DuelingNetwork(10, 13).to(device)
    optimizer_e = optim.Adam(net_e.parameters(), lr=learning_rate)

    loss_func = nn.SmoothL1Loss(reduction='none') # Huber loss

    """ Replay buffer """
    replay_buffer_p = PrioritizedReplayBuffer(buffer_size)
    replay_buffer_e = PrioritizedReplayBuffer(buffer_size)
    beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))

    """ Epsilon """
    epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

    """ Trainer """
    trainer_p = Trainer(net_p, target_net_p, optimizer_p, loss_func, replay_buffer_p, gamma, device)
    trainer_e = Trainer(net_e, target_net_e, optimizer_e, loss_func, replay_buffer_e, gamma, device)

    """ Environment """
    from chase1_and_escape import Chase1AndEscape
    speed_p = float(args.speed_p)
    speed_e = 3
    env = Chase1AndEscape(speed_pursuer=speed_p, speed_evader=speed_e, max_step=max_step_episode)
    filename_model_p = "../model/c1ae/p_" + args.speed_p + ".pth"
    filename_model_e = "../model/c1ae/e_" + args.speed_p + ".pth"

    main()