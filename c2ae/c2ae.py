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
    parser.add_argument('speed_p', choices=['3.6', '3.0', '2.4'])
    parser.add_argument('--in_sh', choices=['indiv', 'share'], default='indiv')

    args = parser.parse_args()
    
    return args


def main():

    print('Start!')
    start_time = time.time()

    step = 0
    for episode in range(num_episodes):
        step_episode = 0
        obs_p1, obs_p2, obs_e = env.reset()
        obs_p1, obs_p2, obs_e = torch.Tensor(obs_p1), torch.Tensor(obs_p2), torch.Tensor(obs_e)
        done = False
        total_reward_p1, total_reward_p2, total_reward_e = 0, 0, 0
        
        while not done:      
            action_p1 = net_p1.act(obs_p1.float().to(device), epsilon_func(step))
            action_p2 = net_p2.act(obs_p2.float().to(device), epsilon_func(step))
            action_e = net_e.act(obs_e.float().to(device), epsilon_func(step))        
            
            next_obs_p1, next_obs_p2, next_obs_e, reward_p1, reward_p2, reward_e, done = env.step(action_p1, action_p2, action_e, step_episode)
            next_obs_p1, next_obs_p2, next_obs_e = torch.Tensor(next_obs_p1), torch.Tensor(next_obs_p2), torch.Tensor(next_obs_e)
            
            total_reward_p1 += reward_p1
            total_reward_p2 += reward_p2
            total_reward_e += reward_e

            replay_buffer_p1.push([obs_p1, action_p1, reward_p1, next_obs_p1, done])
            replay_buffer_p2.push([obs_p2, action_p2, reward_p2, next_obs_p2, done])
            replay_buffer_e.push([obs_e, action_e, reward_e, next_obs_e, done])
            
            obs_p1 = next_obs_p1
            obs_p2 = next_obs_p2
            obs_e = next_obs_e

            if len(replay_buffer_e) >= initial_buffer_size:
                trainer_p1.update(batch_size, beta_func(step))
                trainer_p2.update(batch_size, beta_func(step))
                trainer_e.update(batch_size, beta_func(step))

            if (step + 1) % target_update_interval == 0:
                target_net_p1.load_state_dict(net_p1.state_dict())
                target_net_p2.load_state_dict(net_p2.state_dict())
                target_net_e.load_state_dict(net_e.state_dict())
            
            step += 1
            step_episode += 1

        if (episode + 1) % print_interval_episode == 0:
            print('Episode: {},  Step: {}, Episode_step: {},  Reward_p1: {}, Reward_p2: {}, Reward_e: {}'.format(episode + 1, step + 1, step_episode + 1, total_reward_p1, total_reward_p2, total_reward_e))

        if (episode + 1) % save_interval_episode == 0:
            torch.save(net_p1.state_dict(), filename_model_p1)
            torch.save(net_p2.state_dict(), filename_model_p2)
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

    ''' divice '''
    device = torch.device('cpu')
    print(device)

    """ Network """
    net_p1 = DuelingNetwork(18, 13).to(device)
    target_net_p1 = DuelingNetwork(18, 13).to(device)
    optimizer_p1 = optim.Adam(net_p1.parameters(), lr=learning_rate)

    net_p2 = DuelingNetwork(18, 13).to(device)
    target_net_p2 = DuelingNetwork(18, 13).to(device)
    optimizer_p2 = optim.Adam(net_p2.parameters(), lr=learning_rate)

    net_e = DuelingNetwork(18, 13).to(device)
    target_net_e = DuelingNetwork(18, 13).to(device)
    optimizer_e = optim.Adam(net_e.parameters(), lr=learning_rate)

    loss_func = nn.SmoothL1Loss(reduction='none') # Huber loss

    """ Replay buffer """
    replay_buffer_p1 = PrioritizedReplayBuffer(buffer_size)
    replay_buffer_p2 = PrioritizedReplayBuffer(buffer_size)
    replay_buffer_e = PrioritizedReplayBuffer(buffer_size)
    beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))

    """ Epsilon """
    epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

    """ Trainer """
    trainer_p1 = Trainer(net_p1, target_net_p1, optimizer_p1, loss_func, replay_buffer_p1, gamma, device)
    trainer_p2 = Trainer(net_p2, target_net_p2, optimizer_p2, loss_func, replay_buffer_p2, gamma, device)
    trainer_e = Trainer(net_e, target_net_e, optimizer_e, loss_func, replay_buffer_e, gamma, device)

    """ Environment """
    from chase2_and_escape import Chase2AndEscape
    speed_p = float(args.speed_p)
    speed_e = 3

    if args.in_sh=='indiv':
        env = Chase2AndEscape(speed_pursuer1=speed_p, speed_pursuer2=speed_p, speed_evader=speed_e, max_step=max_step_episode, reward_share=False)
    elif args.in_sh=='share':
        env = Chase2AndEscape(speed_pursuer1=speed_p, speed_pursuer2=speed_p, speed_evader=speed_e, max_step=max_step_episode, reward_share=True)

    filename_model_p1 = "../model/c2ae/reward_" + args.in_sh + "/p1_" + args.speed_p + ".pth"
    filename_model_p2 = "../model/c2ae/reward_" + args.in_sh + "/p2_" + args.speed_p + ".pth"
    filename_model_e = "../model/c2ae/reward_" + args.in_sh + "/e_" + args.speed_p + ".pth"

    main()
