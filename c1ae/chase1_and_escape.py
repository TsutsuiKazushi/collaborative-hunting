import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import *


class Chase1AndEscape:
    
    def __init__(self, speed_pursuer, speed_evader, mass_pursuer=1,  mass_evader=1, damping=0.25, dt=0.1, max_step=300):
        
        self.speed_p = speed_pursuer
        self.speed_e = speed_evader
        self.mass_p = mass_pursuer     
        self.mass_e = mass_evader
        self.damping = damping
        self.dt = dt
        self.max_step = max_step
        
    
    def reset(self):
        self.pos_p = np.random.uniform(-0.5, 0.5, 2)
        self.vel_p = np.zeros(2)
        self.pos_e = np.random.uniform(-0.5, 0.5, 2)
        self.vel_e = np.zeros(2)
        
        obs_p = get_obs_p(self.pos_p, self.vel_p, self.pos_e, self.vel_e)
        obs_e = get_obs_e(self.pos_e, self.vel_e, self.pos_p, self.vel_p)
               
        return obs_p, obs_e
    
    
    def step(self, action_p, action_e, num_step):
        
        abs_u_p = get_abs_u(action_p, self.pos_p, self.pos_e)
        next_pos_p, next_vel_p = get_next_own_state(self.pos_p, self.vel_p, abs_u_p, \
                                                                self.mass_p, self.speed_p, self.damping, self.dt) 
     
        abs_u_e = get_abs_u(action_e, self.pos_e, self.pos_p)
        next_pos_e, next_vel_e = get_next_own_state(self.pos_e, self.vel_e, abs_u_e, \
                                                              self.mass_e, self.speed_e, self.damping, self.dt)
    
        next_obs_p = get_obs_p(next_pos_p, next_vel_p, next_pos_e, next_vel_e)
        next_obs_e = get_obs_e(next_pos_e, next_vel_e, next_pos_p, next_vel_p)
                        
        reward_p = get_reward_pursuer(next_pos_p, next_pos_e)
        reward_e = get_reward_evader(next_pos_e, next_pos_p)
        # reward_p = get_reward_pursuer(next_pos_p, next_pos_e, action_p)
        # reward_e = get_reward_evader(next_pos_e, next_pos_p, action_e)

        done = get_done(next_pos_p, next_pos_e, num_step, self.max_step)
        
        self.pos_p = next_pos_p
        self.vel_p = next_vel_p
        self.pos_e = next_pos_e
        self.vel_e = next_vel_e
        
        return next_obs_p, next_obs_e, reward_p, reward_e, done


def get_obs_p(pos_p, vel_p, pos_e, vel_e):
 
    sub_pos_adv = get_sub_pos(pos_p, pos_e)
    sub_vel_own = get_sub_vel(pos_p, pos_e, vel_p)
    sub_vel_adv = get_sub_vel(pos_p, pos_e, vel_e)
    obs_p = np.concatenate([pos_p] + [sub_vel_own] + \
                           [pos_e] + [sub_pos_adv] + [sub_vel_adv]).reshape(1,10)
    
    return obs_p


def get_obs_e(pos_e, vel_e, pos_p, vel_p):
    
    sub_pos_adv = get_sub_pos(pos_e, pos_p)
    sub_vel_own = get_sub_vel(pos_e, pos_p, vel_e)
    sub_vel_adv = get_sub_vel(pos_e, pos_p, vel_p)
    obs_e = np.concatenate([pos_e] + [sub_vel_own] + \
                           [pos_p] + [sub_pos_adv] + [sub_vel_adv]).reshape(1,10)
    
    return obs_e


def get_reward_pursuer(abs_pos_own, abs_pos_adv):
    dist = get_dist(abs_pos_own, abs_pos_adv)
    reward = 0

    if dist < 0.1:
        reward = 1        
    elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
        reward = -1

    return reward


def get_reward_evader(abs_pos_own, abs_pos_adv):
    dist = get_dist(abs_pos_own, abs_pos_adv)
    reward = 0

    if dist < 0.1:
        reward = -1
    elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
        reward = -1

    return reward

# def get_reward_pursuer(abs_pos_own, abs_pos_adv, action):
#     dist = get_dist(abs_pos_own, abs_pos_adv)
#     reward = 0

#     if action <= 11:
#         reward = -0.001

#     if dist < 0.1:
#         reward = 1        
#     elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
#         reward = -1

#     return reward


# def get_reward_evader(abs_pos_own, abs_pos_adv, action):
#     dist = get_dist(abs_pos_own, abs_pos_adv)
#     reward = 0

#     if action <= 11:
#         reward = -0.001

#     if dist < 0.1:
#         reward = -1
#     elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
#         reward = -1

#     return reward


def get_done(abs_pos_own, abs_pos_adv, num_step, max_step):
    dist = get_dist(abs_pos_own, abs_pos_adv)

    if dist < 0.1 or num_step > max_step or \
      abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1 or \
      abs_pos_adv[0] < -1 or abs_pos_adv[1] < -1 or abs_pos_adv[0] > 1 or abs_pos_adv[1] > 1:        
        done = True
    else:
        done = False

    return done
