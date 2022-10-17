import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import *


class Chase3AndEscape:
    
    def __init__(self, speed_pursuer1, speed_pursuer2, speed_pursuer3, speed_evader, mass_pursuer1=1, mass_pursuer2=1, mass_pursuer3=1, mass_evader=1, \
                 damping=0.25, dt=0.1, max_step=300, reward_share=True):
        
        self.speed_p1 = speed_pursuer1
        self.speed_p2 = speed_pursuer2
        self.speed_p3 = speed_pursuer3
        self.speed_e = speed_evader
        self.mass_p1 = mass_pursuer1
        self.mass_p2 = mass_pursuer2
        self.mass_p3 = mass_pursuer3
        self.mass_e = mass_evader
        self.damping = damping
        self.dt = dt
        self.max_step = max_step
        self.reward_share = reward_share
        
    
    def reset(self):
        self.pos_p1 = np.random.uniform(-0.5, 0.5, 2)
        self.vel_p1 = np.zeros(2)
        self.pos_p2 = np.random.uniform(-0.5, 0.5, 2)
        self.vel_p2 = np.zeros(2)
        self.pos_p3 = np.random.uniform(-0.5, 0.5, 2)
        self.vel_p3 = np.zeros(2)
        self.pos_e = np.random.uniform(-0.5, 0.5, 2)
        self.vel_e = np.zeros(2)
        
        obs_p1 = get_obs_p(self.pos_p1, self.vel_p1, self.pos_p2, self.vel_p2, self.pos_p3, self.vel_p3, self.pos_e, self.vel_e)
        obs_p2 = get_obs_p(self.pos_p2, self.vel_p2, self.pos_p3, self.vel_p3, self.pos_p1, self.vel_p1, self.pos_e, self.vel_e)
        obs_p3 = get_obs_p(self.pos_p3, self.vel_p3, self.pos_p1, self.vel_p1, self.pos_p2, self.vel_p2, self.pos_e, self.vel_e)
        obs_e = get_obs_e(self.pos_e, self.vel_e, self.pos_p1, self.vel_p1, self.pos_p2, self.vel_p2, self.pos_p3, self.vel_p3)
                
        return obs_p1, obs_p2, obs_p3, obs_e
    
    
    def step(self, action_p1, action_p2, action_p3, action_e, num_step):
        
        abs_u_p1 = get_abs_u(action_p1, self.pos_p1, self.pos_e)
        next_pos_p1, next_vel_p1 = get_next_own_state(self.pos_p1, self.vel_p1, abs_u_p1, \
                                                      self.mass_p1, self.speed_p1, self.damping, self.dt) 
        
        abs_u_p2 = get_abs_u(action_p2, self.pos_p2, self.pos_e)
        next_pos_p2, next_vel_p2 = get_next_own_state(self.pos_p2, self.vel_p2, abs_u_p2, \
                                                      self.mass_p2, self.speed_p2, self.damping, self.dt)
        
        abs_u_p3 = get_abs_u(action_p3, self.pos_p3, self.pos_e)
        next_pos_p3, next_vel_p3 = get_next_own_state(self.pos_p3, self.vel_p3, abs_u_p3, \
                                                      self.mass_p3, self.speed_p3, self.damping, self.dt)
     
        pos_adv1, _, _, _, _, _ = get_order_adv(self.pos_e, self.pos_p1, self.vel_p1, self.pos_p2, self.vel_p2, self.pos_p3, self.vel_p3)
        abs_u_e = get_abs_u(action_e, self.pos_e, pos_adv1)
        next_pos_e, next_vel_e = get_next_own_state(self.pos_e, self.vel_e, abs_u_e, \
                                                    self.mass_e, self.speed_e, self.damping, self.dt)
        
        next_obs_p1 = get_obs_p(next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2, next_pos_p3, next_vel_p3, next_pos_e, next_vel_e)
        next_obs_p2 = get_obs_p(next_pos_p2, next_vel_p2, next_pos_p1, next_vel_p1, next_pos_p3, next_vel_p3, next_pos_e, next_vel_e)
        next_obs_p3 = get_obs_p(next_pos_p3, next_vel_p3, next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2, next_pos_e, next_vel_e)
        next_obs_e = get_obs_e(next_pos_e, next_vel_e, next_pos_p1, next_vel_p1, next_pos_p2, next_vel_p2, next_pos_p3, next_vel_p3)
                        
        reward_p1 = get_reward_pursuer(next_pos_p1, next_pos_p2, next_pos_p3, next_pos_e, self.reward_share)
        reward_p2 = get_reward_pursuer(next_pos_p2, next_pos_p1, next_pos_p3, next_pos_e, self.reward_share)
        reward_p3 = get_reward_pursuer(next_pos_p3, next_pos_p1, next_pos_p2, next_pos_e, self.reward_share)
        reward_e = get_reward_evader(next_pos_e, next_pos_p1, next_pos_p2, next_pos_p3)
        
        done = get_done(next_pos_e, next_pos_p1, next_pos_p2, next_pos_p3, num_step, self.max_step)
        
        self.pos_p1 = next_pos_p1
        self.vel_p1 = next_vel_p1
        self.pos_p2 = next_pos_p2
        self.vel_p2 = next_vel_p2
        self.pos_p3 = next_pos_p3
        self.vel_p3 = next_vel_p3
        self.pos_e = next_pos_e
        self.vel_e = next_vel_e
        
        return next_obs_p1, next_obs_p2, next_obs_p3, next_obs_e, reward_p1, reward_p2, reward_p3, reward_e, done
        

def get_obs_p(pos_p1, vel_p1, pos_p2_tmp, vel_p2_tmp, pos_p3_tmp, vel_p3_tmp, pos_e, vel_e):
 
    pos_p2, vel_p2, pos_p3, vel_p3 = get_order_mate(pos_p1, pos_p2_tmp, vel_p2_tmp, pos_p3_tmp, vel_p3_tmp)
    
    sub_pos_mate1 = get_sub_pos(pos_p1, pos_p2)
    sub_pos_mate2 = get_sub_pos(pos_p1, pos_p3)
    sub_pos_adv = get_sub_pos(pos_p1, pos_e)
    
    sub_vel_own_mate1 = get_sub_vel(pos_p1, pos_p2, vel_p1)
    sub_vel_own_mate2 = get_sub_vel(pos_p1, pos_p3, vel_p1)
    sub_vel_own_adv = get_sub_vel(pos_p1, pos_e, vel_p1)
    
    sub_vel_mate1 = get_sub_vel(pos_p1, pos_p2, vel_p2)
    sub_vel_mate2 = get_sub_vel(pos_p1, pos_p3, vel_p3)
    sub_vel_adv = get_sub_vel(pos_p1, pos_e, vel_e)
               
    obs_p = np.concatenate([pos_p1] + [sub_vel_own_mate1] + [sub_vel_own_mate2] + [sub_vel_own_adv] + \
                           [pos_p2] + [sub_pos_mate1] + [sub_vel_mate1] + \
                           [pos_p3] + [sub_pos_mate2] + [sub_vel_mate2] + \
                           [pos_e] + [sub_pos_adv] + [sub_vel_adv]).reshape(1,26)
    return obs_p


def get_obs_e(pos_e, vel_e, pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp, pos_p3_tmp, vel_p3_tmp):
    
    pos_p1, vel_p1, pos_p2, vel_p2, pos_p3, vel_p3 = get_order_adv(pos_e, pos_p1_tmp, vel_p1_tmp, pos_p2_tmp, vel_p2_tmp, pos_p3_tmp, vel_p3_tmp)
    
    sub_pos_adv1 = get_sub_pos(pos_e, pos_p1)
    sub_pos_adv2 = get_sub_pos(pos_e, pos_p2)
    sub_pos_adv3 = get_sub_pos(pos_e, pos_p3)
    
    sub_vel_own_adv1 = get_sub_vel(pos_e, pos_p1, vel_e)
    sub_vel_own_adv2 = get_sub_vel(pos_e, pos_p2, vel_e)
    sub_vel_own_adv3 = get_sub_vel(pos_e, pos_p3, vel_e)
        
    sub_vel_adv1 = get_sub_vel(pos_e, pos_p1, vel_p1)
    sub_vel_adv2 = get_sub_vel(pos_e, pos_p2, vel_p2)
    sub_vel_adv3 = get_sub_vel(pos_e, pos_p3, vel_p3)
               
    obs_e = np.concatenate([pos_e] + [sub_vel_own_adv1] + [sub_vel_own_adv2] + [sub_vel_own_adv3] +\
                           [pos_p1] + [sub_pos_adv1] + [sub_vel_adv1] + \
                           [pos_p2] + [sub_pos_adv2] + [sub_vel_adv2] + \
                           [pos_p3] + [sub_pos_adv3] + [sub_vel_adv3]).reshape(1,26)
    
    return obs_e


def get_order_mate(pos_own, pos_mate1_tmp, vel_mate1_tmp, pos_mate2_tmp, vel_mate2_tmp):
    dist1 = get_dist(pos_own, pos_mate1_tmp)
    dist2 = get_dist(pos_own, pos_mate2_tmp)

    d = [dist1, dist2]
    p = [pos_mate1_tmp, pos_mate2_tmp]
    v = [vel_mate1_tmp, vel_mate2_tmp]
    l = list(zip(d, p, v))
    l.sort()
    d, p, v = zip(*l)
    
    pos_mate1, vel_mate1 = p[0], v[0] 
    pos_mate2, vel_mate2 = p[1], v[1]
        
    return pos_mate1, vel_mate1, pos_mate2, vel_mate2


def get_order_adv(pos_own, pos_adv1_tmp, vel_adv1_tmp, pos_adv2_tmp, vel_adv2_tmp, pos_adv3_tmp, vel_adv3_tmp):
    dist1 = get_dist(pos_own, pos_adv1_tmp)
    dist2 = get_dist(pos_own, pos_adv2_tmp)
    dist3 = get_dist(pos_own, pos_adv3_tmp)

    d = [dist1, dist2, dist3]
    p = [pos_adv1_tmp, pos_adv2_tmp, pos_adv3_tmp]
    v = [vel_adv1_tmp, vel_adv2_tmp, vel_adv3_tmp]
    l = list(zip(d, p, v))
    l.sort()
    d, p, v = zip(*l)
    
    pos_adv1, vel_adv1 = p[0], v[0] 
    pos_adv2, vel_adv2 = p[1], v[1]
    pos_adv3, vel_adv3 = p[2], v[2]
        
    return pos_adv1, vel_adv1, pos_adv2, vel_adv2, pos_adv3, vel_adv3


def get_reward_pursuer(abs_pos_own, abs_pos_mate1, abs_pos_mate2, abs_pos_adv, reward_share):
    dist1 = get_dist(abs_pos_own, abs_pos_adv)
    dist2 = get_dist(abs_pos_mate1, abs_pos_adv)
    dist3 = get_dist(abs_pos_mate2, abs_pos_adv)
    reward = 0

    if reward_share == True:
        if dist1 < 0.1 or dist2 < 0.1 or dist3 < 0.1:
            reward = 1
        elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
            reward = -1
    elif reward_share == False:
        if dist1 < 0.1:
            reward = 1
        elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
            reward = -1

    return reward


def get_reward_evader(abs_pos_own, abs_pos_adv1, abs_pos_adv2, abs_pos_adv3):
    dist1 = get_dist(abs_pos_own, abs_pos_adv1)
    dist2 = get_dist(abs_pos_own, abs_pos_adv2)
    dist3 = get_dist(abs_pos_own, abs_pos_adv3)
    reward = 0

    if dist1 < 0.1 or dist2 < 0.1 or dist3 < 0.1:
        reward = -1
    elif abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1:
        reward = -1

    return reward


def get_done(abs_pos_own, abs_pos_adv1, abs_pos_adv2, abs_pos_adv3, num_step, max_step):
    dist1 = get_dist(abs_pos_own, abs_pos_adv1)
    dist2 = get_dist(abs_pos_own, abs_pos_adv2)
    dist3 = get_dist(abs_pos_own, abs_pos_adv3)
    
    if dist1 < 0.1 or dist2 < 0.1 or dist3 < 0.1 or num_step > max_step or  \
       abs_pos_own[0] < -1 or abs_pos_own[1] < -1 or abs_pos_own[0] > 1 or abs_pos_own[1] > 1 or \
       abs_pos_adv1[0] < -1 or abs_pos_adv1[1] < -1 or abs_pos_adv1[0] > 1 or abs_pos_adv1[1] > 1 or \
       abs_pos_adv2[0] < -1 or abs_pos_adv2[1] < -1 or abs_pos_adv2[0] > 1 or abs_pos_adv2[1] > 1 or \
       abs_pos_adv3[0] < -1 or abs_pos_adv3[1] < -1 or abs_pos_adv3[0] > 1 or abs_pos_adv3[1] > 1:
        done = True
    else:
        done = False

    return done