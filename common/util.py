import numpy as np


def get_sub_pos(abs_pos_own, abs_pos_adv):
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(abs_pos_own[1], abs_pos_own[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sub_pos = np.dot(rot, pos_rel)

    return sub_pos


def get_sub_vel(abs_pos_own, abs_pos_adv, abs_vel):
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(abs_pos_own[1], abs_pos_own[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sub_vel = np.dot(rot, abs_vel)

    return sub_vel


def get_abs_u(action, abs_own_pos, abs_adv_pos):
    if action <= 11:
        ang = action * -np.pi / 6
        sub_u = [np.cos(ang), np.sin(ang)]            
        abs_u = rotate_u(sub_u, abs_own_pos, abs_adv_pos)
    elif action == 12:
        abs_u = [0, 0]
    
    return abs_u


def rotate_u(sub_u, abs_pos_own, abs_pos_adv):
    sub_u = np.array(sub_u)
    pos_rel = abs_pos_adv - abs_pos_own
    theta = np.arctan2(pos_rel[1], pos_rel[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    abs_u = np.dot(rot, sub_u)

    return abs_u


def get_next_own_state(abs_pos_own, abs_vel_own, abs_u, mass, speed, damping, dt):
    abs_acc_own = np.array(abs_u) / mass
    next_abs_vel_own = abs_vel_own * (1 - damping) + abs_acc_own * speed * dt
    next_abs_pos_own = abs_pos_own + next_abs_vel_own * dt

    return next_abs_pos_own, next_abs_vel_own


def get_dist(abs_pos_own, abs_pos_adv):
    pos_rel = abs_pos_adv - abs_pos_own
    dist = np.sqrt(np.sum(np.square(pos_rel)))

    return dist
