import math
import random

import torch.nn as nn
import torch
from scipy.stats import bernoulli

import numpy as np


def thermostat(lin, safe_bound):
    x = lin
    tOff = 78.0
    tOn = 66.0
    isOn = 0.0
    trajectory_list = list()

    for i in range(40):
        state = (lin, x)
        if isOn <= 0.5: # ifblock1
            x = x - 0.1 * (x - lin)
            if x <= tOn: # ifelse_tOn
                isOn = 1.0
            else:
                isOn = 0.0
        else: # ifblock2
            x = x - 0.1 * (x - lin) + 10.0
            if x <= tOff:
                isOn = 1.0
            else:
                isOn = 0.0
        
        # if x > safe_bound:
        #     x = safe_bound
        trajectory_list.append((state[0], state[1], x))
        # print(f"x: {x}, isOn:{isOn}")
    
    return trajectory_list


def safe_acceleration(p, v, safe_bound):
    u = 0.0
    if v <= 0.0:
        u = - 1.0
    else:
        u = safe_bound
    
    return u


def mountain_car(p0, safe_bound):
    # pos should be in [-1.2, 0.6]
    # initial range: [-1.6, 0.0]
    v = 0
    p = p0
    min_position = -1.2
    goal_position = 0.5
    min_speed = -0.07
    max_speed = 0.07
    reward = 0
    i = 0
    min_u = -1.6
    max_u = 1.6
    trajectory_list = list()

    while p <= goal_position:
        # Reset position if position is out of range
        if p <= min_position:
            p = min_position
            v = 0
        # update acceleration
        u = safe_acceleration(p, v, safe_bound)
        # update trajectory
        trajectory_list.append((p, v, u))
        
        # update velocity
        v = v + 0.0015 * u - 0.0025 * math.cos(3 * p)
        # Reset v if v is out of range
        if v <= min_speed: 
            v = min_speed
        else:
            if v <= max_speed:
                v = v 
            else:
                v = max_speed
        # update position
        p = p + v
        # i += 1
        # print(i, p)
    # print(f'finish one')
    
    return trajectory_list


def safe_acceleration_p(p, v, safe_bound):
    u_p = 0.0
    if v <= 0.0:
        u_p = 0 # left
    else:
        u_p = 1 # right
    
    return u_p


def mountain_car_1(p0, safe_bound):
    # pos should be in [-1.2, 0.6]
    # initial range: [-0.4, -0.6]
    # safe area of v: [-0.07, 0.07]
    v = 0
    p = p0
    min_position = -1.2
    goal_position = 0.5

    i = 0
    trajectory_list = list()

    while p <= goal_position:
        # Reset position if position is out of range
        if p <= min_position:
            p = min_position
            v = 0
        # update acceleration
        u_p = safe_acceleration_p(p, v, safe_bound)
        # update trajectories
        trajectory_list.append((p, v, u_p))

        # u = np.random.binomial(1, u_p, 1).tolist()[0]
        if u_p <= 0.5:
            u = -1.0
        else:
            u = 1.0
        
        # update velocity
        v = v + 0.0015 * u - 0.0025 * math.cos(3 * p)

        p = p + v
        # i += 1
        # print(i, p)
    print(f'finish one')
    
    return trajectory_list


def linear_nn(x, a, b):
    y = a * x + b
    return y


def unsmooth_1(x, safe_bound):
    # x in [-5, 5]
    a = 2.0
    b = 20.0
    bar = 1.0
    z = 0.0
    trajectory_list = list()

    # y = a * x + b
    y = linear_nn(x, a, b)
    if y <= bar:
        z = 10 
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def unsmooth_2_separate(x, safe_bound):
    # x in [-5, 5]
    a = 0.1
    b = 2.0
    bar = 1.0
    z = 0.0
    trajectory_list = list()
    y = a * x + b
    trajectory_list.append((x, y))
    if y <= bar:
        z = 10
    else:
        z = 1
    y = a * z + b
    trajectory_list.append((z, y))

    return trajectory_list


def unsmooth_2_overall(x, safe_bound):
    # x in [-5, 5]
    a = 0.1
    b = 2.0
    bar = 1.0
    z = 0.0
    trajectory_list = list()
    y = a * x + b
    if y <= bar:
        z = 10
    else:
        z = 1
    y = a * z + b
    trajectory_list.append((x, y))

    return trajectory_list


def generate_p(x, y):
    sigmoid = nn.Sigmoid()
    z = 2 * x + 5 * y + 1
    return sigmoid(torch.Tensor([z]))


def sampling_1(x, safe_bound):
    # x: [-1.0, 1.0]
    # condition: safe area: z == 1; max_unsafe_probability=0.0
    trajectory_list = list() # for data loss

    # select from bernoulli distribution
    y = np.random.binomial(1, 0.2, 1).tolist()[0] 
    bar = 0.5
    
    v = generate_p(x, y)
    v = v.detach().cpu().numpy().tolist()[0]
    trajectory_list.append((x, y, v))

    if v <= bar:
        z = 10
    else:
        z = 1

    return trajectory_list


def extract_expRank(yExp, colRank):
    y = 0.2 * (10 - yExp) + colRank
    return y 
    

def sampling_2(yExp, safe_bound):
    # yExp: [0.0, 10.0]
    # condition: p of (hire==1) <= 0.3
    # in our framework, safe area: hire==0, max_unsafe_probability = 0.3

    p_gender = 0.5
    p_colRank = 0.5
    trajectory_list = list()

    gender = np.random.binomial(1, p_gender, 1).tolist()[0]
    # top, non-top: [0, 1]
    colRank = np.random.binomial(1, p_colRank, 1).tolist()[0] 
    if gender <= 0.5:
        colRank = 1.0 # if gender is in a minor group, colRank is assigned to non-top
    
    # extract the relative rank
    expRank = extract_expRank(yExp, colRank)
    trajectory_list.append((yExp, colRank, expRank))

    if colRank <= 0.5:
        hire = 1
    elif expRank <= 1.8:
        hire = 1
    else:
        hire = 0.0
    
    # TODO: for fairness
    # if gender == 0 and hire == 1: 
    #     z = 1 
    # elif gender == 1 and hire == 1:
    #     z = 5
    # else:
    #     z = 10
    #  assert(p(z==1) > p(z==5))
    
    return trajectory_list


def extract_expRank_fairness_1(yExp, colRank):
    y = 0.2 * (10 - yExp) + colRank
    return y 


def fairness_1(yExp, safe_bound):
    p_gender = 0.5
    p_colRank = 0.5
    trajectory_list = list()

    gender = np.random.binomial(1, p_gender, 1).tolist()[0]
    # top, non-top: [0, 1]
    colRank = np.random.binomial(1, p_colRank, 1).tolist()[0] 
    if gender <= 0.5:
        colRank = 1.0 # if gender is in a minor group, colRank is assigned to non-top
    
    # extract the relative rank
    expRank = extract_expRank_fairness_1(yExp, colRank)
    # trajectory_list.append((yExp, colRank, expRank))

    if colRank <= 0.5:
        hire = 1
    elif expRank <= 1.8:
        hire = 1
    else:
        hire = 0.0
    
    trajectory_list.append((yExp, colRank, gender, hire))

    m = 0
    n = 0
    g_f = 0
    g_m = 0
    if gender <= 0.5:
        g_f = 1
    else:
        g_m = 1
    
    # for fairness
    if hire <= 0.5:
        pass
    else:
        if gender <= 0.5:
            m = 1
        else:
            n = 1

    # safe area: m==0, g_f==0, n==0, g_m==0
    # assert(p(m==1)/p(g_f==1) >= p(n==1)/p(g_m==1))
    # assert(p(m==1)*p(g_m==1) >= p(n==1)*p(g_f==1))
    # (1 - p(m==0)) * (1 - p(g_m==0)) / (p(n==1) * p(g_f==1))
    
    return trajectory_list


def extract_bound(h0):
    y = 0.0001 * h0 + 5.49
    return y


def extract_h(h):
    return h * 2 - 4.9


def path_explosion(h0, safe_bound):
    # h0: [2.0, 4.8]
    # safe area: h: [0.0, 5.0]

    trajectory_list = list()

    bar1 = 3.0
    bar2 = 5.0
    bar3 = 2.5

    h = h0
    for i in range(50):
        h += 0.2
        if h <= bar1:
            h += 0.2
        elif h <= bar2:
            h_pre = h
            h = extract_h(h_pre)
            if h_pre > 4.8:
                pass
            else:
                trajectory_list.append((h_pre, h))

    return trajectory_list


def path_explosion_2(h0, safe_bound):
    # h0: [2.0, 4.8]
    # safe area: h: [0.0, 5.0]

    trajectory_list = list()

    bar1 = 3.0
    bar2 = 5.0
    bar3 = 2.5

    h = h0
    for i in range(50):
        h += 0.2
        if h <= bar1:
            h += 0.2
        elif h <= bar2:
            h_pre = h
            h = extract_h(h_pre)
            if h_pre > 4.8:
                pass
            else:
                trajectory_list.append((h_pre, h))
        if h <= bar3:
            h = 2*h + 1

    return trajectory_list
        

    
