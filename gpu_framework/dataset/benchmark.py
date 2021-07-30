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


# NN is used as part of the condition, the loss to minimize has nothing to do with the NN calculation
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


# NN is used as part of the condition, the loss to minimize has something to do with the NN calculation
def unsmooth_1_a(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [1, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    z = 0.0
    trajectory_list = list()

    # y = a * x + b
    y = linear_nn(x, a, b)
    if y <= bar:
        z = 10 + y
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def unsmooth_1_b(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [1, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    z = 0.0
    trajectory_list = list()

    # y = a * x + b
    y = linear_nn(x, a, b)
    if y <= bar:
        z = 10 + y
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def unsmooth_1_c(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [1, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    z = 0.0
    trajectory_list = list()

    # y = a * x + b
    y = linear_nn(x, a, b)
    if y <= bar:
        z = 10 - y
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


def nn(x, a, b):
    y = a * x + b
    return y

# DiffAI stuck, DSE can learn
# benchmark: pattern1
def pattern1_a(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [1, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = 10.0
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def pattern1_b(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = 10.0
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def pattern2(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 0]
    a = 2.0
    b = 20.0
    bar = 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = x + 10
    else:
        z = x - 5
    trajectory_list.append((x, z))

    return trajectory_list


def pattern3_a(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [1, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = 10 - y
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def pattern3_b(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = 10 - y
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def pattern31_a(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [1, 1]
    a = 2.0
    b = -20.0
    bar = - 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = 1
    else:
        z = 2 + y*y
    trajectory_list.append((x, z))

    return trajectory_list


def pattern31_b(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 1]
    a = 2.0
    b = 20.0
    bar = 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = 1
    else:
        z = 10 - y
    trajectory_list.append((x, z))

    return trajectory_list


def pattern5_a(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [1, 1]
    a = 0.2
    b = 1.0
    bar = 1.0
    trajectory_list = list()
    y = a * x + b # [0, 2]
    w1, b1 = 0.2, 0.0
    if y <= bar:
        z = nn(x, w1, b1)
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


def pattern5_b(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 1]
    a = 0.2
    b = 1.0
    bar = 1.0
    trajectory_list = list()
    y = a * x + b # [0, 2]
    w1, b1 = 0.2, 0.0
    if y <= bar:
        z = nn(x, w1, b1)
    else:
        z = 1
    trajectory_list.append((x, z))

    return trajectory_list


# bad join hurts
def pattern6(x, safe_bound):
    # x in [-1, 1]
    # safe area: z: [-5, 0]
    a = 0.5
    b = - 0.5
    bar = 0.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = y # + random.random()
    else:
        z = - 10.0
    trajectory_list.append((x, z))

    return trajectory_list


def pattern7(x, safe_bound):
    # x in [-1, 1]
    # safe area: z: [-5, 0]
    a = 0.5
    b = - 0.5
    bar = 0.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = y + random.random()
    else:
        z = - 10.0
    trajectory_list.append((x, z))

    return trajectory_list


def pattern8(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 1]
    a = 2
    b = 1
    bar = 0.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = y + 2 # y <= - 1
    else: # y >= 0
        z = - y
    trajectory_list.append((x, z))

    return trajectory_list


# lots of branches
def pattern9(x, safe_bound):
    return trajectory_list


# path explosion
def pattern10(x, safe_bound):
    return trajectory_list


# map
# 10 x 20
# initial area: x: [4.0, 6.0], y: [0, 0]
# safe area: # not reach -> unsafety
# x: [0, 3], y: [14.0, 19.0]
# x: [4, 6], y: [0.0, 19.0]
# x: [7, 7], y: [4.0, 19.0]
# x: [8, 8], y: [8.0, 19.0]
# x: [9, 9], y: [12.0, 19.0]
# goal area: x: [0.0, 1.0], y: [14.0, 19.0]
# xxxxxxxxxxxxxxx..ggg
# xxxxxxxxxxxxxxx....g
# xxxxxxxxxxxxxxx....g
# xxxxxxxxxxxxxxx....x
# s.............aa...x
# s.............a....x
# s........aaa.aa....x
# xxxx......aaaaa....x
# xxxxxxxx...aaa.....x
# xxxxxxxxxxxx...a...x

# control the steel angle
# steel angle: [-1.0, -0.25]: up-right, (-0.25, 0.25]: right, (0.25, 1.0]: down-right
def car_control(x, y):
    angle = 0.0
    if y <= 9:
        angle = 0.5
    elif y <= 12:
        angle = 1
    elif y <= 15:
        angle = 0.0
    else:
        angle = 0.0
    return angle
        
# data loss: angle trajectory(same length)
def racetrack_easy(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        angle = car_control(x, y)
        if angle <= 0.25:
            x -= 1
        elif angle <= 0.75:
            x = x
        else:
            x += 1
        y += 1
        trajectory_list.append((x, y, angle))
    return trajectory_list


# easier
def racetrack_easy_1(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        angle = car_control(x, y)
        if angle <= 0.25:
            x -= 1
        elif angle <= 0.75:
            x = x
        else:
            x += 1
        y += 1
        trajectory_list.append((x, y, angle))
    return trajectory_list


# easiest
def racetrack_easy_2(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        angle = car_control(x, y)
        if angle <= 0.25:
            x -= 1
        elif angle <= 0.75:
            x = x
        else:
            x += 1
        y += 1
        trajectory_list.append((x, y, angle))
    return trajectory_list


# thermostat
# assume the dt is 0.1
def warming(x, h):
    k = 0.1
    # h ~ 150.0
    dt = 0.5
    x = x - dt * k * x + h # increase by the gap between heat and the current temperature
    return x

def cooling(x):
    k = 0.1
    dt = 0.5
    x = x - dt * k * x
    return x

def nn_heat_policy(x):
    tOff = 76.0
    if x <= tOff:
        h = 15.0
        isOn = 1.0
    else:
        h = 0.0
        isOn = 0.0
        
    return h, isOn

def nn_cool_policy(x):
    # tOff = 80.0
    tOn = 65.0
    if x <= tOn:
        isOn = 1.0
    else:
        isOn = 0.0
    return isOn

def thermostat_refined(x, safe_bound):
    # x: [58.0, 68.0]
    isOn = 0.0
    steps = 40
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn < 0.5:
            isOn = nn_cool_policy(x) # isOn should have a sigmoid
            x = cooling(x)
            trajectory_list.append(([x], [isOn], ['cool']))
        else: 
            h, isOn = nn_heat_policy(x)
            x = warming(x, h)
            trajectory_list.append(([x], [isOn, h], ['heat']))
        # separate two nns
        
    return trajectory_list
    





    



    
