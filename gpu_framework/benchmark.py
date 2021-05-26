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



def acceleration(p, v):
    u = 0.0
    if v <= 0.0:
        u = - 1.0
    else:
        u = 1.0
    return u


def safe_acceleration(p, v, safe_bound):
    u = 0.0
    # if v <= 0.0:
    #     # u = - 0.8
    #     u = - safe_bound + random.uniform(0.00001, 0)
    # else:
    #     u = safe_bound + random.uniform(-0.00001, 0)
    # if v <= 0.0:
    #     u = -(-v/0.05)**0.5 * safe_bound + 0.01* (0.5 - p)/1.2
    # else:
    #     u = (v/0.05) ** 0.5 * safe_bound + 0.01 * (0.5 - p)/1.2

    # u = (v/0.1)**2 * 10 * safe_bound + 0.01* (0.5 - p)/1.2
    # u = v * (v ** 2 / 0.025) ** 0.5 * safe_bound + 0.01* (0.5 - p)/1.2
    # print(p, v, u)
    # if v <= 0.0:
    #     # u = - 0.8
    #     u = - safe_bound + random.uniform(0.00001, 0)
    # else:
    #     u = safe_bound + random.uniform(-0.00001, 0)
    # u = v / ((v+0.1)**2) * (-v**2 + (safe_bound - 0.2)) + 0.01* (0.5 - p)/1.2
    # if v <= 0.0:
    #     u = - safe_bound * v + random.uniform(0.00001, 0)
    # else:
    #     u = safe_bound + random.uniform(-0.00001, 0)
    # fraction = - 1 / (0.1**2) * v ** 2 + 1
    # u = safe_bound * v/0.07 * 10 * fraction + 0.01* (0.5 - p)/1.2 * v * fraction
    # print(p, v, u)
    if v <= 0.0:
        u = - safe_bound
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
        # pre-filter of acceleration, to add unsoundness

        if u <= min_u:
            u = min_u
        elif u <= max_u:
            u = u
        else:
            u = max_u
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


def linear_nn(x, a, b):
    y = a * x + b
    return y


def unsound_1(x, safe_bound):
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


def unsound_2_separate(x, safe_bound):
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


def unsound_2_overall(x, safe_bound):
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


# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# nn = SampleNN()

def generate_p(x, y):
    sigmoid = nn.Sigmoid()
    z = 2 * x + 5 * y + 1
    return sigmoid(torch.Tensor([z]))


def sampling_1(x, safe_bound):
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


def sampling_2(x, safe_bound):
    




'''
Conceptual Benchmarks
'''

# Linear, ReLu, Linear, Sigmoid
class SampleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.sigmoid(res)

        return res

def mountain_car_algo(p, v):
    # p: [goal_position, infinity], 0.1
    # u: [-0.9, 0.9], 0.1
    while p <= goal_position:
        if p <= min_position: p, v = Reset(p, v)
        u = DNN(p, v)
        v = Update(v, p, u)
        if v <= min_speed or v >= max_speed: 
            v = Reset(v)
        p = p + v

        reward = reward + (-0.1) * u  * u

        i += 1
        if i > 1000:
            break
    
    if p >= goal_position:
        reward += 100
    
    return reward


def mountain_car_concept(p, v):
    trajectory = list()
    while p <= goal_position:
        if p <= min_position: p, v = Reset(p, v)

        u = acceleration(p, v)
        trajectory.append((p, v, u))

        v = Update(v, p, u)
        if v <= min_speed or v >= max_speed: 
            v = Reset(v)

        p = p + v
    
    return trajectory


# def sampling_1(x, safe_bound):
#     trajectory_list = list()
#     bar = 0.5

#     # create a very small p0 by faking nn
#     p0 = generate_p(x)
#     p1 = 1 - p0

#     # v is a new variable, with new probability
#     v = random.choices(
#             population=[0, 1],
#             weights=[p0, p1], # the p0 is a distribution, p1 is also a distribution
#             k=1,
#         )[0]
    
#     # DiffAI version: if p0 > p1: y=10 else: y=1; if intersection: both
#     if v <= bar:
#         y = 10
#     else:
#         y = 1
    
#     trajectory_list.append((x, y))
#     return trajectory_list

# def p1(r, g, b):
#     win = -1
#     # extract the probability for deciding the color
#     output_layer = NN(r, g, b)
#     # for a three-class classifier
#     # output_layer is [p1, p2, p3], pi is the probability to be in class i
#     index = sample_index(output_layer) 
#     if index == 1:
#         win = 1
#     else:
#         win = 0
#     return win


# def p2(coin, color):
#     if coin == "head" and color == "red":
#         win = 1
#     else:
#         win = 0
#     return win

    
