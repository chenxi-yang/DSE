import math
import random

import torch.nn as nn
import torch
from scipy.stats import bernoulli
import random

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

# trajectory_list.append(([x, 0], [isOn, 0.0]))

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
    trajectory_list.append(([x], [z]))

    return trajectory_list


def pattern2(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 0]
    # ax+b \in [10, 30]
    a = 2.0
    b = 20.0
    bar = 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = x + 10
    else:
        z = x - 5
    trajectory_list.append(([x], [z]))

    return trajectory_list


def nn_example(x):
    if x > 5:
        y = math.sin(x) # , random.random() + (- 0.4) # 1 + 0.01 * random.random()
    else:
        if x <= -2.5:
            y = math.sin(x) + 1.00000000000001
        elif x <= 0: # >-2.5, <= 0
            y = max(x * x, 1 + random.random())
        elif x <= 2.5: # > 0, x<= 2.5
            y = math.cos(x) + 1.00000000000001
        else: # x>2.5 && x<= 5
            y = max((x - 4) * (x - 4), 1.00000000000001) # 1 + (x - 1) * random.random() * (x - 1) * random.random()
    return y # , acc


def pattern_example(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 1]
    bar = 1.0
    trajectory_list = list()
    for i in range(1):
        y = nn_example(x)
        # x >= 0: y <= 1, acc <= -5
        # x < 0: y > 1, 
        trajectory_list.append(([x], [y]))
        if y <= bar: # -x >= 0 # x > 0 
            z = x + 10
        else: # x >= 0
            z = x - 5
    
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
    trajectory_list.append(([x], [z]))

    return trajectory_list


def pattern31_a(x, safe_bound):
    # x in [-5, 5]
    # safe area: z: [-oo, 1]
    a = 2.0
    b = -20.0
    bar = - 1.0
    trajectory_list = list()
    y = nn(x, a, b)
    if y <= bar:
        z = 1
    else:
        z = 2 + y*y
    trajectory_list.append(([x], [z]))

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
    trajectory_list.append(([x], [z]))

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
    trajectory_list.append(([x], [z]))

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
    trajectory_list.append(([x], [z]))

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

    trajectory_list.append(([x], [y+random.random()]))
    if y <= bar:
        z = y #  + random.random()
    else:
        z = - 10.0
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
        trajectory_list.append(([x, y], [angle]))
        if angle <= 0.25:
            x -= 1
        elif angle <= 0.75:
            x = x
        else:
            x += 1
        y += 1
        # trajectory_list.append((x, y, angle))
    return trajectory_list


def car_control_classifier(x, y):
    p0, p1, p2 = 0.0, 0.0, 0.0

    if y <= 9:
        p1 = 1.0
    elif y <= 12:
        p2 = 1.0
    elif y <= 13: # add difficulty in the training trajectory
        p1 = 1.0
    else:
        p0 = 1.0
    
    return p0, p1, p2


def racetrack_easy_classifier(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        p0, p1, p2 = car_control_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        if p0 == 1:
            x -= 1
        elif p1 == 1:
            x = x
        else:
            x += 1
        y += 1
    return trajectory_list


def racetrack_easy_classifier_ITE(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p0, p1, p2 = car_control_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        # additional comparison between pi to model argmax
        a = p1 - p0
        b = p2 - p0
        c = p2 - p1
        if a <= 0: # p1 <= p0
            if b <= 0: # p2 <= p0
                index = 0
            else: # p2 > p0 >= p1
                index = 2
        else: # p1 > p0
            if c <= 0: # p2 <= p1 (p1 >= p2, p1 > p0)
                index = 1
            else: # p2 > p1 > p0
                index = 2
        if index == 0:
            x -= 1
        elif index == 1:
            x = x
        else:
            x += 1
        y += 1
    
    return trajectory_list


# map
# 10x20
# initial area: x: [4.0, 6.0], y: [0, 0]
# xxxxxxxxxxxxxxx..goo
# xxxxxxxxxxxxxxx..o.g
# xxxxxxxxxxxxxxx.o..g
# xxxxxxxxxxxxxxxo..oo
# ..............oa.oox
# ..............a.oo.x
# .........aaa.aaoo..x
# xxxx......aaaaoo...x
# xxxxxxxx...aaoo....x
# xxxxxxxxxxxx.o.a...x

# fuck u
map_safe_range_easy_multi = [
        [4.0, 6.0], [4.0, 6.0], [4.0, 6.0], [4.0, 6.0],
        [4.0, 7.0], [4.0, 7.0], [4.0, 7.0], [4.0, 7.0],
        [4.0, 8.0], [4.0, 8.0], [4.0, 8.0], [4.0, 8.0],
        [4.0, 9.0], [4.0, 9.0], [4.0, 9.0], [0.0, 9.0], 
        [0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.0, 4.0],
    ]
absolute_safe_range_easy_multi = [
        [4.0, 6.0], [4.0, 6.0], [4.0, 6.0], [4.0, 6.0],
        [4.0, 7.0], [4.0, 7.0], [4.0, 7.0], [4.0, 7.0],
        [4.0, 8.0], [4.0, 8.0], [4.0, 8.0], [4.0, 8.0],
        [4.0, 9.0], [4.0, 9.0], [4.0, 8.0], [0.0, 7.0], 
        [0.0, 7.0], [0.0, 6.0], [0.0, 5.0], [0.0, 4.0],
    ]

def car_control_easy_multi_classifier(x, y):
    p0, p1, p2 = 0.0, 0.0, 0.0
    # 
    next_abs_safe_range = absolute_safe_range_easy_multi[y]
    select_list = []
    if x <= next_abs_safe_range[1] and x >= next_abs_safe_range[0]:
        select_list.append(1)
    if x - 1 <= next_abs_safe_range[1] and x - 1 >= next_abs_safe_range[0]:
        select_list.append(0)
    if x + 1 <= next_abs_safe_range[1] and x + 1 >= next_abs_safe_range[0]:
        select_list.append(2)
    index = random.choice(select_list)
    if index == 0:
        p0 = 1.0
    elif index == 1:
        p1 = 1.0
    else:
        p2 = 1.0
    
    return p0, p1, p2


def from_index3_to_position(p0, p1, p2, x, y):
    a = p1 - p0
    b = p2 - p0
    c = p2 - p1
    if a <= 0: # p1 <= p0
        if b <= 0: # p2 <= p0
            index = 0
        else: # p2 > p0 >= p1
            index = 2
    else: # p1 > p0
        if c <= 0: # p2 <= p1 (p1 >= p2, p1 > p0)
            index = 1
        else: # p2 > p1 > p0
            index = 2
    if index == 0:
        x -= 1
    elif index == 1:
        x = x
    else:
        x += 1
    y += 1
    return x, y


def racetrack_easy_multi(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x1, y1, x2, y2 = x, 0, x, 0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p00, p01, p02 = car_control_easy_multi_classifier(x1, y1)
        trajectory_list.append(([x1, y1, 0], [p00, p01, p02])) # agent 1
        p10, p11, p12 = car_control_easy_multi_classifier(x2, y2)
        trajectory_list.append(([x2, y2, 1], [p10, p11, p12])) # agent 2
        # additional comparison between pi to model argmax
        x1, y1 = from_index3_to_position(p00, p01, p02, x1, y1)
        x2, y2 = from_index3_to_position(p10, p11, p12, x2, y2)
        
    return trajectory_list


def racetrack_easy_multi2(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x1, y1, x2, y2 = x, 0, x, 0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p00, p01, p02 = car_control_easy_multi_classifier(x1, y1)
        trajectory_list.append(([x1, y1, 0], [p00, p01, p02])) # agent 1
        p10, p11, p12 = car_control_easy_multi_classifier(x2, y2)
        trajectory_list.append(([x2, y2, 1], [p10, p11, p12])) # agent 2
        # additional comparison between pi to model argmax
        x1, y1 = from_index3_to_position(p00, p01, p02, x1, y1)
        x2, y2 = from_index3_to_position(p10, p11, p12, x2, y2)
        
    return trajectory_list


# map (relaxed multi)
# 10x20
# initial area: x: [5.0, 6.0], y: [0, 0]
# 0 xxxxxxxxxxxxxxx..goo
# 1 xxxxxxxxxxxxxxx..o.g
# 2 xxxxxxxxxxxxxxx.o..g
# 3 xxxxxxxxxxxxxxxo..oo
# 4 ..............oa.oox
# 5 ..............a.oo.x
# 6 .........aaa.aaoo..x
# 7 xxxx......aaaaoo...x
# 8 xxxxxxxx...aaoo....x
# 9 xxxxxxxxxxxx.o.a...x

# fxxxxx!!!! a!
map_safe_range_relaxed_multi = [
        [4.0, 7.0], [4.0, 7.0], [4.0, 7.0], [4.0, 7.0],
        [4.0, 8.0], [4.0, 8.0], [4.0, 8.0], [4.0, 8.0],
        [4.0, 9.0], [4.0, 9.0], [4.0, 9.0], [4.0, 9.0],
        [4.0, 10.0],[4.0, 10.0],[4.0, 10.0],[0.0, 10.0], 
        [0.0, 10.0],[0.0, 10.0],[0.0, 10.0],[0.0, 4.0],
    ]
absolute_safe_range_relaxed_multi = [
        [4.0, 7.0], [4.0, 7.0], [4.0, 7.0], [4.0, 7.0],
        [4.0, 8.0], [4.0, 8.0], [4.0, 8.0], [4.0, 8.0],
        [4.0, 9.0], [4.0, 9.0], [4.0, 9.0], [4.0, 9.0],
        [4.0, 10.0],[4.0, 10.0],[4.0, 9.0], [0.0, 8.0], 
        [0.0, 7.0], [0.0, 6.0], [0.0, 5.0], [0.0, 4.0],
    ]

def car_control_relaxed_multi_classifier(x, y):
    p0, p1, p2 = 0.0, 0.0, 0.0
    # 
    next_abs_safe_range = absolute_safe_range_relaxed_multi[y]
    select_list = []
    if x <= next_abs_safe_range[1] and x >= next_abs_safe_range[0]:
        select_list.append(1)
    if x - 1 <= next_abs_safe_range[1] and x - 1 >= next_abs_safe_range[0]:
        select_list.append(0)
    if x + 1 <= next_abs_safe_range[1] and x + 1 >= next_abs_safe_range[0]:
        select_list.append(2)
    index = random.choice(select_list)
    if index == 0:
        p0 = 1.0
    elif index == 1:
        p1 = 1.0
    else:
        p2 = 1.0
    
    return p0, p1, p2


def from_index3_to_position(p0, p1, p2, x, y):
    a = p1 - p0
    b = p2 - p0
    c = p2 - p1
    if a <= 0: # p1 <= p0
        if b <= 0: # p2 <= p0
            index = 0
        else: # p2 > p0 >= p1
            index = 2
    else: # p1 > p0
        if c <= 0: # p2 <= p1 (p1 >= p2, p1 > p0)
            index = 1
        else: # p2 > p1 > p0
            index = 2
    if index == 0:
        x -= 1
    elif index == 1:
        x = x
    else:
        x += 1
    y += 1
    return x, y

# start from one cell in the center, expecting the trajectories to be separate from each other
def racetrack_relaxed_multi(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x1, y1, x2, y2 = x, 0, x, 0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p00, p01, p02 = car_control_relaxed_multi_classifier(x1, y1)
        trajectory_list.append(([x1, y1, 0], [p00, p01, p02])) # agent 1
        p10, p11, p12 = car_control_relaxed_multi_classifier(x2, y2)
        trajectory_list.append(([x2, y2, 1], [p10, p11, p12])) # agent 2
        # additional comparison between pi to model argmax
        x1, y1 = from_index3_to_position(p00, p01, p02, x1, y1)
        x2, y2 = from_index3_to_position(p10, p11, p12, x2, y2)
        
    return trajectory_list


def racetrack_relaxed_multi2(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x1, y1, x2, y2 = x, 0, x, 0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p00, p01, p02 = car_control_relaxed_multi_classifier(x1, y1)
        trajectory_list.append(([x1, y1, 0], [p00, p01, p02])) # agent 1
        p10, p11, p12 = car_control_relaxed_multi_classifier(x2, y2)
        trajectory_list.append(([x2, y2, 1], [p10, p11, p12])) # agent 2
        # additional comparison between pi to model argmax
        x1, y1 = from_index3_to_position(p00, p01, p02, x1, y1)
        x2, y2 = from_index3_to_position(p10, p11, p12, x2, y2)
        
    return trajectory_list

# map
# 10x30
# initial area: x: [7.0, 10.0], y: [0, 0]
# xxxxx xxxxx xxxxx xxxxx xxxxx xxxxx
# xxxxx xxxx. ..xxx xxxxx xxxxx xxxxx
# xxxxx xxx.. oo.xx xxxxx x..oo o.xxx
# xxxxx x...o ..o.. xxxxx ..o.. .oxxx
# xxxxx x..o. oo.o. xxxxx .o.o. ..oxx
# xxxxx x.o.o xxo.o .xxx. o.oxo ...ox
# xxxxx xo.o. xx.o. oxxxo .oxx. oo..o
# sooxx o.o.x xxx.o .oxo. oxxxx xxo..
# s..oo .o..x xxx.. o.ooo xxxxx xx.oo
# soooo o..xx xxxx. .ooxx xxxxx xxxxx
map_safe_range_moderate = [
        [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [8.0, 10.0], [8.0, 10.0],
        [7.0, 10.0], [3.0, 10.0], [3.0, 10.0], [2.0, 9.0],  [1.0, 7.0],
        [1.0, 5.0],  [1.0, 5.0],  [2.0, 7.0],  [3.0, 9.0],  [3.0, 10.0],
        [5.0, 10.0], [7.0, 10.0], [8.0, 10.0], [7.0, 9.0],  [5.0, 9.0],
        [3.0, 8.0],  [2.0, 7.0],  [2.0, 6.0],  [2.0, 5.0],  [2.0, 7.0], 
        [2.0, 7.0],  [2.0, 7.0],  [4.0, 9.0],  [5.0, 9.0],  [6.0, 9.0],
    ]
absolute_safe_range_moderate = [
        [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [8.0, 10.0], [8.0, 10.0],
        [7.0, 10.0], [6.0, 9.0],  [5.0, 8.0],  [4.0, 7.0],  [3.0, 6.0],
        [2.0, 5.0],  [2.0, 5.0],  [3.0, 6.0],  [4.0, 7.0],  [5.0, 8.0],
        [6.0, 9.0],  [7.0, 10.0], [8.0, 10.0], [7.0, 9.0],  [6.0, 9.0],
        [5.0, 8.0],  [4.0, 7.0],  [3.0, 6.0],  [2.0, 5.0],  [2.0, 6.0], 
        [2.0, 7.0],  [3.0, 7.0],  [4.0, 8.0],  [5.0, 9.0],  [6.0, 9.0],
    ]

def car_control_moderate_classifier(x, y):
    p0, p1, p2 = 0.0, 0.0, 0.0
    # 
    next_abs_safe_range = absolute_safe_range_moderate[y]
    select_list = []
    if x <= next_abs_safe_range[1] and x >= next_abs_safe_range[0]:
        select_list.append(1)
    if x - 1 <= next_abs_safe_range[1] and x - 1 >= next_abs_safe_range[0]:
        select_list.append(0)
    if x + 1 <= next_abs_safe_range[1] and x + 1 >= next_abs_safe_range[0]:
        select_list.append(2)
    index = random.choice(select_list)
    if index == 0:
        p0 = 1.0
    elif index == 1:
        p1 = 1.0
    else:
        p2 = 1.0
    
    return p0, p1, p2


def racetrack_moderate_classifier_ITE(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x, y = x, 0
    steps = 30
    trajectory_list = list()
    for i in range(steps):
        p0, p1, p2 = car_control_moderate_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        # additional comparison between pi to model argmax
        a = p1 - p0
        b = p2 - p0
        c = p2 - p1
        if a <= 0: # p1 <= p0
            if b <= 0: # p2 <= p0
                index = 0
            else: # p2 > p0 >= p1
                index = 2
        else: # p1 > p0
            if c <= 0: # p2 <= p1 (p1 >= p2, p1 > p0)
                index = 1
            else: # p2 > p1 > p0
                index = 2
        if index == 0:
            x -= 1
        elif index == 1:
            x = x
        else:
            x += 1
        y += 1
    
    return trajectory_list


# map
# 10x20
# initial area: x: [7.0, 10.0], y: [0, 0]
# xxxxx xxxxx xxxxx xxxxx
# xxxxx xxxx. ..xxx xxxxx
# xxxxx xxx.. oo.xx xxxxx
# xxxxx x...o ..o.. xxxxx
# xxxxx x..o. oo.o. xxxxx
# xxxxx x.o.o xxo.o .xxxx
# xxxxx oo.o. xx.o. oxxxx
# soooo o.o.x xxx.o .oooo
# s.... .o..x xxx.. o....
# soooo o..xx xxxx. .oooo 
map_safe_range_moderate_2 = [
        [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
        [6.0, 10.0], [3.0, 10.0], [3.0, 10.0], [2.0, 9.0],  [1.0, 7.0],
        [1.0, 5.0],  [1.0, 5.0],  [2.0, 7.0],  [3.0, 9.0],  [3.0, 10.0],
        [5.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
    ]
absolute_safe_range_moderate_2 = [
        [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
        [7.0, 10.0], [6.0, 9.0],  [5.0, 8.0],  [4.0, 7.0],  [3.0, 6.0],
        [2.0, 5.0],  [2.0, 5.0],  [3.0, 6.0],  [4.0, 7.0],  [5.0, 8.0],
        [6.0, 9.0],  [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
    ]

def car_control_moderate_2_classifier(x, y):
    p0, p1, p2 = 0.0, 0.0, 0.0
    # 
    next_abs_safe_range = absolute_safe_range_moderate_2[y]
    select_list = []
    if x <= next_abs_safe_range[1] and x >= next_abs_safe_range[0]:
        select_list.append(1)
    if x - 1 <= next_abs_safe_range[1] and x - 1 >= next_abs_safe_range[0]:
        select_list.append(0)
    if x + 1 <= next_abs_safe_range[1] and x + 1 >= next_abs_safe_range[0]:
        select_list.append(2)
    index = random.choice(select_list)
    if index == 0:
        p0 = 1.0
    elif index == 1:
        p1 = 1.0
    else:
        p2 = 1.0
    
    return p0, p1, p2


def racetrack_moderate_2_classifier_ITE(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x, y = x, 0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p0, p1, p2 = car_control_moderate_2_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        # additional comparison between pi to model argmax
        a = p1 - p0
        b = p2 - p0
        c = p2 - p1
        if a <= 0: # p1 <= p0
            if b <= 0: # p2 <= p0
                index = 0
            else: # p2 > p0 >= p1
                index = 2
        else: # p1 > p0
            if c <= 0: # p2 <= p1 (p1 >= p2, p1 > p0)
                index = 1
            else: # p2 > p1 > p0
                index = 2
        if index == 0:
            x -= 1
        elif index == 1:
            x = x
        else:
            x += 1
        y += 1
    
    return trajectory_list


# map
# 10x20
# initial area: x: [7.0, 10.0], y: [0, 0]
# xxxxx xxxxx xxxxx xxxxx
# xxxxx xxxx. o.xxx xxxxx
# xxxxx xxx.o oo.xx xxxxx
# xxxxx x..oo ..o.. xxxxx
# xxxxx x.oo. oo.o. xxxxx
# xxxxx xoo.o xxo.o .xxxx
# xxxxx oo.o. xxoo. oxxxx
# soooo o.o.x xxxoo .oooo
# s.... .o..x xxxoo o....
# ooooo o..xx xxxxo ooooo 
map_safe_range_moderate_3 = [
        [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
        [6.0, 10.0], [3.0, 10.0], [3.0, 10.0], [2.0, 9.0],  [1.0, 7.0],
        [1.0, 5.0],  [1.0, 5.0],  [2.0, 7.0],  [3.0, 9.0],  [3.0, 10.0],
        [5.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
    ]
absolute_safe_range_moderate_3 = [
        [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
        [6.0, 10.0], [5.0, 9.0],  [4.0, 8.0],  [3.0, 7.0],  [2.0, 6.0],
        [1.0, 5.0],  [2.0, 5.0],  [3.0, 7.0],  [4.0, 9.0],  [5.0, 10.0],
        [6.0, 10.0],  [7.0, 10.0], [7.0, 10.0], [7.0, 10.0], [7.0, 10.0],
    ]

def car_control_moderate_3_classifier(x, y):
    p0, p1, p2 = 0.0, 0.0, 0.0
    # 
    next_abs_safe_range = absolute_safe_range_moderate_3[y]
    select_list = []
    if x <= next_abs_safe_range[1] and x >= next_abs_safe_range[0]:
        select_list.append(1)
    if x - 1 <= next_abs_safe_range[1] and x - 1 >= next_abs_safe_range[0]:
        select_list.append(0)
    if x + 1 <= next_abs_safe_range[1] and x + 1 >= next_abs_safe_range[0]:
        select_list.append(2)
        # Racetrack-Moderate3-1
        select_list.append(2)
        select_list.append(2)
        select_list.append(2)
        # add weight to these kind of trajectories
    index = random.choice(select_list)
    if index == 0:
        p0 = 1.0
    elif index == 1:
        p1 = 1.0
    else:
        p2 = 1.0
    
    return p0, p1, p2


def racetrack_moderate_3_classifier_ITE(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x, y = x, 0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p0, p1, p2 = car_control_moderate_3_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        # additional comparison between pi to model argmax
        a = p1 - p0
        b = p2 - p0
        c = p2 - p1
        if a <= 0: # p1 <= p0
            if b <= 0: # p2 <= p0
                index = 0
            else: # p2 > p0 >= p1
                index = 2
        else: # p1 > p0
            if c <= 0: # p2 <= p1 (p1 >= p2, p1 > p0)
                index = 1
            else: # p2 > p1 > p0
                index = 2
        if index == 0:
            x -= 1
        elif index == 1:
            x = x
        else:
            x += 1
        y += 1
    
    return trajectory_list

# map
# 10x20
# initial area: x: [4.0, 6.0], y: [0, 0]
# xxxx. ..... ..... xxxxx
# xxx.. ..... ..... ...xx
# xx... ..... ..... ....x
# x.... xxxxx xxxxx x....
# s.xxx xxxxx xxxxx xx...
# s..xx xxxxx xxxxx x....
# x.... xxxxx xxxxx ....x
# xx... ..... ..... ..xxx
# xxx.. ..... ..... .xxxx
# xxxx. ..... ..... xxxxx 
map_safe_range_hard = [
        [[4.0, 6.0]], [[3.0, 7.0]], [[2.0, 4.0], [5.0, 8.0]], [[1.0, 4.0], [6.0, 9.0]], [[0.0, 4.0], [6.0, 10.0]],
        [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]],
        [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]],
        [[1.0, 3.0], [6.0, 9.0]], [[1.0, 4.0], [5.0, 8.0]], [[1.0, 7.0]], [[2.0, 7.0]], [[3.0, 6.0]],
    ]
absolute_safe_range_hard = [
        [[4.0, 6.0], [4.0, 6.0]], [[3.0, 7.0], [3.0, 7.0]], [[2.0, 4.0], [5.0, 8.0]], [[1.0, 4.0], [6.0, 9.0]], [[1.0, 4.0], [6.0, 10.0]],
        [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]],
        [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]], [[0.0, 3.0], [7.0, 10.0]],
        [[1.0, 3.0], [6.0, 9.0]], [[1.0, 4.0], [5.0, 8.0]], [[1.0, 7.0], [1.0, 7.0]], [[2.0, 7.0], [2.0, 7.0]], [[3.0, 6.0], [3.0, 6.0]],
    ]

def car_control_hard_classifier(x, y):
    p0, p1, p2 = 0.0, 0.0, 0.0
    # 
    next_abs = absolute_safe_range_hard[y]
    # print(x, y, next_abs)
    select_list = []
    if (x <= next_abs[0][1] and x >= next_abs[0][0]) or (x <= next_abs[1][1] and x >= next_abs[1][0]):
        select_list.append(1)
    if (x - 1 <= next_abs[0][1] and x - 1 >= next_abs[0][0]) or (x - 1 <= next_abs[1][1] and x - 1 >= next_abs[1][0]):
        select_list.append(0)
    if (x + 1 <= next_abs[0][1] and x + 1 >= next_abs[0][0]) or (x + 1 <= next_abs[1][1] and x + 1 >= next_abs[1][0]):
        select_list.append(2)
    index = random.choice(select_list)
    if index == 0:
        p0 = 1.0
    elif index == 1:
        p1 = 1.0
    else:
        p2 = 1.0
    
    return p0, p1, p2


def racetrack_hard_classifier_ITE(x, safe_bound):
    # convert a classifier into an ITE version when using DSE
    x, y = x, 0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        p0, p1, p2 = car_control_hard_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        # additional comparison between pi to model argmax
        a = p1 - p0
        b = p2 - p0
        c = p2 - p1
        if a <= 0: # p1 <= p0
            if b <= 0: # p2 <= p0
                index = 0
            else: # p2 > p0 >= p1
                index = 2
        else: # p1 > p0
            if c <= 0: # p2 <= p1 (p1 >= p2, p1 > p0)
                index = 1
            else: # p2 > p1 > p0
                index = 2
        if index == 0:
            x -= 1
        elif index == 1:
            x = x
        else:
            x += 1
        y += 1
    
    return trajectory_list

def racetrack_easy_1_classifier(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        p0, p1, p2 = car_control_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        if p0 == 1:
            x -= 1
        elif p1 == 1:
            x = x
        else:
            x += 1
        y += 1
    return trajectory_list


def racetrack_easy_2_classifier(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        p0, p1, p2 = car_control_classifier(x, y)
        trajectory_list.append(([x, y], [p0, p1, p2]))
        if p0 == 1:
            x -= 1
        elif p1 == 1:
            x = x
        else:
            x += 1
        y += 1
    return trajectory_list


# easier
def racetrack_easy_1(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        angle = car_control(x, y)
        trajectory_list.append(([x, y], [angle]))
        if angle <= 0.25:
            x -= 1
        elif angle <= 0.75:
            x = x
        else:
            x += 1
        y += 1
        # trajectory_list.append((x, y, angle))
        # trajectory_list.append(([x, y], [angle]))
    return trajectory_list


def car_control_sample(x, y):
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

# sample
def racetrack_easy_sample(x, safe_bound):
    x, y = x, 0.0
    steps = 20
    trajectory_list = list()
    for i in range(steps):
        angle = car_control(x, y)
        if angle <= 0.6 and angle > 0.4:
            new_x = x
            branch = 0
        else: # 0.5 vs. 0.5 select up or down
            star = random.random()
            star += (angle - 0.5) / 4 # add more probability
            if star < 0.5: # up
                new_x = x - 1
                branch = 1
            else: # down
                new_x = x + 1
                branch = 2
        new_y = y + 1
        # trajectory_list.append((x, y, angle))
        trajectory_list.append(([x, y, angle, branch], [new_x]))
        x = new_x
        y = new_y
    return trajectory_list


# # sample
# def racetrack_easy_sample_1(x, safe_bound):
#     x, y = x, 0.0
#     steps = 20
#     trajectory_list = list()
#     for i in range(steps):
#         angle = car_control(x, y)
#         if angle <= 0.6 and angle > 0.4:
#             x = x
#             branch = 0
#         else: # 0.5 vs. 0.5 select up or down
#             star = random.random()
#             star += (angle - 0.5) / 4 # add more probability
#             if star < 0.5: # up
#                 x -= 1
#                 branch = 1
#             else: # down
#                 x += 1
#                 branch = 2
#         y += 1
#         # trajectory_list.append((x, y, angle))
#         trajectory_list.append(([x, y, angle, branch], [x]))
#     return trajectory_list


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
        h = 1.0
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
    # x: [60.0, 64.0]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit = 15.0
    steps = 10
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.5:
            isOn = nn_cool_policy(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling(x)
        else: 
            h, isOn = nn_heat_policy(x)
            trajectory_list.append(([x, 1], [isOn, h]))
            h = h * h_unit
            x = warming(x, h)
        # separate two nns
        
    return trajectory_list


# thermostat
# assume the dt is 0.1
def warming_new(x, h):
    k = 0.1
    dt = 0.5
    x = x - dt * k * x + h # increase by the gap between heat and the current temperature
    return x

def cooling_new(x):
    k = 0.1
    dt = 0.5
    x = x - dt * k * x
    return x


def nn_heat_policy_new(x):
    tOff = 76.0
    if x <= tOff:
        # h = random.random() * min(1.0 - 0.5, (83.0 - 0.95 * x)/11.0 - 0.5) + 0.5
        h = min(1.0, (83.0 - 0.95*x)/15.0)
        isOn = random.random() * 0.5 + 0.5
    else:
        h = random.random() * (83.0 - 0.95*x)/15.0
        isOn = random.random() * 0.5
        
    return h, isOn


def nn_cool_policy_new(x):
    # tOff = 80.0
    tOn = 60.95
    if x <= tOn:
        isOn = random.random() * 0.5 + 0.5
    else:
        isOn = random.random() * 0.5
    return isOn


def thermostat_new(x, safe_bound):
    # x: [60.0, 64.0]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit = 15.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.5:
            isOn = nn_cool_policy_new(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling_new(x)
        else: 
            h, isOn = nn_heat_policy_new(x)
            trajectory_list.append(([x, 1], [isOn, h]))
            h = h * h_unit
            x = warming_new(x, h)
        # separate two nns
        
    return trajectory_list


def thermostat_new_cnn(x, safe_bound):
    # x: [60.0, 64.0]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit = 15.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.5:
            isOn = nn_cool_policy_new(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling_new(x)
        else: 
            h, isOn = nn_heat_policy_new(x)
            trajectory_list.append(([x, 1], [isOn, h]))
            h = h * h_unit
            x = warming_new(x, h)
        # separate two nns
        
    return trajectory_list

def nn_heat_policy_tinyinput(x):
    tOff = 76.0
    if x <= tOff:
        # h = random.random() * min(1.0 - 0.5, (83.0 - 0.95 * x)/11.0 - 0.5) + 0.5
        h = min(1.0, (85.0 - 0.95*x)/15.0)
        isOn = random.random() * 0.5 + 0.5
    else:
        h = random.random() * (85.0 - 0.95*x)/15.0
        isOn = random.random() * 0.5
        
    return h, isOn


def nn_cool_policy_tinyinput(x):
    # tOff = 80.0
    tOn = 62.0
    if x <= tOn:
        isOn = random.random() * 0.5 + 0.5
    else:
        isOn = random.random() * 0.5
    return isOn


def thermostat_new_tinyinput(x, safe_bound):
    # x: [60.0, 60.1]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit = 15.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.5:
            isOn = nn_cool_policy_tinyinput(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling_new(x)
        else: 
            h, isOn = nn_heat_policy_tinyinput(x)
            trajectory_list.append(([x, 1], [isOn, h]))
            h = h * h_unit
            x = warming_new(x, h)
        # separate two nns
        
    return trajectory_list


def thermostat_new_40(x, safe_bound):
    # x: [60.0, 64.0]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit = 15.0
    steps = 40
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.5:
            isOn = nn_cool_policy_new(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling_new(x)
        else: 
            h, isOn = nn_heat_policy_new(x)
            trajectory_list.append(([x, 1], [isOn, h]))
            h = h * h_unit
            x = warming_new(x, h)
        # separate two nns
        
    return trajectory_list


def nn_heat_low_policy_new(x):
    tOff = 77.0
    if x <= tOff:
        # h = random.random() * min(1.0 - 0.5, (83.0 - 0.95 * x)/11.0 - 0.5) + 0.5
        h = min(1.0, (83.0 - 0.95*x)/15.0)
        isOn = random.random() * 0.3333 + 0.3333
    else:
        h = random.random() * (83.0 - 0.95*x)/15.0
        isOn = random.random() * 0.333
        
    return h, isOn


def nn_heat_high_policy_new(x):
    tOff = 77.0
    if x <= tOff:
        # h = random.random() * min(1.0 - 0.5, (83.0 - 0.95 * x)/11.0 - 0.5) + 0.5
        h = min(1.0, (83.0 - 0.95*x)/15.0)
        isOn = random.random() * 0.333 + 0.6666
    else:
        h = random.random() * (83.0 - 0.95*x)/15.0
        isOn = random.random() * 0.333
        
    return h, isOn


def nn_cool_policy_new_3branches(x):
    # tOff = 80.0
    tOnHigh = 60.0
    tOnLow = 64.0
    if x <= tOnHigh:
        isOn = random.random() * 0.333 + 0.6666
    elif x <= tOnLow:
        isOn = random.random() * 0.3333 + 0.3333
    else:
        isOn = random.random() * 0.33333

    return isOn


def thermostat_new_3branches(x, safe_bound):
    # x: [60.0, 64.0]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit_low = 10.0
    h_unit_high = 15.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.33:
            isOn = nn_cool_policy_new(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling_new(x)
        else: 
            if isOn <= 0.66: # Low Heat
                h, isOn = nn_heat_low_policy_new(x)
                trajectory_list.append(([x, 1], [isOn, h]))
                h = h * h_unit_low
                x = warming_new(x, h)
            else: # High Heat
                h, isOn = nn_heat_high_policy_new(x)
                trajectory_list.append(([x, 2], [isOn, h]))
                h = h * h_unit_high
                x = warming_new(x, h)

        # separate two nns
        
    return trajectory_list


def nn_heat_policy_new_unsafe25(x):
    tOff = 76.0
    if x <= tOff:
        # h = random.random() * min(1.0 - 0.5, (83.0 - 0.95 * x)/11.0 - 0.5) + 0.5
        h = min(1.0, (83.0 - 0.95*x)/15.0)
        isOn = random.random() * 0.5 + 0.5
    else:
        h = random.random() * (83.0 - 0.95*x)/15.0
        isOn = random.random() * 0.5
    if random.random() < 0.25:
        isOn = 1 - isOn
    return h, isOn


def nn_cool_policy_new_unsafe25(x):
    # tOff = 80.0
    tOn = 60.95
    if x <= tOn:
        isOn = random.random() * 0.5 + 0.5
    else:
        isOn = random.random() * 0.5
    if random.random() < 0.25:
        isOn = 1 - isOn
    return isOn


def thermostat_new_unsafe25(x, safe_bound):
    # x: [60.0, 64.0]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit = 15.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.5:
            isOn = nn_cool_policy_new_unsafe25(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling_new(x)
        else: 
            h, isOn = nn_heat_policy_new_unsafe25(x)
            trajectory_list.append(([x, 1], [isOn, h]))
            h = h * h_unit
            x = warming_new(x, h)
        # separate two nns
        
    return trajectory_list


def nn_heat_policy_new_unsafe50(x):
    tOff = 76.0
    if x <= tOff:
        # h = random.random() * min(1.0 - 0.5, (83.0 - 0.95 * x)/11.0 - 0.5) + 0.5
        h = min(1.0, (83.0 - 0.95*x)/15.0)
        isOn = random.random() * 0.5 + 0.5
    else:
        h = random.random() * (83.0 - 0.95*x)/15.0
        isOn = random.random() * 0.5
    if random.random() < 0.5:
        isOn = 1 - isOn
    return h, isOn


def nn_cool_policy_new_unsafe50(x):
    # tOff = 80.0
    tOn = 60.95
    if x <= tOn:
        isOn = random.random() * 0.5 + 0.5
    else:
        isOn = random.random() * 0.5
    if random.random() < 0.5:
        isOn = 1 - isOn
    return isOn


def thermostat_new_unsafe50(x, safe_bound):
    # x: [60.0, 64.0]
    # safe area for x: [55.0, 83.0]
    isOn = 0.0
    h_unit = 15.0
    steps = 20
    trajectory_list = list()
    for i in range(steps): 
        # isOn comes from the previous step
        if isOn <= 0.5:
            isOn = nn_cool_policy_new_unsafe50(x) # isOn should have a sigmoid
            trajectory_list.append(([x, 0], [isOn, 0.0]))
            x = cooling_new(x)
        else: 
            h, isOn = nn_heat_policy_new_unsafe50(x)
            trajectory_list.append(([x, 1], [isOn, h]))
            h = h * h_unit
            x = warming_new(x, h)
        # separate two nns
        
    return trajectory_list


def aircraft_distance(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) **2


def nn_left(x1, y1):
    return 0.5

def nn_right(x1, y1):
    return 0.5

# x1, y1: the aircraft in control
# x2, y2: the aircraft keep flying horizontally
# stage: ['CRUISE', 'LEFT', 'STRAIGHT', 'RIGHT']
# x1 in [12, 16], y1 = -15
# x2, y2 = (0, 0)
# safe_distance_square: 20
def aircraft_collision(x, safe_bound):
    stage = 'CRUISE'
    critical_distance_square = 212
    steps = 15
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    x_unit = 10.0
    straight_speed = 5.0
    step = 0
    x1_co = 0.0
    trajectory_list = list()
    min_ = 1000000
    for i in range(steps):
        # print(aircraft_distance(x1, y1, x2, y2))
        min_ = min(aircraft_distance(x1, y1, x2, y2), min_)
        if stage == 'CRUISE':
            if aircraft_distance(x1, y1, x2, y2) <= critical_distance_square:
                stage = 'LEFT'
                step = 0
        elif stage == 'LEFT':
            x1_co = nn_left(x1, y1)
            trajectory_list.append(([x1, y1, 0], [x1_co]))
            x1 = x1 - x1_co * x_unit
            step += 1
            if step > 3:
                stage = 'STRAIGHT'
                step = 0
        elif stage == 'STRAIGHT':
            step += 1
            if step > 2:
                stage = 'RIGHT'
                step = 0
        elif stage == 'RIGHT':
            x1_co = nn_right(x1, y1)
            trajectory_list.append(([x1, y1, 1], [x1_co]))
            x1 = x1 + x1_co * x_unit
            step += 1
            if step > 3:
                stage = 'CRUISE'
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
    # exit(0)
    # print(min_)
    
    return trajectory_list


def f_nn_control_stage(x1, y1, x2, y2, step, stage):
    critical_distance_square = 200
    if stage == 0.0:
        if aircraft_distance(x1, y1, x2, y2) <= critical_distance_square:
            stage = 0.3
            step = 0
    elif stage == 0.3:
        step += 1
        if step > 3:
            stage = 0.6
            step = 0
    elif stage == 0.6:
        step += 1
        if step > 2:
            stage = 0.9
            step = 0
    elif stage == 0.9:
        step += 1
        if step > 3:
            stage = 0.0
    
    return stage, step


# x1, y1: the aircraft in control
# x2, y2: the aircraft keep flying horizontally
# stage: ['CRUISE', 'LEFT', 'STRAIGHT', 'RIGHT']
# x1 in [12, 16], y1 = -15
# x2, y2 = (0, 0)
# safe_distance_square: 20
# <= 0.25: CRUISE; <=0.5: LEFT; <=0.75: STRAIGHT; <=1.0: RIGHT
def aircraft_collision_refined(x, safe_bound):
    stage = 0.0
    steps = 15
    straight_speed= 5.0
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    step = 0
    trajectory_list = list()
    for i in range(steps):
        stage, step = f_nn_control_stage(x1, y1, x2, y2, step, stage)
        trajectory_list.append(([x1, y1, x2, y2, stage], [stage]))
        if stage <= 0.25:
            pass
        elif stage <= 0.5:
            x1 = x1 - 5.0
        elif stage <= 0.75:
            pass
        else:
            x1 =  x1 + 5.0
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
    
    return trajectory_list


def classifier_stage(x1, y1, x2, y2, stage, step):
    p0, p1, p2, p3 = 0, 0, 0, 0
    critical_distance_square = 250
    if stage == 0:
        if aircraft_distance(x1, y1, x2, y2) <= critical_distance_square:
            stage = 1
            step = 0
    elif stage == 1:
        step += 1
        if step > 3:
            stage = 2
            step = 0
    elif stage == 2:
        step += 1
        if step > 2:
            stage = 3
            step = 0
    elif stage == 3:
        step += 1
        if step > 3:
            stage = 0
    
    if stage == 0: 
        p0 = 1
    elif stage == 1:
        p1 = 1
    elif stage == 2:
        p2 = 1
    else:
        p3 = 1
    
    return p0, p1, p2, p3, stage, step, aircraft_distance(x1, y1, x2, y2)


# the output of the classifier is a vector of four
# p0, p1, p2, p3
def aircraft_collision_refined_classifier(x, safe_bound):
    stage = 0.0
    steps = 15
    straight_speed= 5.0
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    step = 0
    trajectory_list = list()
    for i in range(steps):
        # assign stage based on the branch
        p0, p1, p2, p3, stage, step, aircraft_distance = classifier_stage(x1, y1, x2, y2, stage, step)
        trajectory_list.append(([x1, y1, x2, y2, stage], [p0, p1, p2, p3]))
        if p0 == 1:
            pass # stage = 0
        elif p1 == 1:
            x1 = x1 - 5.0 # stage = 1
        elif p2 == 1:
            pass # stage = 2
        else:
            x1 =  x1 + 5.0 # stage = 3
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
        if aircraft_distance < 40:
            print(f"Collision: {aircraft_distance}")
    
    return trajectory_list


def classifier_stage_new(x1, y1, x2, y2, stage, step):
    p0, p1, p2, p3 = 0, 0, 0, 0
    critical_distance_square = 250
    # print(f"input state: {aircraft_distance(x1, y1, x2, y2), stage, step}")
    if stage == 0:
        if aircraft_distance(x1-5, y1+5, x2+5, y2) < 40:
            stage = 3 
            step = 0
        elif aircraft_distance(x1, y1+5, x2+5, y2) < 40 or aircraft_distance(x1, y1, x2, y2) <= critical_distance_square:
            stage = 1
            step = 0
    elif stage == 1:
        step += 1
        if step > 3:
            stage = 2
            step = 0
    elif stage == 2:
        step += 1
        if step > 2:
            stage = 3
            step = 0
    elif stage == 3:
        step += 1
        if step > 3:
            stage = 0
    
    if stage == 0: 
        p0 = 1
    elif stage == 1:
        p1 = 1
    elif stage == 2:
        p2 = 1
    else:
        p3 = 1
    # print(f"output state: {stage, step}")
    
    return p0, p1, p2, p3, stage, step, aircraft_distance(x1, y1, x2, y2)


def aircraft_collision_new(x, safe_bound):
    stage = 0.0
    steps = 15
    straight_speed= 5.0
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    step = 0
    trajectory_list = list()
    for i in range(steps):
        # assign stage based on the branch
        p0, p1, p2, p3, stage, step, aircraft_distance = classifier_stage_new(x1, y1, x2, y2, stage, step)
        trajectory_list.append(([x1, y1, x2, y2, stage], [p0, p1, p2, p3]))        
        # print(x1, y1, x2, y2, stage, aircraft_distance)
        if p0 == 1:
            pass # stage = 0
        elif p1 == 1:
            x1 = x1 - 5.0 # stage = 1
        elif p2 == 1:
            pass # stage = 2
        else:
            x1 =  x1 + 5.0 # stage = 3
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
        if aircraft_distance < 40:
            print(f"Collision: {aircraft_distance}")
    
    return trajectory_list


def aircraft_collision_new_1(x, safe_bound): # TODO: an updated version using steps
    stage = 0.0
    steps = 15
    straight_speed= 5.0
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    step = 0
    trajectory_list = list()
    for i in range(steps):
        # assign stage based on the branch
        before_step = step
        p0, p1, p2, p3, stage, step, aircraft_distance = classifier_stage_new(x1, y1, x2, y2, stage, step)
        trajectory_list.append(([x1, y1, x2, y2, stage, before_step], [p0, p1, p2, p3, step]))        
        # print(x1, y1, x2, y2, stage, aircraft_distance)
        if p0 == 1:
            pass # stage = 0
        elif p1 == 1:
            x1 = x1 - 5.0 # stage = 1
        elif p2 == 1:
            pass # stage = 2
        else:
            x1 =  x1 + 5.0 # stage = 3
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
        if aircraft_distance < 40:
            print(f"Collision: {aircraft_distance}")
    
    return trajectory_list


def aircraft_collision_new_1_cnn(x, safe_bound): # TODO: an updated version using steps
    stage = 0.0
    steps = 15
    straight_speed= 5.0
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    step = 0
    trajectory_list = list()
    for i in range(steps):
        # assign stage based on the branch
        before_step = step
        p0, p1, p2, p3, stage, step, aircraft_distance = classifier_stage_new(x1, y1, x2, y2, stage, step)
        trajectory_list.append(([x1, y1, x2, y2, stage, before_step], [p0, p1, p2, p3, step]))        
        # print(x1, y1, x2, y2, stage, aircraft_distance)
        if p0 == 1:
            pass # stage = 0
        elif p1 == 1:
            x1 = x1 - 5.0 # stage = 1
        elif p2 == 1:
            pass # stage = 2
        else:
            x1 =  x1 + 5.0 # stage = 3
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
        if aircraft_distance < 40:
            print(f"Collision: {aircraft_distance}")
    
    return trajectory_list


def classifier_stage_new_unsafe25(x1, y1, x2, y2, stage, step):
    p0, p1, p2, p3 = 0, 0, 0, 0
    critical_distance_square = 250
    # print(f"input state: {aircraft_distance(x1, y1, x2, y2), stage, step}")
    if stage == 0:
        if aircraft_distance(x1-5, y1+5, x2+5, y2) < 40:
            stage = 3 
            step = 0
        elif aircraft_distance(x1, y1+5, x2+5, y2) < 40 or aircraft_distance(x1, y1, x2, y2) <= critical_distance_square:
            stage = 1
            step = 0
    elif stage == 1:
        step += 1
        if step > 3:
            stage = 2
            step = 0
    elif stage == 2:
        step += 1
        if step > 2:
            stage = 3
            step = 0
    elif stage == 3:
        step += 1
        if step > 3:
            stage = 0
    
    if stage == 0: 
        p0 = 1
        if random.random() < 0.25:
            p0 = 0
            another_branch = random.random()
            if another_branch < 0.33:
                p1 = 1
            elif another_branch < 0.66:
                p2 = 1
            else:
                p3 = 1
    elif stage == 1:
        p1 = 1
        if random.random() < 0.25:
            p1 = 0
            another_branch = random.random()
            if another_branch < 0.33:
                p0 = 1
            elif another_branch < 0.66:
                p2 = 1
            else:
                p3 = 1
    elif stage == 2:
        p2 = 1
        if random.random() < 0.25:
            p2 = 0
            another_branch = random.random()
            if another_branch < 0.33:
                p0 = 1
            elif another_branch < 0.66:
                p1 = 1
            else:
                p3 = 1
    else:
        p3 = 1
        if random.random() < 0.25:
            p3 = 0
            another_branch = random.random()
            if another_branch < 0.33:
                p0 = 1
            elif another_branch < 0.66:
                p1 = 1
            else:
                p2 = 1
    # print(f"output state: {stage, step}")
    
    return p0, p1, p2, p3, stage, step, aircraft_distance(x1, y1, x2, y2)


def aircraft_collision_new_1_unsafe25(x, safe_bound): # TODO: an updated version using steps
    stage = 0.0
    steps = 15
    straight_speed= 5.0
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    step = 0
    trajectory_list = list()
    for i in range(steps):
        # assign stage based on the branch
        before_step = step
        p0, p1, p2, p3, stage, step, aircraft_distance = classifier_stage_new_unsafe25(x1, y1, x2, y2, stage, step)
        trajectory_list.append(([x1, y1, x2, y2, stage, before_step], [p0, p1, p2, p3, step]))        
        # print(x1, y1, x2, y2, stage, aircraft_distance)
        if p0 == 1:
            pass # stage = 0
        elif p1 == 1:
            x1 = x1 - 5.0 # stage = 1
        elif p2 == 1:
            pass # stage = 2
        else:
            x1 =  x1 + 5.0 # stage = 3
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
        # if aircraft_distance < 40:
        #     print(f"Collision: {aircraft_distance}")
    
    return trajectory_list


def classifier_stage_refined_classifier_ITE(x1, y1, x2, y2, stage, step):
    p0, p1, p2, p3 = 0, 0, 0, 0
    critical_distance_square = 250
    if stage == 0:
        if aircraft_distance(x1, y1, x2, y2) <= critical_distance_square:
            stage = 1
            step = 0
    elif stage == 1:
        step += 1
        if step > 3:
            stage = 2
            step = 0
    elif stage == 2:
        step += 1
        if step > 2:
            stage = 3
            step = 0
    elif stage == 3:
        step += 1
        if step > 3:
            stage = 0
    
    if stage == 0: 
        p0 = 1
    elif stage == 1:
        p1 = 1
    elif stage == 2:
        p2 = 1
    else:
        p3 = 1
    
    return p0, p1, p2, p3, stage, step


def aircraft_collision_refined_classifier_ITE(x, safe_bound):
    stage = 0.0
    steps = 15
    straight_speed = 5.0
    x1, y1, x2, y2 = x, -15.0, 0.0, 0.0
    step = 0
    trajectory_list = list()
    for i in range(steps):
        p0, p1, p2, p3, stage, step = classifier_stage_refined_classifier_ITE(x1, y1, x2, y2, stage, step)
        trajectory_list.append(([x1, y1, x2, y2, stage], [p0, p1, p2, p3]))
        if p0 == 1:
            pass
        elif p1 == 1:
            x1 = x1 - 5.0
        elif p2 == 1:
            pass
        else:
            x1 = x1 + 5.0
        y1 = y1 + straight_speed
        x2 = x2 + straight_speed
    
    return trajectory_list


def NN_green_walker(x):
    if x >= 2:
        right = random.random() * 0.5
        straight = 1 - right
    else:
        right = random.random()
        straight = 1 - right
    return right, straight

def NN_red_walker(x):
    if x >= 2:
        # no right
        stop = random.random()
        right = random.random()*stop
        straight = random.random() * right
    else:
        stop = random.random()
        right = random.random()
        straight = random.random() * min(stop, right)
    return right, straight, stop

def argmax(x):
    return x.index(max(x))


# the additional benchmark
def perception(pedestrian_number, safe_bound):
    # pedestrian_number: [2, +oo)
    # use a small coverage over each img as symbolic img, focus on the robustness of image
    # use the entire pedestrian_number range as symbolic pedestrian_number
    # GREEN, RED = 0, 1
    # green, red = NN_img(img)
    # light = argmax([green, red])
    # if light == GREEN:
    #     # angle, acceleration
    #     right, straight = NN_green_walker(pedestrian_number)
    #     stop = 0.0
    # elif light == RED:
    #     right, straight, stop = NN_red_walker(pedestrian_number)
    # action = argmax(right, straight, stop)
    # if 
    
    # # safety check
    # assert(((label(img) == 'RED') != (action == 'STRAINT'))
    #         and
    #         ((pedestrian_number > 2) != (action == 'RIGHT')))
    # # what's the quantitative distance function?
    trajectory_list = list()
    right, straight = NN_green_walker(pedestrian_number)
    stop = 0.0
    trajectory_list.append(([pedestrian_number, 0], [right, straight, stop]))
    right, straight, stop = NN_red_walker(pedestrian_number)
    trajectory_list.append(([pedestrian_number, 1], [right, straight, stop]))

    return trajectory_list


def cart_pole(x): # [-4.8, 4.8]
    x_dot = 0 # [-inf, inf]
    theta = 0 # [-0.418, 0.418]
    theta_dot = 0 # [-inf, inf]
    # They are initialized to be [-0.05, 0.05]^4
    # safe area: theta \in [-0.209, 0.209]

    for i in range(200):
        action = NN(x, x_dot, theta, theta_dot)
        if action <= 0.5:
            force = -10.0
        else:
            force = 10.0
        costheta = 1 # 1.0 + theta # math.cos(theta)
        sintheta = theta # math.sin(theta)
        temp = (force + 0.05*theta_dot**2 * sintheta)/1.1
        thetaacc = 2* (9.8*sintheta - costheta*temp) / (4/3 - 0.1*costheta**2 / 1.1)
        xacc = temp - 0.05 * thetaacc * costheta / 1.1
        x = x + 0.02*x_dot
        x_dot = x_dot + 0.02*xacc
        theta = theta + 0.02 * theta_dot
        theta_dot = theta_dot + 0.02 * thetaacc

        # the linear approximation version
        temp = (force + 0.05*theta_dot**2 * sintheta)/1.1
        thetaacc = 2* (9.8*sintheta - 1*temp) / (4/3 - 0.1*1**2 / 1.1)
        xacc = temp - 0.05 * thetaacc * 1 / 1.1
        x = x + 0.02*x_dot
        x_dot = x_dot + 0.02*xacc
        theta = theta + 0.02 * theta_dot
        theta_dot = theta_dot + 0.02 * thetaacc
        
# the additional benchmark
def perception(img,  x, safe_bound):
    p0, p1 = NN_img(img)
    light = argmax(p0, p1)
    if light == 0:
        right, straight, stop = NN_green_walker(x)
    elif light == 1:
        right, straight, stop = NN_red_walker(x)
    action = argmax(right, straight, stop)
    
    # safety check
    if img==red:
        if walker > 2:
            res =  stop
        else:
            res = right
    if img==green:
        if walker > 2:
            res = straight
        else:
            res = right or straight
    return 


def pcc():
    return