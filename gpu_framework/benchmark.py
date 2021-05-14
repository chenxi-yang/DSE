import math
import random

def thermostat(lin):
    x = lin
    x_max = -100000
    x_min = 100000
    tOff = 78.0
    tOn = 66.0
    isOn = 0.0
    trajectory_list = list()

    for i in range(40):
        state = (lin, x)
        if isOn <= 0.5: # ifblock1
            state = (lin, x)
            x = x - 0.1 * (x - lin)
            if x <= tOn: # ifelse_tOn
                isOn = 1.0
            else:
                isOn = 0.0
        else: # ifblock2
            x = x - 0.1 * (x - lin) + 5.0
            if x <= tOff:
                isOn = 1.0
            else:
                isOn = 0.0
        trajectory_list.append((state[0], state[1], x))
        # print(f"x: {x}, isOn:{isOn}")
        x_min = min(x, x_min)
        x_max = max(x, x_max)
    # exit(0)
    
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
    if v <= 0.0:
        # u = - 0.8
        u = - safe_bound + random.uniform(0.00001, 0)
    else:
        u = safe_bound + random.uniform(-0.00001, 0)
    return u


def mountain_car(p0, safe_bound):
    # pos in [-1.2, 0.6]
    # initial range: [-0.6, -0.4]
    v = 0
    p = p0
    min_position = -1.2
    goal_position = 0.5
    min_speed = -0.07
    max_speed = 0.07
    reward = 0
    i = 0
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
    
    return trajectory_list


def unsound_1(x, safe_bound):
    # x in [-5, 5]
    a = 2.0
    b = 20.0
    bar = 1.0
    z = 0.0
    trajectory_list = list()

    y = a * x + b
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

    
