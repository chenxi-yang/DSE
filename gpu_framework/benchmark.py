import math

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

def mountain_car(p0):
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
        if i > 1000:
            break
        if p <= min_position:
            p = min_position
            v = 0
        u = acceleration(p, v)
        trajectory_list.append((p, v, u))

        # reward = reward + (-0.1) * u  * u
        v = v + 0.0015 * u - 0.0025 * math.cos(3 * p)
        if v <= min_speed: 
            v = min_speed
        else:
            if v <= max_speed:
                v = v # skip()
            else:
                v = max_speed
        p = p + v

        i += 1
    
    # if p < goal_position:
    #     reward = reward
    # else:
    #     reward += 100
    # exit(0)
        
    return trajectory_list


def mountain_car(p, v):
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


def mountain_car(p, v):
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
