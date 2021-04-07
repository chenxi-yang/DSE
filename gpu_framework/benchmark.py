import math

def thermostat(lin):
    x = lin
    x_max = -100000
    x_min = 100000
    tOff = 78.0
    tOn = 66.0
    isOn = 0.0

    for i in range(40):
        if isOn <= 0.5: # ifblock1
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
        # print(f"x: {x}, isOn:{isOn}")
        x_min = min(x, x_min)
        x_max = max(x, x_max)
    # exit(0)
    
    return lin, x, x_min, x_max


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
    goal_positition = 0.5
    min_speed = -0.07
    max_speed = 0.07

    while p <= goal_position:
        if p <= min_position:
            p = min_position
            v = 0
        u = acceleration(p, v)
        reward = reward + (-0.1) * u  * u
        v = 0.0015 * v - 0.0025 * math.cos(3 * p)
        if v <= min_speed: v = min_speed
        elif v >= max_speed: v = max_speed
        else: v = v
        p = p + v
    
    reward += 100
        
    return p0, v, reward, reward


