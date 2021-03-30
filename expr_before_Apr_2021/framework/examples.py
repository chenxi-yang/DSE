t = 0.0
h0 = domain(10.0, 12.0)
v0 = 0.0
g = 9.8
while(h0 > 3):
    t_down = sqrt(2*h0/g)
    t_up = k/g * (v0 + g*t_down)
    t = t + t_down + t_up
    h0 = 0.5*g*t_up*t_up
    v0 = 0.0

    if h0 < 5.0:
        v0 = 7.0
    else:
        v0 = 0.0

return t




def benchmark_test_disjunction(count, h, theta):
    count = 0
    while(h > 10.0):
        h = h + 0.01
        if (h <= theta):
            if (h <= theta - 0.001):
                h = 2*h
            else:
                h = 3*h
        else:
            h = h
        count += 1
    if (h <= 3*theta - 0.001):
        h = h
    else:
        h = h + 10

    return count


def benchmark_thermostat_wo_loop(theta, x):
    for i in range(40):
        if isOn <= 0.5:
            x = x - 0.1*(x-60)
            if x <= theta:
                isOn = 1.0
            else:
                isOn = 0.0
        else:
            x = x - 0.1*(x-60) + 5.0
            if x <= 80.0:
                isOn = 1.0
            else:
                isOn = 0.0
    
    return x


def benchmark_thermostat_w_loop(theta, x):

    while(x<=74.0):
        if isOn <= 0.5:
            x = x - 0.1*(x-60)
            if x <= theta:
                isOn = 1.0
            else:
                isOn = 0.0
        else:
            x = x - 0.1*(x-60) + 5.0
            if x <= 77.0:
                isOn = 1.0
            else:
                x = 0.0
    
    while(-x<=-78.0):
        if isOn <= 0.5:
            x = x - 0.1*(x-60)
            if x <= theta:
                isOn = 1.0
            else:
                isOn = 0.0
        else:
            x = x = x - 0.1*(x-60) + 5.0
            if x <= 77.0:
                isOn = 1.0
            else:
                isOn = 0.0    

    return x  


def aircraft_collision_wo_loop(theta, v1, v2, delay, delay2):
    # safe property: dist(x1, y1, x2, y2)
    for i in range(50):
        if stage <= 1.0:
            x1, y1, x2, y2 = move_straight(x1, y1, x2, y2, v1, v2)
            if dist(x1, y1, x2, y2) < dangerDist:
                stage = 1.5
                steps = 0
        elif stage <= 1.5:
            x1, y1, x2, y2 = move_left(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps <= theta:
                pass
            else:
                stage = 2.5
                steps = 0
        elif stage <= 2.5:
            x1, y1, x2, y2 = move_straight(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps <= delay2:
                pass
            else:
                stage = 3.5
                steps = 0
        else:
            x1, y1, x2, y2 = move_right(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps <= theat:
                pass
            else:
                stage = 0.5
    
    return delay*2 + delay2


def aircraft_collision_w_loop(theta, v1, v2, delay, delay2):

    while dist(x1, y1, x2, y2) <= 30:
        if stage <= 1.0:
            x1, y1, x2, y2 = move_straight(x1, y1, x2, y2, v1, v2)
            if dist(x1, y1, x2, y2) < dangerDist:
                stage = 1.5
                steps = 0
        elif stage <= 1.5:
            x1, y1, x2, y2 = move_left(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps <= theta:
                pass
            else:
                stage = 2.5
                steps = 0
        elif stage <= 2.5:
            x1, y1, x2, y2 = move_straight(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps <= delay2:
                pass
            else:
                stage = 3.5
                steps = 0
        else:
            x1, y1, x2, y2 = move_right(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps <= theat:
                pass
            else:
                stage = 0.5
    
    return delay*2 + delay2


def bouncing_ball_w_loop(h):
    # safety property: h
    h = domain.Interval(8.0, 12.0)
    v0 = 0.0
    while h >= 3:
        t_down = (2 * h/g) ** (-1/2)
        t_up = k g *(v0 + g * t_down)
        t = t + t_down + t_up
        h = 1/2 * g * (t_up ** 2)
        if h < theta:
            v0 = 7.0
        else:
            v0 = 0.0

    return t





        




        

