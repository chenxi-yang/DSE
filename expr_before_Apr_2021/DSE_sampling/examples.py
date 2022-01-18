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


def benchmark_thermostat_nn(lin):
    def NN(x, y):
        return Linear(Sigmoid(Linear(x, y)))
    #  safe constraint of x's trajectory: [57.02, 83.20]
    x = lin
    tOff = ??((60.0, 67.0))
    tOn, isOn = 80.0, 0.0
    for i in range(40):
        if isOn <= 0.5:
            x = NN(x, lin) # x - 0.1*(x-60)
            if x <= tOff:
                isOn = 1.0
        else:
            x = x - 0.1*(x-60) + 5.0
            if x <= tOn:
                isOn = 1.0
            else:
                isOn = 0.0
        assert (x > 58.43 and x < 83.19)
    
    return x


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


# def benchmark_thermostat_w_loop(theta, x):

#     while(x<=74.0):
#         if isOn <= 0.5:
#             x = x - 0.1*(x-60)
#             if x <= theta:
#                 isOn = 1.0
#             else:
#                 isOn = 0.0
#         else:
#             x = x - 0.1*(x-60) + 5.0
#             if x <= 77.0:
#                 isOn = 1.0
#             else:
#                 x = 0.0
    
#     while(-x<=-78.0):
#         if isOn <= 0.5:
#             x = x - 0.1*(x-60)
#             if x <= theta:
#                 isOn = 1.0
#             else:
#                 isOn = 0.0
#         else:
#             x = x = x - 0.1*(x-60) + 5.0
#             if x <= 77.0:
#                 isOn = 1.0
#             else:
#                 isOn = 0.0    

#     return x  

def benchmark_thermostat_w_loop(theta, x):
    # x = ??(62.0, 72.0)
    
    tOff = ??(53.0, 60.0)
    tOn = 77.0
    curT = ??(55.0, 62.0)  # room temperature
    while(x<=74.0):
        if isOn <= 0.5:
            x = x - 0.1*(x - curT)
            if x <= tOff:
                isOn = 1.0
        else:
            x = x - 0.1*(x - curT) + 5.0
            if x > 77.0:
                isOn = 0.0
    
    while(x > 78.0):
        if isOn <= 0.5:
            x = x - 0.1*(x-curT)
            if x <= tOff:
                isOn = 1.0
            else:
                isOn = 0.0
        else:
            x = x - 0.1*(x-curT) + 5.0
            if x <= 77.0:
                isOn = 1.0
            else:
                isOn = 0.0    

    return x 


def aircraft_collision(theta, v1, v2, x1, y1, x2, y2):
# safe constraint of dist(x1, y1, x2, y2): (2.86, INFINITY)
    ...
    safeDist = ??(4.7, 5.8)
    delay = ??(4.0, 6.0)
    delay2 = 10.0
    for i in range(50):
        if stage == CRUISE:
            x1, y1, x2, y2 = move_straight(x1, y1, x2, y2, v1, v2)
            if dist(x1, y1, x2, y2) < safeDist:
                stage = LEFT
                steps = 0
        elif stage == LEFT:
            x1, y1, x2, y2 = move_left(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps > delay:
                stage = STRAIGHT
                steps = 0
        elif stage == 3.0:
            x1, y1, x2, y2 = move_straight(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps > delay2:
                stage = RIGHT
                steps = 0
        else:
            x1, y1, x2, y2 = move_right(x1, y1, x2, y2, v1, v2)
            steps = steps + 1
            if steps > delay:
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


def bouncing_ball_with_loop(h):
# safety constraint of h: (3.0, 9.3)
    bound = ??(1.0, 6.0)
    v0 = 0.0
    while h >= 4:
        t_down = (2 * h/g) ** (-0.5)
        t_up = k g *(v0 + g * t_down)
        t = t + t_down + t_up
        h = 0.5 * g * (t_up ** 2)
        if h < theta:
            v0 = 7.0
        else:
            v0 = 0.0

    return t


def bouncing_ball_deep_branch(h):
# safe constraint of h: (2.368, 7.04)
    bound = ??(1.0, 9.0)
    g = 9.8
    v0 = 0.0
    for i in range(40):
        t_down = (2 * h/g) ** (-0.5)
        t_up = k g *(v0 + g * t_down)
        t = t + t_down + t_up
        h = 0.5 * g * (t_up ** 2)
        if h < bound:
            v0 = 7.0
        else:
            v0 = 0.0

    return t

def atrial_fibrilation_deep_branch(u, w, v, s):
# safe constrait of u: (-0.062, 0.99342)
    # jfi1 = ??(), bound - 0.0055
    # epi_to1 = ??(),  bound
    bound = ??(3.0, 4.0)
    for t in range(8):
        if STAGE == MODE1:
            u = u + delta_t*((stim - jfi1) - ((u/epi_to1) + jsi1))
            w = w + delta_t*((1.0 -(u/EPI_TWINF) - w)/(EPI_TW1M + (EPI_TW2M - EPI_TW1M) * (1/(1+exp(-2*EPI_KWM*(u - EPI_UWM))))))
            v = v + delta_t*((1.0 - v)/EPI_TV1M)
            s = s + delta_t*(((1/(1+exp( -2 * EPI_KS * (u - EPI_US) ))) - s)/EPI_TS1)
            if u >= bound:
                STAGE == MODE2
        elif STAGE == MODE2:
            u = u + delta_t*((stim - jfi2) - ((u/EPI_TO2) + jsi2))
            w = w + delta_t*((0.94-w)/(EPI_TW1M + (EPI_TW2M - EPI_TW1M) * (1/(1+exp(-2*EPI_KWM*(u - EPI_UWM))))))
            v = v + delta_t*(-v/EPI_TV2M)
            s = s + delta_t*(((1/(1+exp( -2 * EPI_KS * (u - EPI_US) ))) - s)/EPI_TS1)
            if u >= 0.013:
                STAGE == MODE3
        elif STAGE == MODE3:
            u = u + delta_t*((stim - jfi3) - (1.0/(EPI_TSO1+((EPI_TSO2- EPI_TSO1)*(1/(1+exp(-2*EPI_KSO*(u- EPI_USO)))))) + (0 - (w * s)/EPI_TSI)))
            w = w + delta_t*(-w/EPI_TWP)
            v = v + delta_t*(-v/EPI_TV2M)
            s = s + delta_t*(((1/(1+exp( -2 * EPI_KS * (u - EPI_US) ))) - s)/EPI_TS2)
            if u >= 0.3:
                STAGE == MODE4
        elif STAGE == MODE4:
            u = u + delta_t*(stim - (0 - v * (u - EPI_THV) * (EPI_UU - u)/EPI_TFI)) - ((1.0 / (EPI_TSO1+((EPI_TSO2 - EPI_TSO1)*(1/(1+exp(-2*EPI_KSO*(u- EPI_USO))))))) + ( 0 - (w * s)/EPI_TSI))
            w = w + delta_t*(-w/EPI_TWP)
            v = v + delta_t*(-v/EPI_TVP)
            s = s + delta_t*(((1/(1+exp( -2 * EPI_KS * (u - EPI_US) ))) - s)/EPI_TS2)
            if u <= 2.0:
                STAGE == MODE3
        
    return u


def atrial_fibrilation_mode1_loop(u, w, v, s):
# safe constraint of u: (-0.008, 0.99342)
    epi_to1 = ??(2.5, 5.0)
    while u <= 1.0:
        u = u + delta_t*((stim - jfi1) - ((u/epi_to1) + jsi1))
        w = w + delta_t*((1.0 -(u/EPI_TWINF) - w)/(EPI_TW1M + (EPI_TW2M - EPI_TW1M) * (1/(1+exp(-2*EPI_KWM*(u - EPI_UWM))))))
        v = v + delta_t*((1.0 - v)/EPI_TV1M)
        s = s + delta_t*(((1/(1+exp( -2 * EPI_KS * (u - EPI_US) ))) - s)/EPI_TS1)
        
    return u


def electronic_oscillator_deep_branch(x, y, z, omega1, omega2, tau):
# safe constraint of y: (-5.7658, 5.0)
    constant = ??(-1.5, 1.5)
    bound = ??(1.5, 2.5)

    for i in range(20):
        if STAGE == MODE1:
            x = x + delta_t*(- ax * sin(omega * tau))
            y = y + delta_t*(- ay * sin( (omega1 + constant) * tau) * sin(omega2)*2)
            z = z + delta_t*(- az * sin( (omega2 + 1.0) * tau) * cos(omega1)*2)
            omega1 = omega1 + delta_t*(0 - 0.5 * omega1)
            omega2 = omega2 + delta_t*(0 - omega2)
            tau = tau + delta_t*(1.0)
            if tau >= bound:
                STAGE == MODE2
                tau = 0.0
                x = x
                y = y * 0.2
                z = z
                omega1 = 1.5
                omega2 = 1
        elif STAGE == MODE2:
            x = x + delta_t * (- ax * sin(omega * tau))
            y = y + delta_t * (- ay * sin( (omega1 + 1.0) * tau) * sin(omega2)*2)
            z = z + delta_t * (- az * sin( (2.0 - omega2) * tau) * sin(omega1)*2)
            omega1 = omega1 + delta_t * (0 - omega1)
            omega2 = omega2 + delta_t * (0 - omega2)
            tau = tau + delta_t*(1.0)
            if tau >= 8.0:
                STAGE == MODE3
                x = 0.2 * x
                y = 0.5 * y
                z = z
                tau = 0.0
                omega1 = sin(omega1)
                omega2 = 0 - omega2
        elif STAGE == MODE3:
            x = x + delta_t * (- ax * sin(omega * tau))
            y = y + delta_t * (- ay * sin( (omega1 + 1.0) * tau) * sin(omega2)*2)
            z = z + delta_t * (- az * sin( (omega2 + 2.0) * tau) * cos(omega1)*2)
            omega1 = omega1 + delta_t * (0 - 0.5 * omega1)
            omega2 = omega2 + delta_t * (0 - omega2)
            tau = tau + delta_t*(1.0)
            if tau >= 5:
                STAGE == MODE1
                tau = 0.0
                omega1 = 1
                omega2 = 1
    return x
            




def electronic_oscillator_loop(x, y, z, omega1, omega2, tau):
# safe constraint of y: (-6.5992, 5.0)
    bound = ??(1.5, 2.5)
    while STAGE == MODE3:
        if STAGE == MODE1:
            x = x + delta_t*(- ax * sin(omega * tau))
            y = y + delta_t*(- ay * sin( (omega1 + 1.0) * tau) * sin(omega2)*2)
            z = z + delta_t*(- az * sin( (omega2 + 1.0) * tau) * cos(omega1)*2)
            omega1 = omega1 + delta_t*(0 - 0.5 * omega1)
            omega2 = omega2 + delta_t*(0 - omega2)
            tau = tau + delta_t*(1.0)
            if tau >= bound:
                STAGE == MODE2
                tau = 0.0
                x = x
                y = y * 0.2
                z = z
                omega1 = 1.5
                omega2 = 1
        elif STAGE == MODE2:
            x = x + delta_t * (- ax * sin(omega * tau))
            y = y + delta_t * (- ay * sin( (omega1 + 1.0) * tau) * sin(omega2)*2)
            z = z + delta_t * (- az * sin( (2.0 - omega2) * tau) * sin(omega1)*2)
            omega1 = omega1 + delta_t * (0 - omega1)
            omega2 = omega2 + delta_t * (0 - omega2)
            tau = tau + delta_t*(1.0)
            if tau >= 8.0:
                STAGE == MODE3
                x = 0.2 * x
                y = 0.5 * y
                z = z
                tau = 0.0
                omega1 = sin(omega1)
                omega2 = 0 - omega2
    
    return x


def path_explosion(h):
# safe constraint of h: (4.0, 26.48)
    bound  = ??(4.0, 6.0)
    i = 0
    while(h < 10.0):
        h = h + 0.01
        if (h <= bound):
            if (h <= bound - 0.001):
                h = 2 * h
            else:
                h = 3 * h
        count += 1
    if (h <= 3*bound - 0.001):
        h = h
    else:
        h = 10 * h
    
    return count

















        




        

