'''
Integrate the methods for searching
'''
import torch
import time
import random
from torch.autograd import Variable
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import copy

from helper import *
from data_generator import *
from constants import *


def distance_f_point(pred_y, y):
    return torch.abs(pred_y.sub(y))

if MODE in [2,3,4,5]:
    #! Change to smooth max
    def distance_f_interval(X_list, target):
        # alpha_smooth_max_var = var(alpha_smooth_max)
        res = var(0.0)
        # print('X_list', len(X_list))
        #! Smooth Max
        res_up = var(0.0)
        res_base = var(0.0)
        if len(X_list) == 0:
            res = var(1.0)
            return res
        for X_table in X_list:
            X_min = X_table['x_min'].getInterval()
            X_max = X_table['x_max'].getInterval()
            pi = X_table['probability']
            p = X_table['explore_probability']
            # print('pi, p', pi.data.item(), p.data.item())

            X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
            X.left = torch.min(X_min.left, X_max.left)
            X.right = torch.max(X_min.right, X_max.right)
            # print(X.min.left, X.min.right, X.max.left, X.max.right)

            reward = var(0.0)
            intersection_interval = get_intersection(X, target)
            # print('intersection:', intersection_interval.left, intersection_interval.right)
            if intersection_interval.isEmpty():
                # print('isempty')
                reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
            else:
                # print('not empty')
                reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
            # print(X.left, X.right, tmp_res)
            # ! Smooth Max following two lines
            # res_up = res_up.add(tmp_res.mul(torch.exp(tmp_res.mul(alpha_smooth_max_var))))
            # res_base = res_base.add(torch.exp(tmp_res.mul(alpha_smooth_max_var)))
            tmp_res = reward.mul(pi)# pi.div(p))
            # tmp_res is the reward
            # 

            res = res.add(tmp_res)
        res = res.div(var(len(X_list)).add(EPSILON))
        # res = res_up.div(res_basse)

        # TODO: for REINFORCE part
        # TODO: return list(reward), list(pi.div(p))
        
        return res
    
    def distance_f_interval_REINFORCE(X_list, target, Theta):
        alpha_smooth_max_var = var(alpha_smooth_max)
        res = var(0.0)
        # print('X_list', len(X_list))
        reward_list = list()
        log_p_list = list()
        p_list = list()
        #! Smooth Max
        res_up = var(0.0)
        res_base = var(0.0)
        if len(X_list) == 0:
            res = var(1.0)
            return res
        for X_table in X_list:
            X_min = X_table['x_min'].getInterval()
            X_max = X_table['x_max'].getInterval()
            pi = X_table['probability']
            p = X_table['explore_probability']
            # print('pi, p', pi.data.item(), p.data.item())

            X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
            X.left = torch.min(X_min.left, X_max.left)
            X.right = torch.max(X_min.right, X_max.right)
            # print(X.min.left, X.min.right, X.max.left, X.max.right)
            # try:
            #     a = torch.autograd.grad(X.left, Theta, retain_graph=True, allow_unused=True)
            #     print(f"DEBUG: X.left gradient: {a}")
            # except RuntimeError:
            #     print(f"DEBUG: X.left No gradient")
            # try:
            #     a = torch.autograd.grad(X.right, Theta, retain_graph=True, allow_unused=True)
            #     print(f"DEBUG: X.right gradient: {a}")
            # except RuntimeError:
            #     print(f"DEBUG: X.right No gradient")

            reward = var(0.0)
            intersection_interval = get_intersection(X, target)
            # print('intersection:', intersection_interval.left, intersection_interval.right)
            if intersection_interval.isEmpty():
                # print('isempty')
                reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
            else:
                # print('not empty')
                reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
            # print(X.left, X.right, tmp_res)
            # ! Smooth Max following two lines
            # res_up = res_up.add(tmp_res.mul(torch.exp(tmp_res.mul(alpha_smooth_max_var))))
            # res_base = res_base.add(torch.exp(tmp_res.mul(alpha_smooth_max_var)))
            tmp_res = reward.mul(pi.div(p))
            # tmp_res is the reward
            tmp_p = torch.log(pi)

            ## Debug
            # try:
            #     a = torch.autograd.grad(reward, Theta, retain_graph=True, allow_unused=True)
            #     print(f"DEBUG: reward gradient: {a}")
            # except RuntimeError:
            #     print(f"DEBUG: reward No gradient")
            
            # try: 
            #     loss = reward
            #     loss.backward()
            #     a = Theta.grad
            #     print(f"DEBUG: SANITY CHECK reward loss, gradient: {a}")
            # except RuntimeError:
            #     print(f"DEBUG: SANITY CHECK loss error, no grad")
            ## Debug

            log_p_list.append(tmp_p)
            reward_list.append(reward)

            # try:
            #     a = torch.autograd.grad(reward_list[-1], Theta, retain_graph=True, allow_unused=True)
            #     print(f"DEBUG: SANITY CHECK reward gradient: {a}")
            # except RuntimeError:
            #     print(f"DEBUG: reward No gradient")

            p_list.append(p)

            res = res.add(tmp_res)
        res = res.div(var(len(X_list)).add(EPSILON))
        # res = res_up.div(res_basse)
        
        return res, p_list, log_p_list, reward_list
    

    def extract_result_safty(symbol_table_list):
        res_l, res_r = P_INFINITY, N_INFINITY
        for symbol_table in symbol_table_list:
            # X = symbol_table['x']
            res_l = torch.min(res_l, symbol_table['x_min'].getInterval().left)
            res_r = torch.max(res_r, symbol_table['x_max'].getInterval().right)
        
        return res_l.data.item(), res_r.data.item()


def normal_pdf(x, mean, std):
    # print(x, mean, std)
    y = torch.exp((-((x-mean)**2)/(2*std*std)))/ (std* torch.sqrt(2*var(math.pi)))
    # print(y)
    # exit(0)
    return y


def generate_theta_sample_set(Theta):
    sample_theta_list = list()
    sample_theta_probability_list = list()
    for i in range(THETA_SAMPLE_SIZE):
        sample_theta = torch.normal(mean=Theta, std=var(1.0))
        sample_theta_probability = normal_pdf(sample_theta, Theta, var(1.0))
        # theta_normal.log_prob(sample_theta)
        # print(sample_theta, sample_theta_probability)
        sample_theta_list.append(sample_theta)
        sample_theta_probability_list.append(sample_theta_probability)
    return sample_theta_list, sample_theta_probability_list


def update_symbol_table_with_sample_theta(sample_theta_list, sample_theta_probability_list, symbol_table_list):
    for symbol_table in symbol_table_list:
        symbol_table['sample_theta'] = copy.deepcopy(sample_theta_list)
        symbol_table['sample_theta_probability'] = copy.deepcopy(sample_theta_probability_list)
    return symbol_table_list


def gd_direct_noise(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=0.01, epoch=1000, lr=0.00001, theta=None):
    print("--------------------------------------------------------------")
    print('---- Gradient Direct Noise Descent---- ')
    print('====Start Training====')
    len_theta = len(theta_l)
    TIME_OUT = False

    x_min = var(10000.0)
    x_max = var(0.0)
    x_smooth_min = var(10000.0)
    x_smooth_max = var(0.0)

    loop_list = list()
    loss_list = list()

    # if theta is None:
    # Theta = list()
    
    # initialize Theta
    # for idx, value in enumerate(theta_l):
    #     Theta.append(var(random.uniform(theta_l[idx], theta_r[idx]), requires_grad=True))
    tmp_theta_list = [random.uniform(theta_l[idx], theta_r[idx]) for idx, value in enumerate(theta_l)]

    # !debug
    # tmp_theta_list = [60.60904056908072, 0.8263085017772094, 1.224599693577951e-05, 0.08500484058054017, 0.0008327441661250281, -0.0009589305108516826, -0.0001350546846055669, 4.023913408164206, -1.4078577974523796, 55.12680757950901]
    Theta = var_list(tmp_theta_list, requires_grad=True)
    # theta_normal = torch.distributions.normal.Normal(Theta, var(0.01))
    # sample_theta_list = generate_theta_sample_set(Theta)

    # h = Theta.register_hook(lambda grad: grad + random.uniform(-noise, noise))
    # Theta[0] = var(62.0, requires_grad=True) # 59.4
    # Theta[1] = var(0.9, requires_grad=True)
    # Theta[2], Theta[3], Theta[4], Theta[5], Theta[6], Theta[7], Theta[8] = var(0.0, requires_grad=True), var(0.1, requires_grad=True), var(0.0, requires_grad=True), var(0.0, requires_grad=True), var(0.0, requires_grad=True), var(1.0, requires_grad=True), var(1.0, requires_grad=True)
    
    root = construct_syntax_tree(Theta)
    root_smooth_point = construct_syntax_tree_smooth_point(Theta)
    root_point = construct_syntax_tree_point(Theta)
    # Theta = var(69.9)

    start_time = time.time()

    for i in range(epoch):
        symbol_table_list = initialization(x_l, x_r, X_train, y_train)
        # symbol_table_list = update_symbol_table_with_sample_theta(sample_theta_list, sample_theta_probability_list, symbol_table_list)
        
        f = var(0.0)
        y_l = P_INFINITY
        y_r = N_INFINITY

        print('-- Theta:', [i.data.item() for i in Theta])
        for idx, x in enumerate(X_train):
            x, y = x, y_train[idx]
            symbol_table_smooth_point = initialization_point(x)
            # symbol_table_smooth_point = update_symbol_table_with_sample_theta(sample_theta_list, symbol_table_smooth_point)

            # print('run smooth')
            symbol_table_smooth_point = root_smooth_point['entry'].execute(symbol_table_smooth_point)
            # print('run point')

            ###  DEBUG
            # symbol_table_point = initialization_point(x)
            # symbol_table_point = root_point['entry'].execute(symbol_table_point)

            # x_min =  torch.min(symbol_table_point['x_min'], x_min)
            # x_max  = torch.max(symbol_table_point['x_max'], x_max)

            # x_smooth_min = torch.min(symbol_table_smooth_point['x_min'], x_smooth_min)
            # x_smooth_max = torch.max(symbol_table_smooth_point['x_max'], x_smooth_max)

            # y_l = torch.min(symbol_table_smooth_point['res'], y_l)
            # y_r = torch.max(symbol_table_smooth_point['res'], y_r)
            ###  DEBUG

            # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
            f = f.add(distance_f_point(symbol_table_smooth_point['res'], var(y)))

        #TODO: a function return, penalty_f, f
        f = f.div(var(len(X_train)))
        print('quantitive f', f.data.item())
        symbol_table_list = root['entry'].execute(symbol_table_list)
        print('length: ', len(symbol_table_list))

        # print('quantitive f', f.data.item())

        res_l, res_r = extract_result_safty(symbol_table_list)
        #! Change the Penalty
        penalty_f, p_list, log_p_list, reward_list = distance_f_interval_REINFORCE(symbol_table_list, target, Theta)

        print('safe f', penalty_f.data.item(), res_l, res_r) # , x_smooth_min.data.item(), x_smooth_max.data.item())

        ### DEBUG
        # print('safe f, ', penalty_f.data.item(), res_l, res_r, x_min.data.item(), x_max.data.item(), x_smooth_min.data.item(), x_smooth_max.data.item(), y_l.data.item(), y_r.data.item(), ) # , )
        ### DEBUG

        # ! First way to implement
        # try:
        #     dQ = torch.autograd.grad(f, Theta, retain_graph=True, allow_unused=True)
        #     print(f"DEBUG, gradient: {dQ}")
        # except RuntimeError:
        #     print(f"DEBUG,----ERROR , gradient: None")

        # dTheta = dQ[0]
        # grad_start_time = time.time()
        # for idx, value in enumerate(p_list):
        #     p = value
        #     log_p = log_p_list[idx]
        #     reward = reward_list[idx]
        #     ## DEBUG
        #     try:
        #         dReward = torch.autograd.grad(reward, Theta, retain_graph=True, allow_unused=True)
        #         # print(f"DEBUG, gradient: {dReward}")
        #     except RuntimeError:
        #         print(f"DEBUG,----ERROR , reward gradient: None")
        #     try:
        #         dLog_p = torch.autograd.grad(log_p, Theta, retain_graph=True, allow_unused=True)
        #         # print(f"DEBUG, gradient: {dLog_p}")
        #     except RuntimeError:
        #         print(f"DEBUG,----ERROR , log_p gradient: None")
            
        #     ## DEBUG
        #     # print(type(p.data.item()), type(reward.data.item()))
        #     # if p.data.item() != 0.01:
        #     #     print(f"DEBUG: p: {p.data.item()}")
        #     # print(var(reward.data.item()))
        #     dPath = lambda_.mul(p).mul(dReward[0].add(reward.mul(dLog_p[0])))
        #     dTheta = dTheta.add(dPath)
        

        # ! another way of implementation
        loss = f
        loss.backward(retain_graph=True)
        
        grad_start_time = time.time()
        for idx, value in enumerate(p_list):
            p = value
            log_p = log_p_list[idx]
            reward = reward_list[idx]

            loss = lambda_.mul(var(p.data.item()).mul(reward.add(var(reward.data.item()).mul(log_p))))
            try:
                loss.backward(retain_graph=True)
            except RuntimeError:
                print(f"DEBUG: p: {idx}, No grad")
            
            # print(f"DEBUG: theta.grad, {Theta.grad}")

        print("-- Calculate Path Gradient %s seconds ---" % (time.time() - grad_start_time))

        res = f.add(lambda_.mul(penalty_f))
        print(i, '--', [i.data.item() for i in Theta], res.data.item())
        # if i == 0:
        #     continue
        # if i == 1:
        #     exit(0)
        # exit()

        derivation = var(0.0)

        if torch.abs(res.data) < var(stop_val):
            break

        # ! first way
        # print(f"dTheta: {[toy_dtheta.data.item() for toy_dtheta in dTheta]}, \n Theta: {[toy_theta.data.item() for toy_theta in Theta]}")
        # ! second way
        print(f"dTheta: {[toy_dtheta for toy_dtheta in Theta.grad]}, \nTheta: {[toy_theta.data.item() for toy_theta in Theta]}")

        with torch.no_grad():
            for theta_idx in range(len_theta):
                try:
                    # Theta[theta_idx].data -= lr * (dTheta[theta_idx].data + var(random.uniform(-noise, noise)))
                    Theta[theta_idx].data -= lr * (Theta.grad[theta_idx] + var(random.uniform(-noise, noise)))
                except RuntimeError: # for the case no gradient with Theta[theta_idx]
                    Theta[theta_idx].data -= lr * (var(random.uniform(-noise, noise)))
                    if torch.abs(res.data) < var(stop_val):
                        break
            
            # ! second way
            Theta.grad.zero_()
        
        for theta_idx in range(len_theta):
            if Theta[theta_idx].data.item() <= theta_l[theta_idx] or Theta[theta_idx].data.item() >= theta_r[theta_idx]:
                Theta[theta_idx].data.fill_(random.uniform(theta_l[theta_idx], theta_r[theta_idx]))
                    

        loop_list.append(i)
        loss_list.append(res.data)
        if (time.time() - start_time)/(i+1) > 300:
            log_file = open(file_dir, 'a')
            log_file.write('TIMEOUT: avg epoch time > 300s \n')
            log_file.close()
            TIME_OUT = True
            break
        # print("-- One Epoch %s seconds ---" % (time.time() - start_time))
    # plt.plot(loop_list, loss_list, label = "beta")
    # plt.xlabel('expr count')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()
    # print('GOT! Loss, theta', f.data, Theta.data)
    log_file = open(file_dir, 'a')
    spend_time = time.time() - start_time
    log_file.write('Optimization:' + str(spend_time) + ',' + str(i+1) + ',' + str(spend_time/(i+1)) + '\n')
    log_file.close()
    
    print("--- %s seconds ---" % (spend_time))
    print("--------------------------------------------------------------")

    theta = Theta# .data.item()
    loss = res# .data.item()
    print('Theta[0]: {0:.3f}, Loss: {1:.3f}'.format(theta[0].data.item(), loss.data.item()))
    # exit(0)

    return theta, loss, loss_list, f, penalty_f, TIME_OUT


def cal_c(X_train, y_train, theta):
    # only for calculating the value instead of the gradient
    root = construct_syntax_tree(theta)
    symbol_table_list = initialization(x_l, x_r, X_train, y_train)

    symbol_table_list = root['entry'].execute(symbol_table_list)
    c = distance_f_interval(symbol_table_list, target)
    print('cal_c', theta, c)

    return c


def cal_q(X_train, y_train, theta):
    root_smooth_point = construct_syntax_tree_smooth_point(theta)
    q = var(0.0)

    for idx, x in enumerate(X_train):
        x, y = x, y_train[idx]
        symbol_table_smooth_point = initialization_point(x)
        symbol_table_smooth_point = root_smooth_point['entry'].execute(symbol_table_smooth_point)

        # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
        q = q.add(distance_f_point(symbol_table_smooth_point['res'], var(y)))

    q = q.div(var(len(X_train)))
    print('cal_q', theta, q)
    
    return q


# # DIRECT
# def direct(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=0.01, epoch=1000, lr=0.00001, theta=None):
#     print("--------------------------------------------------------------")
#     print('----DIRECT----')
#     print('====Start Training====')
#     start_time = time.time()

#     loss_list = list()
#     res_f = var(0.0)
#     res_penalty = var(0.0)

#     def myfunc(theta, grad):
#         Theta = var(theta)
#         # Theta = var(70.0)
#         root = construct_syntax_tree(Theta)
#         symbol_table_list = initialization(x_l, x_r)
#         # root_point = construct_syntax_tree_point(Theta)
#         root_smooth_point = construct_syntax_tree_smooth_point(Theta)

#         f = var(0.0)

#         for idx, x in enumerate(X_train):
#             x, y = x, y_train[idx]
#             symbol_table_point = initialization_point(x)
#             symbol_table_point = root_smooth_point['entry'].execute(symbol_table_point)

#             # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
#             f = f.add(distance_f_point(symbol_table_point['res'], var(y)))
#         f = f.div(var(len(X_train)))
#         print('quantitive f', f.data.item())
#         # exit(0)

#         # symbol_table_list = root['entry'].execute(symbol_table_list)
#         # # show_symbol_tabel_list(symbol_table_list)
#         # # print(len(symbol_table_list))
#         # penalty_f = distance_f_interval(symbol_table_list, target)
#         # res_l, res_r = extract_result_safty(symbol_table_list)
#         # print('safe f', penalty_f.data.item(), res_l, res_r)

#         res = f.add(var(lambda_).mul(penalty_f))
#         print(Theta.data.item(), f.data.item())
#         f_value = res.data.item()

#         loss_list.append(f_value)

#         # if abs(f_value) < EPSILON.data.item():
#         #     raise ValueError(str(theta[0]) + ',' + str(f_value))

#         return f_value

#     x = np.array([random.uniform(theta_l, theta_r)])
#     opt = nlopt.opt(nlopt.GN_DIRECT, 1)
#     opt.set_lower_bounds([theta_l])
#     opt.set_upper_bounds([theta_r])
#     opt.set_min_objective(myfunc)
#     opt.set_stopval(stop_val)
#     opt.set_maxeval(epoch)
#     try:
#         x = opt.optimize(x)
#     except ValueError as error:
#         error_list = str(error).split(',')
#         error_value = [float(err) for err in error_list]
#         theta = error_value[0]
#         loss = error_value[1]
#         # print('theta, f', error_value[0], error_value[1])

#     print("--- %s seconds ---" % (time.time() - start_time))
#     print("--------------------------------------------------------------")

#     theta = x[0]
#     loss = opt.last_optimum_value()
#     print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta, loss))
    
#     return theta, loss, loss_list, res_f, res_penalty

# # Gradient + noise
# # noise: 1.random  2.Gaussian Noise 
# def gd_direct_noise_old(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=0.01, epoch=1000, lr=0.00001, theta=None):
#     print("--------------------------------------------------------------")
#     print('---- Gradient Direct Noise Descent---- ')
#     print('====Start Training====')
#     len_theta = len(theta_l)
#     TIME_OUT = False

#     x_min = var(10000.0)
#     x_max = var(0.0)

#     loop_list = list()
#     loss_list = list()

#     # if theta is None:
#     Theta = list()
#     # initialize Theta
#     for idx, value in enumerate(theta_l):
#         Theta.append(var(random.uniform(theta_l[idx], theta_r[idx]), requires_grad=True))
#     # Theta[0] = var(59.4, requires_grad=True)
#     # Theta[1] = var(0.9, requires_grad=True)
#     # Theta[2], Theta[3], Theta[4], Theta[5], Theta[6], Theta[7], Theta[8] = var(0.0, requires_grad=True), var(0.1, requires_grad=True), var(0.0, requires_grad=True), var(0.0, requires_grad=True), var(0.0, requires_grad=True), var(1.0, requires_grad=True), var(1.0, requires_grad=True)
#     # Theta = var(random.uniform(theta_l, theta_r), requires_grad=True)
#     # else:
#     #     Theta = theta
#     # Theta = var(2.933, requires_grad=True)
#     root = construct_syntax_tree(Theta)
#     root_smooth_point = construct_syntax_tree_smooth_point(Theta)
#     root_point = construct_syntax_tree_point(Theta)
#     # Theta = var(69.9)

#     start_time = time.time()

#     for i in range(epoch):
#         # Theta = var(4.272021770477295, requires_grad=True)
#         # if i == 1:
#         #     Theta.data = var(5.303304672241211)

#         symbol_table_list = initialization(x_l, x_r, X_train, y_train)

#         f = var(0.0)
#         y_l = P_INFINITY
#         y_r = N_INFINITY

#         print('Theta:', [i.data.item() for i in Theta])
#         for idx, x in enumerate(X_train):
#             x, y = x, y_train[idx]
#             symbol_table_smooth_point = initialization_point(x)
#             # print('run smooth')
#             symbol_table_smooth_point = root_smooth_point['entry'].execute(symbol_table_smooth_point)
#             # print('run point')

#             symbol_table_point = initialization_point(x)
#             symbol_table_point = root_point['entry'].execute(symbol_table_point)

#             x_min =  torch.min(symbol_table_point['x_min'], x_min)
#             x_max  = torch.max(symbol_table_point['x_max'], x_max)

#             y_l = torch.min(symbol_table_smooth_point['res'], y_l)
#             y_r = torch.max(symbol_table_smooth_point['res'], y_r)

#             # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
#             f = f.add(distance_f_point(symbol_table_smooth_point['res'], var(y)))

#         #TODO: a function return, penalty_f, f
#         f = f.div(var(len(X_train)))
#         print('quantitive f', f.data.item())
#         symbol_table_list = root['entry'].execute(symbol_table_list)
#         print('length: ', len(symbol_table_list))

#         # print('quantitive f', f.data.item())

#         res_l, res_r = extract_result_safty(symbol_table_list)
#         #! Change the Penalty
#         penalty_f, p_list, log_p_list, reward_list = distance_f_interval_REINFORCE(symbol_table_list, target)

#         print('safe f, ', penalty_f.data.item(), res_l, res_r, x_min.data.item(), x_max.data.item(), y_l.data.item(), y_r.data.item(), ) # , )

#         exp1 = [var(0.0)] * len_theta
#         exp2 = [var(0.0)] * len_theta
#         gradient_penalty_f = [var(0.0)] * len_theta
#         # exp1 = var(0.0)
#         # exp2 = var(0.0)
#         for theta_idx in range(len_theta):
#             for idx, value in enumerate(p_list):
#                 p = value
#                 log_p = log_p_list[idx]
#                 reward = reward_list[idx]
#                 try:
#                     gradient_reward = torch.autograd.grad(reward, Theta[theta_idx], retain_graph=True)
#                 except RuntimeError:
#                     gradient_reward = (var(0.0), )
#                 try:
#                     # print('log_p', log_p)
#                     gradient_log_p = torch.autograd.grad(log_p, Theta[theta_idx], retain_graph=True)
#                 except RuntimeError:
#                     gradient_log_p = (var(0.0), )
#                 # print(f"DEBUG: \n gradient_reward:{gradient_reward}\n gradient_log_p:{gradient_log_p}")
#                 # if theta_idx == 0:
#                 #     print('DEBUG: p: {0:.6f}, log_p: {1:.6f}, reward: {2:.6f}, gradient_reward: {3:.6f}, gradient_log_p: {4:.6f}'.format(p.data.item(), log_p.data.item(), reward.data.item(), gradient_reward[0].data.item(), gradient_log_p[0].data.item()))
#                 exp1[theta_idx] = exp1[theta_idx].add(p.mul(gradient_reward[0]))
#                 exp2[theta_idx] = exp2[theta_idx].add(p.mul(reward).mul(gradient_log_p[0]))
#                 # if theta_idx== 0:
#                 #     print('DEBUG: p.mul(gradient_reward):{0:.12f}, p.mul(reward).mul(gradient_log_p[0]): {0:.12f}, exp1: {0:.12f}, exp2: {0:.12f}'.format((p.mul(gradient_reward[0])).data.item(), (p.mul(reward).mul(gradient_log_p[0])).data.item(), exp1[theta_idx].data.item(), exp2[theta_idx].data.item()))
#             gradient_penalty_f[theta_idx] = exp1[theta_idx].add(exp2[theta_idx])

#         res = f.add(lambda_.mul(penalty_f))
#         print(i, '--', [i.data.item() for i in Theta], res.data.item())
#         # if i == 0:
#         #     continue
#         # if i == 1:
#         #     exit(0)
#         # exit()
#         derivation = var(0.0)
        
#         for theta_idx in range(len_theta):
#             try:
#                 #! update the gradient
#                 dTheta = torch.autograd.grad(f, Theta[theta_idx], retain_graph=True)
#                 derivation = dTheta[0].add(lambda_.mul(gradient_penalty_f[theta_idx]))
#                 # derivation = dTheta[0]
#                 # print('f, theta, dTheta:', f.data, Theta.data, derivation)
#                 print(f"DEBUG: {theta_idx}, {dTheta[0].data.item()}")

#                 if torch.abs(res.data) < var(stop_val): # epsilon:
#                     # print(f.data, Theta.data)
#                     break
#                 # if torch.abs(derivation.data) < EPSILON:
#                 #     Theta.data.fill_(random.uniform(theta_l, theta_r))
#                 #     continue

#                 Theta[theta_idx].data -= lr_l[theta_idx] * (derivation.data + var(random.uniform(-noise, noise)))
#                 print('deriavation, theta ', derivation.data.item(), Theta[theta_idx].data.item())
        
#             except RuntimeError:
#                 # print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
#                 if torch.abs(res.data) < var(stop_val):
#                     # print(f.data, Theta.data)
#                     print('FIND IT!')
#                     # exit(0)
#                     break

#                 Theta[theta_idx].data -= lr_l[theta_idx] * (derivation.data + var(random.uniform(-1.0, 1.0)))
        
#         if torch.abs(res.data) < var(stop_val):
#             # print(f.data, Theta.data)
#             print('FIND IT!')
#             # exit(0)
#             break

#         for theta_idx in range(len_theta):
#             if Theta[theta_idx].data.item() <= theta_l[theta_idx] or Theta[theta_idx].data.item() >= theta_r[theta_idx]:
#                 Theta[theta_idx].data.fill_(random.uniform(theta_l[theta_idx], theta_r[theta_idx]))

#         loop_list.append(i)
#         loss_list.append(res.data)
#         if (time.time() - start_time)/(i+1) > 300:
#             log_file = open(file_dir, 'a')
#             log_file.write('TIMEOUT: avg epoch time > 250s \n')
#             log_file.close()
#             TIME_OUT = True
#             break
#         # print("-- One Epoch %s seconds ---" % (time.time() - start_time))
#     # plt.plot(loop_list, loss_list, label = "beta")
#     # plt.xlabel('expr count')
#     # plt.ylabel('loss')
#     # plt.legend()
#     # plt.show()
#     # print('GOT! Loss, theta', f.data, Theta.data)
#     log_file = open(file_dir, 'a')
#     spend_time = time.time() - start_time
#     log_file.write('Optimization:' + str(spend_time) + ',' + str(i+1) + ',' + str(spend_time/(i+1)) + '\n')
#     log_file.close()
    
#     print("--- %s seconds ---" % (spend_time))
#     print("--------------------------------------------------------------")

#     theta = Theta# .data.item()
#     loss = res# .data.item()
#     print('Theta[0]: {0:.3f}, Loss: {1:.3f}'.format(theta[0].data.item(), loss.data.item()))
#     # exit(0)

#     return theta, loss, loss_list, f, penalty_f, TIME_OUT

'''
def gd_gaussian_noise(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=0.01, epoch=1000, lr=0.00001, theta=None):

    def generate_gaussian_noise(step):
        step = 0
        sigma = (eta/(1+step)**gamma)**(1/2.0)
        return random.gauss(0, sigma)

    print("--------------------------------------------------------------")
    print('---- Gradient Descent + Gaussian Noise---- ')
    print('====Start Training====')
    start_time = time.time()

    step = 0
    loop_list = list()
    loss_list = list()

    Theta = var(random.uniform(theta_l, theta_r), requires_grad=True)
    root = construct_syntax_tree(Theta)
    root_smooth_point = construct_syntax_tree_smooth_point(Theta)

    for i in range(epoch):

        symbol_table_list = initialization(x_l, x_r)

        f = var(0.0)

        for idx, x in enumerate(X_train):
            x, y = x, y_train[idx]
            symbol_table_point = initialization_point(x)
            symbol_table_point = root_smooth_point['entry'].execute(symbol_table_point)

            # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
            f = f.add(distance_f_point(symbol_table_point['res'], var(y)))
        
        # f = f.div(var(len(X_train)))
        # # print('quantitive f', f.data.item())

        # symbol_table_list = root['entry'].execute(symbol_table_list)
        # penalty_f = distance_f_interval(symbol_table_list, target)

        # # print('safe f', penalty_f.data.item())

        # f = f.add(var(lambda_).mul(penalty_f))
        # # print(Theta.data.item(), f.data.item())
        f = f.div(var(len(X_train)))
        print('quantitive f', f.data.item())

        symbol_table_list = root['entry'].execute(symbol_table_list)
        print('length: ', len(symbol_table_list))
        res_l, res_r = extract_result_safty(symbol_table_list)
        penalty_f = distance_f_interval(symbol_table_list, target)
        print('safe f', penalty_f.data.item(), res_l, res_r) # , y_l.data.item(), y_r.data.item())

        res = f.add(lambda_.mul(penalty_f))
        print(i, '--', Theta.data.item(), res.data.item())

        try:
            dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
            derivation = dTheta[0]
            # print('f, theta, dTheta:', f.data, Theta.data, derivation)

            if torch.abs(f.data) < var(stop_val): # epsilon:
                # print(f.data, Theta.data)
                break
            
            if torch.abs(derivation.data) < EPSILON:
                Theta.data.fill_(random.uniform(theta_l, theta_r))
                continue
        
            Theta.data -= lr * (derivation.data + var(generate_gaussian_noise(step)))
            # print('noise', generate_gaussian_noise(step))
            step += 1
        
        except RuntimeError:
            # print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
            if torch.abs(f.data) < var(stop_val):
                # print(f.data, Theta.data)
                break

            Theta.data -= lr * (derivation.data + var(generate_gaussian_noise(step)))
            # print('noise', generate_gaussian_noise(step))
            step += 1
        
        if Theta.data.item() <= theta_l or Theta.data.item() >= theta_r:
            Theta.data.fill_(random.uniform(theta_l, theta_r))
            continue

        loop_list.append(i)
        loss_list.append(f.data)
    
    # plt.plot(loop_list, loss_list, label = "beta")
    # plt.xlabel('expr count')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()
    # print('GOT! Loss, theta', f.data, Theta.data)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")

    theta = Theta# .data.item()
    loss = f# .data.item()
    print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta, loss))

    return theta, loss, loss_list, f, penalty_f


# GD
def gd(X_train, y_train, theta_l, theta_r, target, stop_val, epoch, lr):
    print("--------------------------------------------------------------")
    print('---- Gradient Descent---- ')
    print('====Start Training====')
    start_time = time.time()

    loop_list = list()
    loss_list = list()

    Theta = var(random.uniform(theta_l, theta_r), requires_grad=True)
    root = construct_syntax_tree(Theta)
    root_smooth_point = construct_syntax_tree_smooth_point(Theta)

    for i in range(epoch):

        symbol_table_list = initialization(x_l, x_r)

        f = var(0.0)

        for idx, x in enumerate(X_train):
            x, y = x, y_train[idx]
            symbol_table_point = initialization_point(x)
            symbol_table_point = root_smooth_point['entry'].execute(symbol_table_point)

            # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
            f = f.add(distance_f_point(symbol_table_point['res'], var(y)))
        f = f.div(var(len(X_train)))
        print('quantitive f', f.data.item())

        symbol_table_list = root['entry'].execute(symbol_table_list)
        penalty_f = distance_f_interval(symbol_table_list, target)

        print('safe f', penalty_f.data.item())

        f = f.add(var(lambda_).mul(penalty_f))
        print(i, '--', Theta.data.item(), f.data.item())

        try:
            dTheta = torch.autograd.grad(f, Theta, retain_graph=True)
            derivation = dTheta[0]
            print('f, theta, dTheta:', f.data, Theta.data, derivation)

            if torch.abs(f.data) < var(stop_val): # epsilon:
                # print(f.data, Theta.data)
                break
            if torch.abs(derivation.data) < EPSILON:
                Theta.data.fill_(random.uniform(theta_l, theta_r))
                continue
        
            Theta.data -= lr * derivation.data
        
        except RuntimeError:
            # print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
            if torch.abs(f.data) < var(stop_val):
                # print(f.data, Theta.data)
                break

            Theta.data -= lr * derivation.data
        
        if Theta.data.item() <= theta_l or Theta.data.item() >= theta_r:
            Theta.data.fill_(random.uniform(theta_l, theta_r))
            continue

        loop_list.append(i)
        loss_list.append(f.data)
    
    # plt.plot(loop_list, loss_list, label = "beta")
    # plt.xlabel('expr count')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()
    # print('GOT! Loss, theta', f.data, Theta.data)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")

    theta = Theta# .data.item()
    loss = f# .data.item()
    print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta, loss))

    return theta, loss, loss_list, f, penalty_f
'''