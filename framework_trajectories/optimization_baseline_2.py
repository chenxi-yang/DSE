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

from helper import *
from data_generator import *
from constants import *


def distance_f_point(pred_y, y):
    return torch.abs(pred_y.sub(y))


if MODE in [2,3,4,5]:
    def distance_f_interval(X_list, target):
        res = var(0.0)
        # print('X_list', len(X_list))
        if len(X_list) == 0:
            res = var(1.0)
        for X_table in X_list:
            X_min = X_table['x_min'].getInterval()
            X_max = X_table['x_max'].getInterval()
            pi = X_table['probability']
            p = X_table['explore_probability']
            # print('pi, p', pi.data.item(), p.data.item())

            X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
            X.left = X_min.left # torch.min(X_min.left, X_max.left)
            X.right = X_max.right # torch.max(X_min.right, X_max.right)
            # print(X.min.left, X.min.right, X.max.left, X.max.right)

            tmp_res = var(0.0)
            intersection_interval = get_intersection(X, target)
            # print('intersection:', intersection_interval.left, intersection_interval.right)
            if intersection_interval.isEmpty():
                # print('isempty')
                tmp_res = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
            else:
                # print('not empty')
                tmp_res = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
            # print(X.left, X.right, tmp_res)
            tmp_res = tmp_res.mul(pi.div(p))

            res = res.add(tmp_res)
        res = res.div(var(len(X_list)).add(EPSILON))
        
        return res
    

    def extract_result_safty(symbol_table_list):
        res_l, res_r = P_INFINITY, N_INFINITY
        for symbol_table in symbol_table_list:
            # X = symbol_table['x']
            res_l = torch.min(res_l, symbol_table['x_min'].getInterval().left)
            res_r = torch.max(res_r, symbol_table['x_max'].getInterval().right)
        
        return res_l.data.item(), res_r.data.item()


if MODE == 1:
    def distance_f_interval(symbol_table, target):
        X = symbol_table['x']
        # print('interval', X.left, X.right)

        intersection_interval = get_intersection(X, target)
        if intersection_interval.isEmpty():
            res = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            res = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        
        return res  
    

    def extract_result_safty(symbol_table_list):
        res_l, res_r = P_INFINITY, N_INFINITY
        X = symbol_table_list['x']
        res_l = torch.min(res_l, X.left)
        res_r = torch.max(res_r, X.right)
        
        return res_l.data.item(), res_r.data.item()


def plot_sep_quan_safe_trend(X_train, y_train, theta_l, theta_r, target, k=100):
    print('in plot_sep_quan_safe_trend')
    # k = 
    unit = (theta_r - theta_l) * 1.0 / k
    
    theta_list = list()
    quan_f_list = list()
    result_safety_l_list = list()
    result_safety_r_list = list()
    target_l_list = list()
    target_r_list = list()   
    y_l_list = list()
    y_r_list = list()

    for i in range(k):
        theta = theta_l + i * unit
        # theta = 2.914
        Theta = var(theta)
        # print('optimization', theta, Theta)
        root = construct_syntax_tree(Theta)
        symbol_table_list = initialization(x_l, x_r)
        root_point = construct_syntax_tree_point(Theta)
        root_smooth_point = construct_syntax_tree_smooth_point(Theta)

        f = var(0.0)
        y_l = P_INFINITY
        y_r = N_INFINITY
        # print(Theta.data.item())
        
        for idx, x in enumerate(X_train):
            # print('In Loop')
            x, y = x, y_train[idx]
            # x = [11.747363060439167]

            # print(x, y)
            symbol_table_point = initialization_point(x)
            symbol_table_point = root_point['entry'].execute(symbol_table_point)
            # print('finish point')
            symbol_table_smooth_point = initialization_point(x)
            symbol_table_smooth_point = root_smooth_point['entry'].execute(symbol_table_smooth_point)
            # print(x, y)
            # print('finish smooth', symbol_table_smooth_point['res'])
            # exit(0)

            f = f.add(distance_f_point(symbol_table_smooth_point['res'], var(y)))
            y_l = torch.min(symbol_table_point['x_min'], y_l)
            y_r = torch.max(symbol_table_point['x_max'], y_r)
            # exit(0)
        print('====finish smooth point computing')
        f = f.div(var(len(X_train)))
        print('point dist', f.data.item())
        # exit(0)
        
        symbol_table_list = root['entry'].execute(symbol_table_list)
        print('====final intervals====', len(symbol_table_list))
        # show_symbol_tabel_list(symbol_table_list)
        approximate_result_safety_l, approximate_result_safety_r = extract_result_safty(symbol_table_list)
        penalty_f = distance_f_interval(symbol_table_list, target)

        print(theta, f.data.item(), penalty_f.data.item(), approximate_result_safety_l, approximate_result_safety_r, y_l.data.item(), y_r.data.item())
        # exit(0)

        theta_list.append(theta)
        quan_f_list.append(f.data.item())
        result_safety_l_list.append(approximate_result_safety_l)
        result_safety_r_list.append(approximate_result_safety_r)
        target_l_list.append(target.left.data.item())
        target_r_list.append(target.right.data.item())
        y_l_list.append(y_l.data.item())
        y_r_list.append(y_r.data.item())

    plt.plot(theta_list, quan_f_list, color='blue', label='quan_f')
    plt.plot(theta_list, target_l_list, color='green', label='target_l')
    plt.plot(theta_list, target_r_list, color='green', label='target_r')
    plt.plot(theta_list, result_safety_l_list, color='red', label='pred_y_l')
    plt.plot(theta_list, result_safety_r_list, color='red', label='pred_y_r')
    plt.plot(theta_list, y_l_list, color='orange', label='real_sample_y_l')
    plt.plot(theta_list, y_r_list, color='orange', label='real_sample_y_r')
    plt.ylabel('Property')
    plt.xlabel('Theta')
    plt.title(CURRENT_PROGRAM + '-' + MODE_NAME)# mode_list[MODE])
    plt.legend()
    plt.savefig('figures/' + CURRENT_PROGRAM + '-' + MODE_NAME + '-' + 'program_stat.png')
    # plt.savefig('figures/' + CURRENT_PROGRAM + '-' + mode_list[MODE] + '-' + 'program_stat.pdf')
    plt.show()


# DIRECT
def direct(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=0.01, epoch=1000, lr=0.00001, theta=None):
    print("--------------------------------------------------------------")
    print('----DIRECT----')
    print('====Start Training====')
    start_time = time.time()

    loss_list = list()
    res_f = var(0.0)
    res_penalty = var(0.0)

    def myfunc(theta, grad):
        Theta = var(theta)
        # Theta = var(70.0)
        # root = construct_syntax_tree(Theta)
        symbol_table_list = initialization(x_l, x_r)
        # root_point = construct_syntax_tree_point(Theta)
        root_smooth_point = construct_syntax_tree_smooth_point(Theta)

        f = var(0.0)

        for idx, x in enumerate(X_train):
            x, y = x, y_train[idx]
            symbol_table_point = initialization_point(x)
            symbol_table_point = root_smooth_point['entry'].execute(symbol_table_point)

            # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
            f = f.add(distance_f_point(symbol_table_point['res'], var(y)))
        f = f.div(var(len(X_train)))
        print('quantitive f', f.data.item())
        # exit(0)

        # symbol_table_list = root['entry'].execute(symbol_table_list)
        # # show_symbol_tabel_list(symbol_table_list)
        # # print(len(symbol_table_list))
        # penalty_f = distance_f_interval(symbol_table_list, target)
        # res_l, res_r = extract_result_safty(symbol_table_list)
        # print('safe f', penalty_f.data.item(), res_l, res_r)

        # res = f.add(var(lambda_).mul(penalty_f))
        # print(Theta.data.item(), f.data.item())
        f_value = f.data.item()

        loss_list.append(f_value)

        # if abs(f_value) < EPSILON.data.item():
        #     raise ValueError(str(theta[0]) + ',' + str(f_value))

        return f_value

    x = np.array([random.uniform(theta_l, theta_r)])
    opt = nlopt.opt(nlopt.GN_DIRECT, 1)
    opt.set_lower_bounds([theta_l])
    opt.set_upper_bounds([theta_r])
    opt.set_min_objective(myfunc)
    opt.set_stopval(stop_val)
    opt.set_maxeval(epoch)
    try:
        x = opt.optimize(x)
    except ValueError as error:
        error_list = str(error).split(',')
        error_value = [float(err) for err in error_list]
        theta = error_value[0]
        loss = error_value[1]
        # print('theta, f', error_value[0], error_value[1])

    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")

    theta = x[0]
    loss = opt.last_optimum_value()
    print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta, loss))
    
    return theta, loss, loss_list, res_f, res_penalty


# Gradient + noise
# noise: 1.random  2.Gaussian Noise 
def gd_direct_noise(X_train, y_train, theta_l, theta_r, target, lambda_=lambda_, stop_val=0.01, epoch=1000, lr=0.00001, theta=None):
    print("--------------------------------------------------------------")
    print('---- Gradient Direct Noise Descent---- ')
    print('====Start Training====')
    start_time = time.time()

    loop_list = list()
    loss_list = list()

    # if theta is None:
    Theta = var(random.uniform(theta_l, theta_r), requires_grad=True)
    # else:
    #     Theta = theta
    # Theta = var(2.933, requires_grad=True)
    # root = construct_syntax_tree(Theta)
    root_smooth_point = construct_syntax_tree_smooth_point(Theta)
    root_point = construct_syntax_tree_point(Theta)
    # Theta = var(69.9)

    for i in range(epoch):
        # Theta = var(4.272021770477295, requires_grad=True)
        # if i == 1:
        #     Theta.data = var(5.303304672241211)

        symbol_table_list = initialization(x_l, x_r)

        f = var(0.0)
        y_l = P_INFINITY
        y_r = N_INFINITY

        for idx, x in enumerate(X_train):
            x, y = x, y_train[idx]
            symbol_table_smooth_point = initialization_point(x)
            # print('run smooth')
            symbol_table_smooth_point = root_smooth_point['entry'].execute(symbol_table_smooth_point)
            # print('run point')

            # symbol_table_point = initialization_point(x)
            # symbol_table_point = root_point['entry'].execute(symbol_table_point)

            # y_l = torch.min(symbol_table_point['res'], y_l)
            # y_r = torch.max(symbol_table_point['res'], y_r)

            # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
            f = f.add(distance_f_point(symbol_table_smooth_point['res'], var(y)))

        f = f.div(var(len(X_train)))
        print('quantitive f', f.data.item())

        # symbol_table_list = root['entry'].execute(symbol_table_list)
        # print('length: ', len(symbol_table_list))
        # res_l, res_r = extract_result_safty(symbol_table_list)
        # penalty_f = distance_f_interval(symbol_table_list, target)
        # print('safe f', penalty_f.data.item(), res_l, res_r) # , y_l.data.item(), y_r.data.item())

        res = f # f.add(lambda_.mul(penalty_f))
        print(i, '--', Theta.data.item(), res.data.item())
        # if i == 0:
        #     continue
        # if i == 1:
        #     exit(0)
        # exit()
        derivation = var(0.0)
        try:
            dTheta = torch.autograd.grad(res, Theta, retain_graph=True)
            derivation = dTheta[0]
            # print('f, theta, dTheta:', f.data, Theta.data, derivation)

            if torch.abs(res.data) < var(stop_val): # epsilon:
                # print(f.data, Theta.data)
                break
            if torch.abs(derivation.data) < EPSILON:
                Theta.data.fill_(random.uniform(theta_l, theta_r))
                continue
            Theta.data -= lr * (derivation.data + var(random.uniform(-noise, noise)))
            print('deriavation, theta ', derivation.data.item(), Theta.data.item())
        
        except RuntimeError:
            # print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
            if torch.abs(res.data) < var(stop_val):
                # print(f.data, Theta.data)
                break

            Theta.data -= lr * (derivation.data + var(random.uniform(-1.0, 1.0)))
        
        if Theta.data.item() <= theta_l or Theta.data.item() >= theta_r:
            Theta.data.fill_(random.uniform(theta_l, theta_r))
            continue

        loop_list.append(i)
        loss_list.append(res.data)
    
    # plt.plot(loop_list, loss_list, label = "beta")
    # plt.xlabel('expr count')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()
    # print('GOT! Loss, theta', f.data, Theta.data)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")

    theta = Theta# .data.item()
    loss = res# .data.item()
    print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta.data.item(), loss.data.item()))

    return theta, loss, loss_list, f, var(0.0)


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

        # symbol_table_list = root['entry'].execute(symbol_table_list)
        # print('length: ', len(symbol_table_list))
        # res_l, res_r = extract_result_safty(symbol_table_list)
        # penalty_f = distance_f_interval(symbol_table_list, target)
        # print('safe f', penalty_f.data.item(), res_l, res_r) # , y_l.data.item(), y_r.data.item())

        res = f # f.add(lambda_.mul(penalty_f))
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

    return theta, loss, loss_list, f, var(0.0)


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

    return theta, loss, loss_list, f, var(0.0)


def cal_c(X_train, y_train, theta):
    root = construct_syntax_tree(theta)
    symbol_table_list = initialization(x_l, x_r)

    symbol_table_list = root['entry'].execute(symbol_table_list)
    c = distance_f_interval(symbol_table_list, target)

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
    
    return q