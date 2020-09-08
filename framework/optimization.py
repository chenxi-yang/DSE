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


if MODE == 2 or MODE==3:
    def distance_f_interval(X_list, target):

        res = var(0.0)
        for X_table in X_list:
            X = X_table['x']
            p = X_table['probability']

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
            tmp_res = tmp_res.mul(p)

            res = res.add(tmp_res)
        
        return res


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


# DIRECT
def direct(X_train, y_train, theta_l, theta_r, target, stop_val, epoch):
    print("--------------------------------------------------------------")
    print('----DIRECT----')
    print('====Start Training====')
    start_time = time.time()

    loss_list = list()

    def myfunc(theta, grad):
        Theta = var(theta)
        # Theta = var(70.0)
        root = construct_syntax_tree(Theta)
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
        # print('quantitive f', f.data.item())

        symbol_table_list = root['entry'].execute(symbol_table_list)
        # show_symbol_tabel_list(symbol_table_list)
        # print(len(symbol_table_list))
        penalty_f = distance_f_interval(symbol_table_list, target)
        # print('safe f', penalty_f.data.item())

        f = f.add(var(lambda_).mul(penalty_f))
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
    
    return theta, loss, loss_list


# Gradient + noise
# noise: 1.random  2.Gaussian Noise 
def gd_direct_noise(X_train, y_train, theta_l, theta_r, target, stop_val, epoch, lr):
    print("--------------------------------------------------------------")
    print('---- Gradient Direct Noise Descent---- ')
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
        print(Theta.data.item(), f.data.item())

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
        
            Theta.data += lr * (derivation.data + var(random.uniform(-20.0, 20.0)))
        
        except RuntimeError:
            # print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
            if torch.abs(f.data) < var(stop_val):
                # print(f.data, Theta.data)
                break

            Theta.data += lr * (derivation.data + var(random.uniform(-20.0, 20.0)))
        
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

    theta = Theta.data.item()
    loss = f.data.item()
    print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta, loss))

    return theta, loss, loss_list


def gd_gaussian_noise(X_train, y_train, theta_l, theta_r, target, stop_val, epoch, lr):

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
        f = f.div(var(len(X_train)))
        # print('quantitive f', f.data.item())

        symbol_table_list = root['entry'].execute(symbol_table_list)
        penalty_f = distance_f_interval(symbol_table_list, target)

        # print('safe f', penalty_f.data.item())

        f = f.add(var(lambda_).mul(penalty_f))
        # print(Theta.data.item(), f.data.item())

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
        
            Theta.data += lr * (derivation.data + var(generate_gaussian_noise(step)))
            # print('noise', generate_gaussian_noise(step))
            step += 1
        
        except RuntimeError:
            # print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
            if torch.abs(f.data) < var(stop_val):
                # print(f.data, Theta.data)
                break

            Theta.data += lr * (derivation.data + var(generate_gaussian_noise(step)))
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

    theta = Theta.data.item()
    loss = f.data.item()
    print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta, loss))

    return theta, loss, loss_list


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
        print(Theta.data.item(), f.data.item())

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
        
            Theta.data += lr * derivation.data
        
        except RuntimeError:
            # print('RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn')
            if torch.abs(f.data) < var(stop_val):
                # print(f.data, Theta.data)
                break

            Theta.data += lr * derivation.data
        
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

    theta = Theta.data.item()
    loss = f.data.item()
    print('Theta: {0:.3f}, Loss: {1:.3f}'.format(theta, loss))

    return theta, loss, loss_list

