import numpy as np
from termcolor import colored

from data_generator import *
from optimization import *


def check_safety(res, target):
    intersection_interval = get_intersection(res, target)
    if intersection_interval.isEmpty():
        # print('isempty')
        score = torch.max(target.left.sub(res.left), res.right.sub(target.right)).div(res.getLength())
    else:
        # print('not empty')
        score = var(1.0).sub(intersection_interval.getLength().div(res.getLength()))
    
    return score


def eval(X, Y, theta, target, category):
    quan_dist = var(0.0)
    safe_dist = var(0.0)
    Theta = var(theta)

    root = construct_syntax_tree(Theta)
    root_point = construct_syntax_tree_point(Theta)
    root_smooth_point = construct_syntax_tree_smooth_point(Theta)
    y_pred = list()
    y_min = P_INFINITY.data.item()
    y_max = N_INFINITY.data.item()
    safe_min = P_INFINITY.data.item()
    safe_max = N_INFINITY.data.item()

    '''
    # add for program test_disjuction_2
    '''
    ###########################################
    x = [theta.data.item()-0.0101]
    symbol_table_point = initialization_point(x)
    root_target_point = construct_syntax_tree_point(var(5.49))
    symbol_table_point = root_target_point['entry'].execute(symbol_table_point)
    # safe_l = symbol_table_point['x_min'].data.item()
    # safe_r = symbol_table_point['x_max'].data.item()
    y = symbol_table_point['res'].data.item()
    # print('add data:', theta.data.item(), y, safe_l, safe_r)
    X = list(X)
    X.append(x)
    X = np.array(X)
    Y = list(Y)
    Y.append([y])
    Y = np.array(Y)
    ############################################

    # quantative distance
    for idx, x in enumerate(X):
        x, y = x, Y[idx]
        symbol_table_point = initialization_point(x)
        symbol_table_point = root_smooth_point['entry'].execute(symbol_table_point)
        quan_dist = quan_dist.add(distance_f_point(symbol_table_point['res'], var(y)))

        y_pred.append(symbol_table_point['res'].data.item())

        symbol_table_point = initialization_point(x)
        symbol_table_point = root_point['entry'].execute(symbol_table_point)

        real_y = symbol_table_point['res'].data.item()
        y_min = min(real_y, y_min)
        y_max = max(real_y, y_max)

        safe_property_min = symbol_table_point['x_min'].data.item()
        safe_property_max = symbol_table_point['x_max'].data.item()
        safe_min = min(safe_property_min, safe_min)
        safe_max = max(safe_property_max, safe_max)

    quan_dist = quan_dist.div(len(X))
    # symbol_table_rep = initialization(x_l, x_r)
    # symbol_table_rep = root.execute(symbol_table_rep)
    print('real y interval', y_min, y_max)
    print('real x interval', safe_min, safe_max)
    safe_res = domain.Interval(safe_min, safe_max)
    safe_dist = check_safety(safe_res, target)

    print(category + ':')
    print('Quantative Objective: {0:.5f}, Safe Objective: {1:.5f}'.format(quan_dist.data.item(), safe_dist.data.item()))
    if safe_dist.data.item() > 0.0:
        print(colored('Not Safe!', 'red'))
    else:
        print(colored('Safe!', 'green'))

    return

    # return quan_dist.data.item(), safe_dist.data.item(), np.array(y_pred)