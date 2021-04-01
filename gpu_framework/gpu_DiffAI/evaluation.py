import numpy as np
from termcolor import colored

from constants import *
from data_generator import *
# from optimization import *


def check_safety(res, target):
    intersection_interval = get_intersection(res, target)
    if intersection_interval.isEmpty():
        # print('isempty')
        score = torch.max(target.left.sub(res.left), res.right.sub(target.right)).div(res.getLength())
    else:
        # print('not empty')
        score = var(1.0).sub(intersection_interval.getLength().div(res.getLength()))
    
    return score


def eval(X, Y, m, target, category):
    quan_dist = var(0.0)
    safe_dist = var(0.0)
    # Theta = var_list(theta)

    # root = construct_syntax_tree(Theta)
    # root_point = construct_syntax_tree_point(Theta)
    # root_smooth_point = construct_syntax_tree_smooth_point(Theta)
    y_pred = list()
    y_min = P_INFINITY.data.item()
    y_max = N_INFINITY.data.item()
    safe_min = P_INFINITY.data.item()
    safe_max = N_INFINITY.data.item()

    # quantative distance
    data_loss = var(0.0)
    for idx, x in enumerate(X):
        point, label = x, Y[idx]
        point_data = initialization_point_nn(point)
        y_point_list = m(point_data, 'concrete')
        data_loss += distance_f_point(y_point_list[0]['x'].c[2], var(label))
        # symbol_table_point = root_point['entry'].execute(symbol_table_point)

        y_pred.append(y_point_list[0]['x'].c[2].data.item())

        safe_property_min = y_point_list[0]['safe_range'].left.data.item()
        safe_property_max = y_point_list[0]['safe_range'].right.data.item()
        safe_min = min(safe_property_min, safe_min)
        safe_max = max(safe_property_max, safe_max)

    quan_dist = data_loss.div(len(X))
    # symbol_table_rep = initialization(x_l, x_r)
    # symbol_table_rep = root.execute(symbol_table_rep)
    # print('real y interval', y_min, y_max)
    print('real safe interval', safe_min, safe_max)
    safe_res = domain.Interval(var(safe_min), var(safe_max))
    safe_dist = check_safety(safe_res, target)

    print(category + ':')
    print('Quantative Objective: {0:.5f}, Safe Objective: {1:.5f}'.format(quan_dist.data.item(), safe_dist.data.item()))
    if safe_dist.data.item() > 0.0: # TODO: set to epsilon?
        print(colored('Not Safe!', 'red'))
    else:
        print(colored('Safe!', 'green'))
    
    log_file = open(file_dir, 'a')
    # Quantitative loss & safe
    log_file.write(f"Real Safe Interval: [{safe_min}, {safe_max}]\n")
    log_file.write('Test:' + str(quan_dist.data.item()) + ',' + str(safe_dist.data.item()) + '\n')
    log_file.close()

    return


def verification(m, component_list, target):
    cal_safe_loss
