import random
from random import shuffle
import pandas as pd
import numpy as np
import time
from timeit import default_timer as timer

from helper import *
from constants import *

# from program1 import *
# from program2 import *
# from program3 import *
# from program4 import *
# from program5 import *
# from program6 import *
# from program6_loop import *
# from program7 import *
from program8 import *
# from program_test_disjunction import *
# from program_test_disjunction_2 import *


# def show_tmp_x_y(symbol_table):
#     print('symbol_table:')
#     print(symbol_table['x1'], symbol_table['y1'], symbol_table['x2'], symbol_table['y2'])


def data_generator(l, r, size, target_theta, test_size=0.33):
    # start_t = time.time()
    start_t = timer()

    root = construct_syntax_tree_point(var(target_theta))
    data_list = list()

    y_min = P_INFINITY.data.item()
    y_max = N_INFINITY.data.item()
    for i in range(size):
        # print('in data_generator')
        x = list()
        for idx, v in enumerate(l):
            tmp_x = random.uniform(l[idx], r[idx])
            # tmp_x = r[idx]# l[idx] + i*(r[idx] - l[idx])*1.0/size # random.uniform(l[idx], r[idx])
            x.append(tmp_x)
        # x.append((target_theta-0.0045)/3.0)
        # x.append((target_theta-0.004)/3.0)
        # x.append((target_theta-0.005)/3.0)
        # x = [1.0393192728816127, 1.7036774083478434]

        
        symbol_table_point = initialization_point(x)
        symbol_table_point = root['entry'].execute(symbol_table_point)

        y = symbol_table_point['res'].data.item()
        safety_l = symbol_table_point['x_min'].data.item()
        safety_r = symbol_table_point['x_max'].data.item()

        data = [x, y]
        y_min = min(y, y_min)
        y_max = max(y, y_max)
        # print('x', x)
        # print('y', y)
        # print(x, y)
        # print('safety dist', safety)
        # show_tmp_x_y(symbol_table_point)
        # exit(0)
        data_list.append(data)
    
    data_list = np.array(data_list)
    
    split_idx = int(size * test_size)
    shuffle(data_list)
    test_list = data_list[:split_idx]
    train_list = data_list[split_idx:]

    # X_train, X_test, y_train, y_test
    # exit(0)
    print("---Data Generation---")
    print("--- %s seconds ---" % (timer() - start_t))
    print("---Data Range---[" + str(y_min) + ',' + str(y_max) + "]")
    return train_list[:, 0], test_list[:, 0], train_list[:, 1], test_list[:, 1]