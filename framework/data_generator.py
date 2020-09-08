import random
import pandas as pd
import numpy as np

from helper import *
from constants import *

# from program1 import *
# from program2 import *
from program3 import *

# def show_tmp_x_y(symbol_table):
#     print('symbol_table:')
#     print(symbol_table['x1'], symbol_table['y1'], symbol_table['x2'], symbol_table['y2'])

def data_generator(l, r, size, target_theta, test_size=0.33):
    root = construct_syntax_tree_point(var(target_theta))
    data_list = list()

    for i in range(size):
        x = list()
        for idx, v in enumerate(l):
            tmp_x = random.uniform(l[idx], r[idx])
            x.append(tmp_x)

        # print('x', x)
        symbol_table_point = initialization_point(x)
        symbol_table_point = root['entry'].execute(symbol_table_point)

        y = symbol_table_point['res'].data.item()
        safety = symbol_table_point['x'].data.item()

        data = [x, y]
        # print(x, y)
        # print('safety dist', safety)
        # show_tmp_x_y(symbol_table_point)
        # exit(0)
        data_list.append(data)
    
    data_list = np.array(data_list)
    
    split_idx = int(size * test_size)
    test_list = data_list[:split_idx]
    train_list = data_list[split_idx:]

    # X_train, X_test, y_train, y_test
    # exit(0)
    return train_list[:, 0], test_list[:, 0], train_list[:, 1], test_list[:, 1]