import random
from random import shuffle
import pandas as pd
import numpy as np
import time
from timeit import default_timer as timer

from constants import *


def load_data(
    train_size,
    test_size,
    dataset_path,
    ):
    start_t = time.time()
    f = open(dataset_path, 'r')
    f.readline()
    data_list = list()
    for line in f:
        content = line[:-2].split(';')
        trajectory_list = list()
        for state_content in content:
            # print(state_content)
            state_content = state_content[1:-1].split(',')
            state_list = [float(v) for v in state_content]
            #! only the last value represents the action
            state, action = state_list[:-1], state_list[-1]
            trajectory_list.append([state, action])
        data_list.append(trajectory_list)

    data_list = np.array(data_list)
    
    shuffle(data_list)
    trajectory_train_list = data_list[:train_size]
    trajectory_test_list = data_list[train_size:train_size + test_size]

    # X_train, X_test, y_train, y_test
    print("---Data Generation---")
    print("--- %s seconds ---" % (time.time() - start_t))
    # return train_list[:, 0], test_list[:, 0], train_list[:, 1], test_list[:, 1]
    return trajectory_train_list, trajectory_test_list