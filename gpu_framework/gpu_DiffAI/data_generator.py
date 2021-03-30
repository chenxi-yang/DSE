import random
from random import shuffle
import pandas as pd
import numpy as np
import time
from timeit import default_timer as timer

from helper import *
from constants import *


def load_data(
    train_size,
    test_size,
    dataset_path=DATASET_PATH,
    ):
    start_t = time.time()
    f = open(dataset_path, 'r')
    f.readline()
    data_list = list()
    for line in f:
        content = line[:-1].split(",")
        x, y = [float(content[0])], float(content[1])
        data_list.append([x, y])

    data_list = np.array(data_list)
    
    shuffle(data_list)
    train_list = data_list[:train_size]
    test_list = data_list[train_size:train_size + test_size]

    # X_train, X_test, y_train, y_test
    print("---Data Generation---")
    print("--- %s seconds ---" % (time.time() - start_t))
    return train_list[:, 0], test_list[:, 0], train_list[:, 1], test_list[:, 1]