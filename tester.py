import numpy as np
from termcolor import colored

from constants import *
import constants
import importlib

from utils import (
    batch_pair_trajectory
    )

import import_hub as hub
importlib.reload(hub)
from import_hub import *


def trajectory2points(trajectory_list, bs):
    states, actions = list(), list()
    for trajectory in trajectory_list:
        for (state, action) in trajectory:
            states.append(state)
            actions.append([action])
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)

    for i in range(0, len(states), bs):
        if torch.cuda.is_available():
            yield torch.from_numpy(states[i:i+bs]).float().cuda(), torch.from_numpy(actions[i:i+bs]).float().cuda()
        else:
            yield torch.from_numpy(states[i:i+bs]).float(), torch.from_numpy(actions[i:i+bs]).float()


def test_objective(m, trajectory_test, criterion, test_bs):
    data_loss = 0.0
    count = 0
    test_bs = 8
    if constants.benchmark_name in ['thermostat']:
        X, y_trajectory = batch_pair_trajectory(trajectory_test, data_bs=None, standard_value=70.0)
        X, y_trajectory = torch.from_numpy(X).float(), [torch.from_numpy(y).float() for y in y_trajectory]
        if torch.cuda.is_available():
            X, y_trajectory = X.cuda(), [y.cuda() for y in y_trajectory]
        yp_trajectory = m(X, version="single_nn_learning")
        test_data_loss = var(0.0)
        for idx, yp in enumerate(yp_trajectory):
            test_data_loss = test_data_loss + criterion(yp, y_trajectory[idx])
            test_data_loss /= len(yp_trajectory)
    else:
        # for x, y in trajectory2points(trajectory_test, test_bs):
        for x, y in batch_pair_yield(trajectory_test, data_bs=test_bs):
            x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
            # print(f"after batch pair: {X.shape}")
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            yp = m(x, version="single_nn_learning")
            batch_data_loss = criterion(yp, y)
            if debug:
                print(f"yp: {yp.squeeze()}, y: {y.squeeze()}")
                print(f"batch data loss: {batch_data_loss}")
            count += 1
            data_loss += batch_data_loss
        test_data_loss = data_loss / count

    if not debug:
        log_file_evaluation = open(constants.file_dir_evaluation, 'a')
        log_file_evaluation.write(f"test data loss: {float(test_data_loss)}\n")
    print(f"test data loss: {float(test_data_loss)}")


def test_data_loss(
    model_path, 
    model_name, 
    trajectory_test, 
    target, 
    test_bs=512,
    test_abstract_bs=32):
    m = Program(l=l, nn_mode=nn_mode)

    _, m = load_model(m, MODEL_PATH, name=model_name)
    if m is None:
        print(f"No model to Tester!!")
        return 
    # m.cuda()
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False

    criterion = torch.nn.MSELoss()
    test_objective(m, trajectory_test, criterion, test_bs)

    




