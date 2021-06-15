import numpy as np
from termcolor import colored

from constants import *
# from optimization import *

if benchmark_name == "thermostat":
    from thermostat_nn_batch import (
        ThermostatNN,
        load_model,
        save_model,
        initialization_abstract_state,
        initialization_point_nn,
    )
if benchmark_name == "mountain_car":
    from mountain_car_batch import (
        MountainCar,
        load_model,
        save_model,
        initialization_abstract_state,
        initialization_point_nn,
    )
if benchmark_name == "unsound_1":
    from unsound_1_batch import (
        Unsound_1,
        load_model,
        save_model,
        initialization_abstract_state,
        initialization_point_nn,
    )
if benchmark_name == "sampling_1":
    from sampling_1_batch import (
        Sampling_1,
        load_model,
        save_model,
        initialization_abstract_state,
        # initialization_point_nn,
    )
if benchmark_name == "sampling_2":
    from sampling_2_batch import (
        Sampling_2,
        load_model,
        save_model,
        initialization_abstract_state,
        # initialization_point_nn,
    )
if benchmark_name == "path_explosion":
    from path_explosion_sound import (
        PathExplosion,
        load_model,
        save_model,
        initialization_abstract_state,
    )
if benchmark_name == "path_explosion_2":
    from path_explosion_2_sound import (
        PathExplosion2,
        load_model,
        save_model,
        initialization_abstract_state,
    )
if benchmark_name == "fairness_1":
    from fairness_1_batch import (
        Fairness_1,
        load_model,
        save_model,
        initialization_abstract_state,
    )

import domain

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
    for x, y in trajectory2points(trajectory_test, test_bs):
        yp = m(x, version="single_nn_learning")
        batch_data_loss = criterion(yp, y)
        if debug:
            print(f"yp: {yp.squeeze()}, y: {y.squeeze()}")
            print(f"batch data loss: {batch_data_loss}")

        count += 1
        data_loss += batch_data_loss
        # update data_loss
    # print/f.write()
    test_data_loss = data_loss / count

    if not debug:
        log_file_evaluation = open(file_dir_evaluation, 'a')
        log_file_evaluation.write(f"test data loss: {test_data_loss.data.item()}\n")
    print(f"test data loss: {test_data_loss.data.item()}")
    if debug:
        exit(0)
        pass


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

    




