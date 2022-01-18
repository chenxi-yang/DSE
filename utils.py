from scipy.stats import truncnorm
from scipy.stats import poisson

import numpy as np
import random
import math

import time
import torch
from torch.autograd import Variable
import os

np.random.seed(seed=1)
random.seed(1)


def var(i, requires_grad=True):
    if torch.cuda.is_available():
        return Variable(torch.tensor(i, dtype=torch.float).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.tensor(i, dtype=torch.float), requires_grad=requires_grad)


def var_list(i_list, requires_grad=True):
    if torch.cuda.is_available():
        res = torch.tensor(i_list, dtype=torch.float, requires_grad=requires_grad).cuda()
    else:
        res = torch.tensor(i_list, dtype=torch.float, requires_grad=requires_grad)
    return res


PI = var((3373259426.0 + 273688.0 / (1 << 21)) / (1 << 30))
PI_TWICE = PI.mul(var(2.0))
PI_HALF = PI.div(var(2.0))


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


def ini_trajectory(trajectory):
    state = trajectory[0][0]
    action = trajectory[0][1]
    return state, action


def batch_pair_yield(trajectory_list, data_bs=None):
    states, actions = list(), list()
    random.shuffle(trajectory_list)
    for trajectory in trajectory_list:
        for (state, action) in trajectory:
            states.append(state)
            actions.append(action)
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)
    k = int((len(states) - 1) / data_bs)
    # print(k)
    for i in range(k+1):
        yield states[data_bs*i:data_bs*(i+1)], actions[data_bs*i:data_bs*(i+1)]
    

def batch_pair(trajectory_list, data_bs=None):
    states, actions = list(), list()
    random.shuffle(trajectory_list)
    for trajectory in trajectory_list:
        for (state, action) in trajectory:
            states.append(state)
            actions.append(action)
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)
    return states[:data_bs], actions[:data_bs]
    


def batch_pair_endpoint(trajectory_list, data_bs=None):
    states, actions = list(), list()
    random.shuffle(trajectory_list)
    for trajectory in trajectory_list:
        ini_state, ini_action = trajectory[0]
        last_state, last_action = trajectory[-1]
        states.append(ini_state)
        actions.append([last_action])
    
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)
    return states[:data_bs], actions[:data_bs]
    


def batch_pair_endpoint(trajectory_list, data_bs=None):
    states, actions = list(), list()
    random.shuffle(trajectory_list)
    for trajectory in trajectory_list:
        ini_state, ini_action = trajectory[0]
        last_state, last_action = trajectory[-1]
        states.append(ini_state)
        actions.append([last_action])
    
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)

    return states[:data_bs], actions[:data_bs]


def batch_pair_trajectory(trajectory_list, data_bs=None, standard_value=0.0):
    # thermostat has the same length for all, for empty location, use the distance to 70.0
    random.shuffle(trajectory_list)
    max_len = 0
    for trajectory in trajectory_list:
        max_len = max(len(trajectory), max_len)
    ini_states, data_trajectories = list(), list()
    # print(f"max len: {max_len}, len tra: {len(trajectory_list)}")
    for trajectory in trajectory_list:
        for idx, (state, action) in enumerate(trajectory):
            if idx == 0:
                ini_states.append(state)
                # print(len(ini_states))
            if idx >= len(data_trajectories):
                data_trajectories.append([[action]])
            else:
                data_trajectories[idx].append([action])
        while(idx < max_len - 1):
            if idx >= len(data_trajectories):
                data_trajectories.append([[standard_value]])
            else:
                data_trajectories[idx].append([standard_value])
            idx += 1
    # c = list(zip(ini_states, data_trajectories))
    # print(len(ini_states), len(data_trajectories))
    # random.shuffle(c)
    # ini_states, data_trajectories = zip(*c)
    ini_states = np.array(ini_states)
    trajectories = list()
    for step in data_trajectories:
        trajectories.append(np.array(step))
    # print(f"states: {ini_states.shape}")
    # print(f"test: {trajectories[0].shape}")
    # exit(0)
    
    return ini_states, trajectories


def batch_pair_trajectory(trajectory_list, data_bs=None, standard_value=0.0):
    # thermostat has the same length for all, for empty location, use the distance to 70.0
    random.shuffle(trajectory_list)
    max_len = 0
    for trajectory in trajectory_list:
        max_len = max(len(trajectory), max_len)
    ini_states, data_trajectories = list(), list()
    # print(f"max len: {max_len}, len tra: {len(trajectory_list)}")
    for trajectory in trajectory_list:
        for idx, (state, action) in enumerate(trajectory):
            if idx == 0:
                ini_states.append(state)
                # print(len(ini_states))
            if idx >= len(data_trajectories):
                data_trajectories.append([[action]])
            else:
                data_trajectories[idx].append([action])
        while(idx < max_len - 1):
            if idx >= len(data_trajectories):
                data_trajectories.append([[standard_value]])
            else:
                data_trajectories[idx].append([standard_value])
            idx += 1
    # c = list(zip(ini_states, data_trajectories))
    # print(len(ini_states), len(data_trajectories))
    # random.shuffle(c)
    # ini_states, data_trajectories = zip(*c)
    ini_states = np.array(ini_states)
    trajectories = list()
    for step in data_trajectories:
        trajectories.append(np.array(step))
    # print(f"states: {ini_states.shape}")
    # print(f"test: {trajectories[0].shape}")
    # exit(0)
    
    return ini_states, trajectories


def batch_points(l):
    # list of elements
    # each element is a list
    L = np.array(l)
    if torch.cuda.is_available():
        res = torch.from_numpy(L).float().cuda()
    else:
        res = torch.from_numpy(L).float()
    return res
    

def show_cuda_memory(name):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    print(f"{name}, free mem: {f}, reserved mem: {r}, allocated mem: {a}")


def show_memory_snapshot():
    snapshot = torch.cuda.memory_snapshot()
    for d in snapshot:
        print(d["segment_type"], d["active_size"], d["allocated_size"], d["total_size"])
    

def show_component(component_list):
    log_component_list = list()
    for component in component_list:
        log_component_list.append((component['center'], component['width']))
    print(f"show component: {log_component_list}")


def show_trajectory(abstract_states):
    for abstract_state in abstract_states:
        for symbol_table in abstract_state:
            X = symbol_table['trajectory'][0][0]
            print(f"first in symbol_table[0]: {float(X.left)}, {float(X.right)}")


def get_truncated_normal_width(mean, std, width):
    try:
        truncated_normal = truncnorm((mean-width-mean)/std, (mean+width-mean)/std, loc=mean, scale=std)
        res = truncated_normal.rvs()
    except ValueError:
        width = width * 2.0
        truncated_normal = truncnorm((mean-width-mean)/std, (mean+width-mean)/std, loc=mean, scale=std)
        res = truncated_normal.rvs()
    return res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def append_log(f_list, log_txt):
    for f_name in f_list:
        f = open(f_name, 'a')
        f.write(log_txt)
        f.close()

def create_components(x_l, x_r, num_components):
    # num_components: each dimension
    # just cut the x[0] space
    x_min, x_max = x_l[0], x_r[0]
    x_c, x_width = list(), list()
    # print(x_l, x_r)
    for idx, idx_l in enumerate(x_l):
        idx_r = x_r[idx]
        x_c.append((idx_r + idx_l)/2)
        x_width.append((idx_r - idx_l)/2)
    component_length = (x_max - x_min) / num_components
    components = list()
    if len(x_l) == 1:
        for i in range(num_components):
            l = x_min + i * component_length
            r = x_min + (i + 1) * component_length
            center = [(r + l) / 2.0]
            width = [(r - l) / 2.0]
            for temp_c in x_c[1:]:
                center.append(temp_c)
            for temp_width in x_width[1:]:
                width.append(temp_width)
            # print(x_c[1:])
            # print(center, x_c)
            # center.extend[x_c[1:]]
            # width.extend[x_width[1:]]
            component_group = {
                'center': center,
                'width': width,
            }
            components.append(component_group)
    elif len(x_l) == 4: # for cartpole task: for same dimensions
        for i in range(num_components):
            l = x_min + i * component_length
            r = x_min + (i + 1) * component_length
            center_0 = (r + l) / 2.0
            width_0 = (r - l) / 2.0
            for j in range(num_components):
                l_1, r_1 = x_min + j * component_length, x_min + (j + 1) * component_length
                center_1, width_1 = (r_1 + l_1) / 2.0, (r_1 - l_1) / 2.0
                for m in range(num_components):
                    l_2, r_2 = x_min + m * component_length, x_min + (m + 1) * component_length
                    center_2, width_2 = (r_2 + l_2) / 2.0, (r_2 - l_2) / 2.0
                    for n in range(num_components):
                        l_3, r_3 = x_min + n * component_length, x_min + (n + 1) * component_length
                        center_3, width_3 = (r_3 + l_3) / 2.0, (r_3 - l_3) / 2.0
                        component_group = {
                            'center': [center_0, center_1, center_2, center_3],
                            'width': [width_0, width_1, width_2, width_3],
                        }
                        components.append(component_group)

    return components


def in_component(X, component):
    center = component['center']
    width = component['width']
    for i, x in enumerate(X):
        if x >= center[i] - width[i] and x < center[i] + width[i]:
            pass
        else:
            return False
    return True


# Current Support: partition one dimension
def extract_abstract_representation(trajectories, x_l, x_r, num_components):
    # extract components
    # interval
    # and all the trajectories starting from that interval
    # x_min, x_max = x_l[0], x_r[0]
    # components = create_components(x_l, x_r, num_components)
    x_min, x_max = x_l[0], x_r[0]
    components = create_components(x_min, x_max, num_components)
    for idx, component in enumerate(components):
        component.update(
            {
                'trajectories': list(),
            }
        )
        for trajectory in trajectories:
            state, _ = ini_trajectory(trajectory)
            if in_component([state[0]], component):
                component['trajectories'].append(trajectory)
        components[idx] = component
    
    return components


def divide_chunks(components_list, bs=1, data_bs=None):
    '''
    component: {
        'center': 
        'width':
        'trajectories':
    }
    bs: number of components in a batch
    data_bs: number of trajectories to aggregate the training points

    # components, bs, data_bs
    # whenever a components end, return the components, otherwise 

    return: refined trajectores, return abstract_states

    abstract_states: {
        'center': batched center,
        'width': batched width,
    }
    '''
    for i in range(0, len(components_list), bs):
        components = components_list[i:i + bs]
        abstract_states = dict()
        trajectories = list()
        center_list, width_list = list(), list()
        count_trajectory = 0
        for component_idx, component in enumerate(components):
            center_list.append(component['center'])
            width_list.append(component['width'])
            for trajectory_idx, trajectory in enumerate(component['trajectories']):
                trajectories.append(trajectory)
                if data_bs is None:
                    pass
                else:
                    count_trajectory += 1
                    if count_trajectory > data_bs:
                        batched_center, batched_width = batch_points(center_list), batch_points(width_list)
                        abstract_states['center'] = batched_center
                        abstract_states['width'] = batched_width
                        res_trajectories = trajectories
                        trajectories = list()
                        count_trajectory = 0
                        yield res_trajectories, abstract_states

        batched_center, batched_width = batch_points(center_list), batch_points(width_list)
        abstract_states['center'] = batched_center
        abstract_states['width'] = batched_width

        yield trajectories, abstract_states


def create_abstract_states_from_components(components):
    center_list, width_list = list(), list()
    abstract_states = dict()
    for component_idx, component in enumerate(components):
        center_list.append(component['center'])
        width_list.append(component['width'])
    batched_center, batched_width = batch_points(center_list), batch_points(width_list)
    abstract_states['center'] = batched_center
    abstract_states['width'] = batched_width

    return abstract_states

def load_model(m, folder, name, epoch=None):
    if os.path.isfile(folder):
        m.load_state_dict(torch.load(folder))
        return None, m
    model_dir = os.path.join(folder, f"model_{name}")
    if not os.path.exists(model_dir):
        return None, None
    if epoch is None and os.listdir(model_dir):
        epoch = max(os.listdir(model_dir), key=int)
    path = os.path.join(model_dir, str(epoch))
    if not os.path.exists(path):
        return None, None
    m.load_state_dict(torch.load(path))
    return int(epoch), m


def save_model(model, folder, name, epoch):
    path = os.path.join(folder, f"model_{name}", str(epoch))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model.state_dict(), path)


def import_benchmarks(benchmark_name):
    if benchmark_name == "thermostat":
        from benchmarks.thermostat import (
            initialize_components,
            Program,
        )
    elif benchmark_name == "mountain_car":
        from benchmarks.mountain_car import (
            initialize_components,
            Program,
        )
    elif benchmark_name == "unsmooth_1":
        from benchmarks.unsmooth import (
            initialize_components,
            Program,
        )
    elif benchmark_name == "unsmooth_2_separate":
        from benchmarks.unsmooth_2_separate import (
            initialize_components,
            Program,
        )
    elif benchmark_name == "unsmooth_2_overall":
        from benchmarks.unsmooth_2_overall import (
            initialize_components,
            Program,
        )
    elif benchmark_name == "path_explosion":
        from benchmarks.path_explosion import (
            initialize_components,
            Program,
        )
    elif benchmark_name == "path_explosion_2":
        from benchmarks.path_explosion_2 import (
            initialize_components,
            Program,
        )


def product(it):
    if isinstance(it, int):
        return it
    product = 1
    for x in it:
        if x >= 0:
            product *= x
    return product


def index_conversion_second_dim(x, index):
    return (torch.arange(x.size(0)), index)


def select_argmax(interval_left, interval_right):
    # lower bound and upper bound of the interval concretization
    assert(interval_left.shape == interval_right.shape)

    B, M = interval_right.shape
    index_mask = torch.zeros((B, M), dtype=torch.bool)
    all_ones = torch.ones(interval_right.shape) + 1.0 # should be larger than 1.0
    if torch.cuda.is_available():
        index_mask = index_mask.cuda()
        all_ones = all_ones.cuda()
    
    # print(f"interval_left: {interval_left}")
    # print(f"interval_right: {interval_right}")
    
    # interval: K x M
    max_right_index = index_conversion_second_dim(interval_right, torch.argmax(interval_right, dim=1))
    for i in range(M - 1):
        # print(f"max_right_index: {max_right_index}")
        # convert to the shape of K x M
        max_left_value = interval_left[max_right_index][:, None]
        # print(f"max_left_value: {max_left_value}")
        # if an interval's right >= the max interval's left, it is markd in max as well
        # >= is for the interval where delta == 0
        # print(f"in index: {interval_right >= max_left_value}")
        index_mask[interval_right >= max_left_value] = True

        # select all the lower bound of a interval in the argmax set, otherwise 1.0
        left_already_in = torch.where(index_mask, interval_left, all_ones)
        # print(f"left_already_in: {left_already_in}")

        max_right_index = index_conversion_second_dim(interval_right, torch.argmin(left_already_in, dim=1))
    
    # print(f"index_mask: {index_mask}")
    # exit(0)
    return index_mask


def aggregate_sampling_states(abstract_states, sample_size, max_allowed=600):
    aggregated_abstract_states_list = list()
    center, width = abstract_states['center'], abstract_states['width']
    
    aggregated_center = center.repeat(sample_size, 1)
    aggregated_width = width.repeat(sample_size, 1)

    B, D = aggregated_center.shape
    unit = int((B - 1) / max_allowed) + 1
    for i in range(unit):
        aggregated_abstract_states = {
            'center': aggregated_center[i * max_allowed: (i+1) * max_allowed, :],
            'width': aggregated_width[i * max_allowed: (i+1) * max_allowed, :],
        }
        aggregated_abstract_states_list.append(aggregated_abstract_states)
    
    # print(len(aggregated_abstract_states_list))
    return aggregated_abstract_states_list



