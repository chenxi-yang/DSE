from scipy.stats import truncnorm
from scipy.stats import poisson

import numpy as np
import random
import math

import time
import torch
from torch.autograd import Variable

np.random.seed(seed=1)
random.seed(1)

def var(i, requires_grad=False):
    if torch.cuda.is_available():
        return Variable(torch.tensor(i, dtype=torch.float).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.tensor(i, dtype=torch.float), requires_grad=requires_grad)


def var_list(i_list, requires_grad=False):
    if torch.cuda.is_available():
        res = torch.tensor(i_list, dtype=torch.float, requires_grad=requires_grad).cuda()
    else:
        res = torch.tensor(i_list, dtype=torch.float, requires_grad=requires_grad)
    return res


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


def ini_trajectory(trajectory):
    state = trajectory[0][0]
    action = trajectory[0][1]
    return state, action


def batch_pair(trajectory_list, data_bs=None):
    states, actions = list(), list()
    random.shuffle(trajectory_list)
    for trajectory in trajectory_list:
        for (state, action) in trajectory:
            states.append(state)
            actions.append([action])
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
        last_state, last_action =trajectory[-1]
        states.append(ini_state)
        actions.append([last_action])
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)

    return states[:data_bs], actions[:data_bs]


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


def create_components(x_min, x_max, num_components):
    component_length = (x_max - x_min) / num_components
    components = list()
    for i in range(num_components):
        l = x_min + i * component_length
        r = x_min + (i + 1) * component_length
        center = [(r + l) / 2.0]
        width = [(r - l) / 2.0]
        component_group = {
            'center': center,
            'width': width,
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


def divide_chunks(components, bs=1, data_bs=None):
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
    for i in range(0, len(components), bs):
        components = component_list[i:i + bs]
        abstract_states = dict()
        trajectories = list()
        center_list, width_list = list()
        for component_idx, component in enumerate(components):
            center_list.append(component['center'])
            width_list.append(component['width'])
            for trajectory_idx, trajectory in enumerate(component['trajectories']):
                trajectories.append(trajectory)

        batched_center, batched_width = batch_points(center_list), batch_points(width_list)
        abstract_states['center'] = batched_center
        abstract_states['width'] = batched_width

        yield trajectories, abstract_states






