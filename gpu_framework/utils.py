from scipy.stats import truncnorm
from scipy.stats import poisson

import numpy as np
import random

import time
import torch



np.random.seed(seed=1)
random.seed(1)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )


def generate_distribution(x, l, r, distribution, unit):
    x_list = list()
    if distribution == "uniform":
        x_list = np.random.uniform(l, r, unit).tolist()
    if distribution == "normal":
        X = get_truncated_normal(x, sd=1, low=l, upp=r)
        x_list = X.rvs(unit).tolist()
    if distribution == "beta":
        l_list = np.random.beta(2, 5, int(unit/2))
        r_list = np.random.beta(2, 5, int(unit/2))
        for v in l_list:
            x_list.append(x - v)
        for v in r_list:
            x_list.append(x + v)  
    if distribution == "original":
        x_list = [x] * unit
    
    return x_list


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
    # print(states.shape, actions.shape)
    # print(f"after shuffle: {states[0], actions[0]}")
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


##### create symbolic approximation of perturbation set of input distribution

def create_ball_perturbation(Trajectory_train, distribution_list, w):
    perturbation_x_dict = {
        distribution: list() for distribution in distribution_list
    }
    for trajectory in Trajectory_train:
        state, _ = ini_trajectory(trajectory)
        # TODO: for now, only use the first input variable
        x = state[0]
        l, r = x - w, x + w
        # print(f"l, r: {l, r}")
        for distribution in distribution_list:
            x_list = generate_distribution(x, l, r, distribution, unit=6)
            # print(f"x_list of {distribution}: {x_list}")
            perturbation_x_dict[distribution].extend(x_list)
        # exit(0)
    return perturbation_x_dict


def split_component(perturbation_x_dict, x_l, x_r, num_components):
    # TODO: add vector-wise component split
    x_min, x_max = x_l[0], x_r[0]
    for distribution in perturbation_x_dict:
        x_min, x_max = min(min(perturbation_x_dict[distribution]), x_min), max(max(perturbation_x_dict[distribution]), x_max)
    component_length = (x_max - x_min) / num_components
    component_list = list()
    for i in range(num_components):
        l = x_min + i * component_length
        r = x_min + (i + 1) * component_length
        center = [(l + r) /  2.0]
        width = [(r - l) / 2.0]
        component_group = {
            'center': center,
            'width': width,
        }
        component_list.append(component_group)
    return component_list


def extract_upper_probability_per_component(component, perturbation_x_dict):

    p_list = list()
    for distribution in perturbation_x_dict:
        x_list = perturbation_x_dict[distribution]
        cnt = 0
        for X in x_list:
            x = [X] #TODO: X is a value
            if in_component(x, component):
                cnt += 1
        p = cnt * 1.0 / (len(x_list) + 1e-100) #  + random.uniform(0, 0.1)
        p_list.append(p)
    return max(p_list)


def assign_probability(perturbation_x_dict, component_list):
    '''
    perturbation_x_dict = {
        distribution: x_list, # x in x_list are in the form of value
    }
    component_list:
    component: {
        'center': center # center is vector
        'width': width # width is vector
    }
    keep track of under each distribution, what portiton of x_list fall 
    in to this component
    '''
    for idx, component in enumerate(component_list):
        p = extract_upper_probability_per_component(component, perturbation_x_dict)
        # print(f"p: {p}")
        component['p'] = p
        component_list[idx] = component
    # print(f"sum of upper bound: {sum([component['p'] for component in component_list])}")
    # exit(0)
    return component_list


def in_component(X, component):
    # TODO: 
    center = component['center']
    width = component['width']
    for i, x in enumerate(X):
        if x >= center[i] - width[i] and x < center[i] + width[i]:
            pass
        else:
            return False
    return True


def assign_data_point(Trajectory_train, component_list):
    for idx, component in enumerate(component_list):
        component.update(
            {
            'trajectory_list': list(),
            }
        )
        for i, trajectory in enumerate(Trajectory_train):
            state, action = ini_trajectory(trajectory) # get the initial <state, action> pair in trajectory
            # when test, only test the first value in state
            if in_component([state[0]], component): # if the initial state in component
                component['trajectory_list'].append(trajectory)
        component_list[idx] = component
    return component_list
        

def extract_abstract_representation(
    Trajectory_train, 
    x_l, 
    x_r, 
    num_components, 
    w=0.3):
    # bs < num_components, w is half of the ball width
    '''
    Steps:
    # 1. generate perturbation, small ball covering following normal, uniform, poission
    # 2. measure probability 
    # 3. slice X_train, y_train into component-wise
    '''
    # TODO: generate small ball based on init(trajectory), others remain
    start_t = time.time()
    # print(f"w: {w}")

    perturbation_x_dict = create_ball_perturbation(Trajectory_train, 
        # distribution_list=["normal", "uniform", "beta", "original"], 
        distribution_list=["normal", "uniform", "original"],  
        #TODO: beta distribution does not account for range
        w=w)
    component_list = split_component(perturbation_x_dict, x_l, x_r, num_components)
    # print(f"after split components: {component_list}")

    # create data for batching, each containing component and cooresponding x, y
    component_list = assign_probability(perturbation_x_dict, component_list)
    component_list = assign_data_point(Trajectory_train, component_list)
    random.shuffle(component_list)

    print(f"component-wise x length: {[len(component['trajectory_list']) for component in component_list]}")

    # print(component_list)
    print(f"-- Generate Perturbation Set --")
    print(f"--- {time.time() - start_t} sec ---")

    return component_list
    

def show_cuda_memory(name):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    print(f"{name}, free mem: {f}, reserved mem: {r}, allocated mem: {a}")


def show_memory_snapshot():
    snapshot = torch.cuda.memory_snapshot()
    for d in snapshot:
        print(d["segment_type"], d["active_size"], d["allocated_size"], d["total_size"])






