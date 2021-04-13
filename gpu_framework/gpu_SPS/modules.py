import torch
import torch.nn.functional as F
import torch.nn as nn

from random import shuffle

from constants import PARTIAL_BETA as beta

import math
import time


'''
Module used as functions
'''

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        if not hasattr(self,'weight') or self.weight is None:
            return
        # print(f"weight size: {self.weight.size()}")
        # print(f"product: {product(self.weight.size())}")

        n = product(self.weight.size()) / self.out_channels
        stdv = 1 / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # print(f"weight: \n {self.weight}")
        # print(f"bias: \n {self.bias}")
        return x.matmul(self.weight).add(self.bias)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()
    

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.relu()


class SigmoidLinear(nn.Module):
    def __init__(self, sig_range):
        super().__init__()
        self.sig_range = sig_range
    
    def forward(self, x):
        return x.sigmoid_linear(sig_range=self.sig_range)

'''
Program Statement
'''

def calculate_abstract_state(target_idx, arg_idx, f, abstract_state):
    # assign_time = time.time()
    for idx, symbol_table in enumerate(abstract_state):
        x = symbol_table['x']
        input = x.select_from_index(0, arg_idx) # torch.index_select(x, 0, arg_idx)
        # print(f"f: {f}")
        res, p = f(input)
        # print(f"calculate_x_list --  target_idx: {target_idx}, res: {res.c}, {res.delta}")
        x.set_from_index(target_idx, res) # x[target_idx[0]] = res
        
        symbol_table['x'] = x
        #! probability of each component does not change
        # symbol_table['probability'] = symbol_table['probability'].mul(p)
        abstract_state[idx] = symbol_table
    # print(f"-- assign -- calculate_x_list: {time.time() - assign_time}")
    return abstract_state


def calculate_abstract_state_list(target_idx, arg_idx, f, abstract_state_list):
    res_list = list()
    for abstract_state in abstract_state_list:
        res_abstract_state = calculate_abstract_state(target_idx, arg_idx, f, abstract_state)
        res_list.append(res_abstract_state)
    return res_list


def pre_build_symbol_table(symbol_table):
    # clone trajectory
    res_symbol_table = dict()
    res_symbol_table['trajectory'] = list()
    for state in symbol_table['trajectory']:
        res_symbol_table['trajectory'].append(state)

    return res_symbol_table


def pre_allocate(symbol_table):
    return symbol_table['probability']


def f_beta():
    return beta


def split_volume(symbol_table, target, delta):
    # 
    target_volume = target.getRight() - target.getLeft()
    new_volume = delta.mul(var(2.0))
    alpha = symbol_table['alpha'].mul(torch.min(var(1.0), new_volume.div(target_volume.mul(f_beta()))))

    return alpha


def update_res_in_branch(res_symbol_table, res, probability, branch, alpha):
    res_symbol_table['x'] = res
    res_symbol_table['probability'] = probability
    res_symbol_table['branch'] = branch
    res_symbol_table['alpha'] = alpha

    return res_symbol_table


def split_branch_symbol_table(target_idx, test, symbol_table):
    body_symbol_table, orelse_symbol_table = dict(), dict()
    branch_time = time.time()

    x = symbol_table['x']
    target = x.select_from_index(0, target_idx)
    # res_symbol_table = pre_build_symbol_table(symbol_table)

    if target.getRight().data.item() <= test.data.item():
        res = x.clone()
        branch = 'body'
        body_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table)
        alpha = symbol_table['alpha'] * (1.0 / f_beta())
        body_symbol_table = update_res_in_branch(body_symbol_table, res, probability, branch, alpha)

    elif target.getLeft().data.item() > test.data.item():
        res = x.clone()
        branch = 'orelse'
        orelse_symbol_table = pre_build_symbol_table(symbol_table)
        probability = pre_allocate(symbol_table)
        alpha = symbol_table['alpha'] * (1.0 / f_beta())
        orelse_symbol_table = update_res_in_branch(orelse_symbol_table, res, probability, branch, alpha)

    else:
        res = x.clone()
        branch = 'body'
        c = (target.getLeft() + test) / 2.0
        delta = (test - target.getLeft()) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta)) # res[target_idx] = Box(c, delta)
        body_symbol_table = pre_build_symbol_table(symbol_table)
        # in SPS, alpha is changed by volume
        probability = pre_allocate(symbol_table)
        alpha = split_volume(symbol_table, target, delta)
        body_symbol_table = update_res_in_branch(body_symbol_table, res, probability, branch, alpha)

        res = x.clone()
        branch = 'orelse'
        c = (target.getRight() + test) / 2.0
        delta = (target.getRight() - test) / 2.0
        res.set_from_index(target_idx, domain.Box(c, delta))
        orelse_symbol_table = pre_build_symbol_table(symbol_table)
        # in SPS, alpha is changed by volume
        probability = pre_allocate(symbol_table)
        alpha = split_volume(symbol_table, target, delta)
        orelse_symbol_table = update_res_in_branch(orelse_symbol_table, res, probability, branch, alpha)

    # print(f"branch time: {time.time() - branch_time}")
    return body_symbol_table, orelse_symbol_table


def split_branch_abstract_state(target_idx, test, abstract_state):
    # because there is join here, allow empty
    body_abstract_state, orelse_abstract_state = list(), list()
    for symbol_table in abstract_state:
        body_symbol_table, orelse_symbol_table = split_branch_symbol_table(target_idx, test, symbol_table)
        body_abstract_state.append(body_symbol_table)
        orelse_abstract_state.append(orelse_symbol_table)
    return body_abstract_state, orelse_abstract_state
            

def split_branch_list(target_idx, test, abstract_state_list):
    # because there is join here, allow empty
    body_abstract_state_list = list(), else_abstract_state_list = list()
    assert(len(abstract_state_list) == 1) # only before split, because there is smooth join
    for abstract_state in abstract_state_list: 
        body_abstract_state, orelse_abstract_state = split_branch_abstract_state(target_idx, test, abstract_state)
        body_abstract_state_list.append(body_abstract_state)
        else_abstract_state_list.append(orelse_abstract_state)
    return body_abstract_state_list, else_abstract_state_list


def sound_join_trajectory(trajectory_1, trajectory_2):
    l1, l2 = len(trajectory_1), len(trajectory_2)
    trajectory = list()
    for idx in range(min(l1 - 1, l2 - 1)):
        states_1, states_2 =  trajectory_1[idx], trajectory_2[idx]
        state_list = list()
        for state_idx, v in enumerate(states_1):
            state_1, state_2 = states_1[state_idx], states_2[state_idx]
            state_list.append(state_1.soundJoin(state_2))
        trajectory.append(state_list)
    if l1 < l2:
        trajectory.extend(trajectory_2[l1:])
    elif l1 > l2:
        trajectory.extend(trajectory_1[l2:])
    
    return trajectory


def smooth_join_symbol_table(symbol_table_list):
    alpha_out = var(0.0)
    alpha_max = var(0.0)
    c_out = var_list([0.0])
    assert(len(symbol_table_list) > 1)
    for symbol_table in symbol_table_list:
        if len(symbol_table) == 0:
            continue
        alpha_max = torch.max(symbol_table['alpha'], alpha_max)
        alpha_out.add(symbol_table['alpha'])
        c_out.add(symbol_table['alpha'] * symbol_table['x'].c)
    c_out = c_out / alpha_out
    alpha_out = torch.min(alpha_out, var(1.0))

    # adjust domain
    joined_symbol_table = {
        'x': None,
        'alpha': alpha_out,
        'branch': '',
        'probability': var(0.0), 
        'trajectory': list(),
    }
    for idx, symbol_table in enumerate(symbol_table_list):
        alpha_prime = symbol_table['alpha'] / alpha_max
        symbol_table['x'].delta = alpha_prime * symbol_table['x'].delta
        symbol_table['x'].c = alpha_prime * symbol_table['x'].c + (1 - alpha_prime) * c_out
        if joined_symbol_table['x'] is None:
            joined_symbol_table['x'] = symbol_table['x']
        else:
            joined_symbol_table['x'] = joined_symbol_table['x'].sound_join(symbol_table['x'])
        
        # longer trajectory is taken
        # if len(symbol_table['trajectory']) > joined_symbol_table['trajectory']:
        #     joined_symbol_table['trajectory'] = [state for state in symbol_table['trajectory']]
        #TODO: smooth join of trajectories
        joined_symbol_table['trajectory'] = sound_join_trajectory(joined_symbol_table['trajectory'], symbol_table['trajectory'])
        joined_symbol_table['probability'] = torch.max(joined_symbol_table['probability'], symbol_table['probability'])

    return joined_symbol_table


def smooth_join(abstract_state_list_1, abstract_state_list_2):
    assert(len(abstract_state_list_1) == 1 and len(abstract_state_list_2) == 1)
    to_join_abstract_state_list = list()
    to_join_abstract_state_list.extend(abstract_state_list_1)
    to_join_abstract_state_list.extend(abstract_state_list_2)
    
    res_abstract_state = list()
    length_symbol_tables = len(to_join_abstract_state_list[0])
    for i in range(length_symbol_table):
        to_join_symbol_table_list = list()
        for abstract_state in to_join_abstract_state_list:
            to_join_symbol_table_list.append(abstract_state[i])
        joined_symbol_table = smooth_join_symbol_table(to_join_symbol_table_list)
        res_abstract_state.append(joined_symbol_table)
    return res_abstract_state


class Skip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        return abstract_state_list


class Assign(nn.Module):
    def __init__(self, target_idx, arg_idx: list(), f):
        super().__init__()
        self.f = f
        self.target_idx = torch.tensor(target_idx)
        self.arg_idx = torch.tensor(arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        # print(f"Assign Before: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        res_list = calculate_abstract_state_list(self.target_idx, self.arg_idx, self.f, abstract_state_list)
        # print(f"Assign After: {[(res['x'].c, res['x'].delta) for res in x_list]}")
        return res_list


class IfElse(nn.Module):
    def __init__(self, target_idx, test, f_test, body, orelse):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.f_test = f_test
        self.body = body
        self.orelse = orelse
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, abstract_state_list):
        # print(f"Ifelse: target_idx: {self.target_idx}")
        # print(f"############one ifelse ##################")

        test = self.f_test(self.test)
        body_list, else_list = split_branch_list(self.target_idx, self.test, abstract_state_list)
        if len(body_list) > 0:
            body_list = self.body(body_list)
        if len(else_list) > 0:
            else_list = self.orelse(else_list)
        res_list = smooth_join(body_list, else_list)

        return res_list


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.body = body
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, abstract_state_list):
        i = 0
        res_list = list()
        while(len(abstract_state_list) > 0):
            body_list, else_list = split_branch_list(self.target_idx, self.test, abstract_state_list)
            assert(len(body_list) == 1 and len(else_list) == 1)

            res_list = smooth_join(res_list, else_list)
            if len(body_list) > 0:
                abstract_state_list = self.body(body_list)
            else:
                return res_list
            
            i += 1
            if i > 1000:
                print(f"Exceed maximum iterations: Have to END.")
                break

        return res_list


def update_trajectory(symbol_table, target_idx):
    # trajectory: list of states
    # states: list of intervals
    input_interval_list = list()
    # print(f"all symbol_table: {symbol_table['x'].c, symbol_table['x'].delta}")
    for idx in target_idx:
        input = symbol_table['x'].select_from_index(0, idx)
        input_interval = input.getInterval()
        # print(f"idx:{idx}, input: {input.c, input.delta}")
        # print(f"input_interval: {input_interval.left.data.item(), input_interval.right.data.item()}")
        assert input_interval.left.data.item() <= input_interval.right.data.item()
        input_interval_list.append(input_interval)
    # print(f"In update trajectory")
    symbol_table['trajectory'].append(input_interval_list)
    # exit(0)
    # print(f"Finish update trajectory")

    return symbol_table


def update_abstract_state_trajectory(abstract_state, target_idx):
    for idx, symbol_table in enumerate(abstract_state):
        symbol_table = update_trajectory(symbol_table, target_idx)
        abstract_state[idx] = symbol_table
    return abstract_state


class Trajectory(nn.Module):
    # TODO: update, add state in trajectory list
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, abstract_state_list, cur_sample_size=0):
        for idx, abstract_state in enumerate(abstract_state_list):
            abstract_state = update_abstract_state_trajectory(abstract_state, self.target_idx)
            abstract_state_list[idx] = abstract_state
        return abstract_state_list




            

            









