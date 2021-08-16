import torch
import torch.nn.functional as F
import torch.nn as nn

from random import shuffle
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical

import domain
from constants import *
import constants

from domain_utils import (
    concatenate_states,
)

from utils import (
    select_argmax,
)

import math

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
        n = product(self.weight.size()) / self.out_channels
        stdv = 1 / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()


class Tanh(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.tanh()
    

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
def calculate_states(target_idx, arg_idx, f, states):
    x = states['x']
    # print(f)
    # print(f"x, c: {x.c}, delta: {x.delta}")
    input = x.select_from_index(1, arg_idx)
    res = f(input)
    # print(f"f: {f} \ninput, c: {input.c.detach().cpu().numpy().tolist()} \n, delta: {input.delta.detach().cpu().numpy().tolist()}\n; res, c: {res.c.detach().cpu().numpy().tolist()}\n, delta: {res.delta.detach().cpu().numpy().tolist()}")
    # print(f"before assign: x.c: {x.c}, res.c: {res.c}")
    # print(f"target_idx: {target_idx}")
    x.c[:, target_idx] = res.c 
    # print(f"after assign: x.c: {x.c}")

    x.delta[:, target_idx] = res.delta
    states['x'] = x
    return states


def extract_branch_probability(target, test):
    p_test = torch.zeros(target.getLeft().shape).cuda()
    p_test[target.getRight() <= test] = 1.0
    p_test[target.getLeft() > test] = 0.0
    cross_idx = torch.logical_and(target.getRight() > test, target.getLeft() <= test)

    p_test[cross_idx] = (test - target.getLeft()[cross_idx]) / (target.getRight()[cross_idx] - target.getLeft()[cross_idx])
    # p_test = (test - target.getLeft()) / (target.getRight() - target.getLeft())
    # print(f"p_test grad: {p_test.grad}; leßn: {target.getRight() - target.getLeft()}")
    # print(p_test)
    # p_test[p_test < 0.0] = 0.0
    # p_test[p_test > 1.0] = 1.0
    # print(p_test)

    return p_test, 1 - p_test


def sample_from_p(p_left, p_right):
    m = Bernoulli(p_left)
    res = m.sample()
    left = res > 0
    right = (1 - res) > 0
    return left, right


def calculate_branch(target_idx, test, states):
    body_states, orelse_states = dict(), dict()
    x = states['x']
    target = x.select_from_index(1, target_idx) # select the batch target from x

    # select the idx of left = target.left < test,  right = target.right >= test
    # select the trajectory accordingly
    # select the idx accordingly
    # split the other

    p_left, p_right = extract_branch_probability(target, test)
    left, right = sample_from_p(p_left, p_right)
    if constants.debug:
        print(f"test: {test}")
        print(f"target c: {target.c}, delta: {target.delta}")
        print(f"left probability: {p_left}")
        print(f"right probability: {p_right}")

    if True in left: # split to left
        left_idx = left.nonzero(as_tuple=True)[0].tolist()
        x_left = domain.Box(x.c[left.squeeze(1)], x.delta[left.squeeze(1)])
        left_target_c, left_target_delta = target.c[left].unsqueeze(1), target.delta[left].unsqueeze(1)
        # get the new c, delta
        new_left_target_c = ((left_target_c - left_target_delta) + torch.min((left_target_c + left_target_delta), test)) / 2.0
        new_left_target_delta = (torch.min((left_target_c + left_target_delta), test) - (left_target_c - left_target_delta)) / 2.0
        x_left.c[:, target_idx:target_idx+1] = new_left_target_c
        x_left.delta[:, target_idx:target_idx+1] = new_left_target_delta

        body_states['x'] = x_left
        body_states['trajectories'] = [states['trajectories'][i] for i in left_idx]
        body_states['idx_list'] = [states['idx_list'][i] for i in left_idx]
        if constants.debug:
            print(f"before update: states p_list")
            print(states['p_list'])
        body_states['p_list'] = [states['p_list'][i].add(torch.log(p_left[i])) for i in left_idx]
        if constants.debug:
            print(f"after update: body_states p_list")
            print(body_states['p_list'])
    
    if True in right: # split to right
        right_idx = right.nonzero(as_tuple=True)[0].tolist()
        x_right = domain.Box(x.c[right.squeeze(1)], x.delta[right.squeeze(1)])
        right_target_c, right_target_delta = target.c[right].unsqueeze(1), target.delta[right].unsqueeze(1)

        new_right_target_c = (torch.max((right_target_c - right_target_delta), test) + (right_target_c + right_target_delta)) / 2.0
        new_right_target_delta = ((right_target_c + right_target_delta) - torch.max((right_target_c - right_target_delta), test)) / 2.0
        x_right.c[:, target_idx:target_idx+1] = new_right_target_c
        x_right.delta[:, target_idx:target_idx+1] = new_right_target_delta

        orelse_states['x'] = x_right
        orelse_states['trajectories'] = [states['trajectories'][i] for i in right_idx]
        orelse_states['idx_list'] = [states['idx_list'][i] for i in right_idx]
        if constants.debug:
            print(f"before update: states p_list")
            print(states['p_list'])
        orelse_states['p_list'] = [states['p_list'][i].add(torch.log(p_right[i])) for i in right_idx]
        if constants.debug:
            print(f"after update: orelse_states p_list")
            print(orelse_states['p_list'])
    
    if constants.debug:
        print(f"body_states p_list")
        if len(body_states) > 0:
            print(body_states['p_list'])
        print(f"orelse_states p_list")
        if len(orelse_states) > 0:
            print(orelse_states['p_list'])
    
    return body_states, orelse_states


def extract_branch_probability_list(target, index_mask):
    # volume_based probability assignment
    # return a list of boolean tensor where the k-th boolean tensor represents the states fall into the k-th branch
    zeros = torch.zeros(index_mask.shape)
    branch = torch.zeros(index_mask.shape, dtype=torch.bool)
    if torch.cuda.is_available():
        zeros = zeros.cuda()
        branch = branch.cuda()

    volume = 2 * target.delta
    # all the volumes belonging to the argmax index set are selected, otherwise 0.0
    selected_volume = torch.where(index_mask, volume, zeros)
    sumed_volume = torch.sum(selected_volume, dim=1)[:, None]
    p_volume = selected_volume / sumed_volume

    m = Categorical(p_volume)
    res = m.sample()
    branch[(torch.arange(p_volume.size(0)), res)] = True
    p_volume = torch.where(branch, p_volume, zeros)

    return branch, p_volume


def assign_states(states, branch, p_volume):
    # K batches, M branches, # no change to the x itself
    K, M = branch.shape
    states_list = list()
    x = states['x']

    for i in range(M):
        new_states = dict()
        p = p_volume[:, i]
        this_branch = branch[:, i]
        if True in this_branch:
            this_idx = this_branch.nonzero(as_tuple=True)[0].tolist()
            x_this = domain.Box(x.c[this_branch], x.delta[this_branch])
            new_states['x'] = x_this
            new_states['trajectories'] = [states['trajectories'][i] for i in this_idx]
            new_states['idx_list'] = [states['idx_list'][i] for i in this_idx]
            new_states['p_list'] = [states['p_list'][i].add(torch.log(p[idx])) for idx in this_idx]
        states_list.append(new_states)

    return states_list 


def calculate_branches(arg_idx, states):
    # TODO: finish all the methods
    # TODO Plus: update the trajectories, idx_list, p_list
    # arg_idx: [0, 1, 2, 3], target is a new box only having values with these indexes

    x = states['x']
    target = x.select_from_index(1, arg_idx)
    # TODO: potential problem: tensor is too dense?
    # index_mask is a boolean tensor
    index_mask = select_argmax(target.c - target.delta, target.c + target.delta)
    # no split of boxes/states, only use the volume based probability distribution
    # branch: boolean tensor k-th colume represents the k-th branch, 
    # p_volume: real tensor, k-th column represents the probability to select the k-th branch(after samping)
    branch, p_volume = extract_branch_probability_list(target, index_mask)

    states_list = assign_states(states, branch, p_volume)
    return states_list
        

class Skip(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, states):
        return states


class Assign(nn.Module):
    def __init__(self, target_idx, arg_idx: list(), f):
        super().__init__()
        self.f = f
        self.target_idx = torch.tensor(target_idx)
        self.arg_idx = torch.tensor(arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, states):
        res_states = calculate_states(self.target_idx, self.arg_idx, self.f, states)

        return res_states


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
    
    def forward(self, states):
        body_states, orelse_states = calculate_branch(self.target_idx, self.test, states)
        if len(body_states) > 0:
            body_states = self.body(body_states)
        if len(orelse_states) > 0:
            orelse_states = self.orelse(orelse_states)
        
        # maintain the same number of components as the initial ones
        res_states = concatenate_states(body_states, orelse_states)
        # if constants.debug:
        #     print(f"trajectories of res_states in [IF_ELSE]")
        #     for trajectory in res_states['trajectories']:
        #         print(f"trajectory length: {len(trajectory)}")

        return res_states


class ArgMax(nn.Module):
    def __init__(self, arg_idx, branch_list):
        super().__init__()
        self.arg_idx = torch.tensor(arg_idx)
        self.branch_list = branch_list
        if torch.cuda.is_available():
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, states):
        #TODO
        res_states_list = list()
        states_list = calculate_branches(self.arg_idx, states)

        for idx, state in enumerate(states_list):
            if len(state) > 0:
                res_states_list.append(self.branch_list[idx](state))
        # TODO, concatenate the states in the list form
        res_states = concatenate_states_list(res_states_list)

        return res_states


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.body = body
        if torch.cuda.is_available():
            # print(f"CHECK: cuda")
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, states):
        i = 0
        res_states = dict()
        while(len(states) > 0):
            body_states, orelse_states = calculate_branch(self.target_idx, self.test, states)
            res_states = concatenate_states(res_states, orelse_states)
            if len(body_states) == 0:
                return res_states
            states = self.body(body_states)
            i += 1
            if i > constants.MAXIMUM_ITERATION:
                break
            # if constants.debug:
            #     for trajectory in body_states['trajectories']:
            #         print(f"trajectory length: {len(trajectory)}")
        res_states = concatenate_states(res_states, orelse_states)
        res_states = concatenate_states(res_states, body_states)

        return res_states


class Trajectory(nn.Module):
    # TODO: update, add state in trajectory list
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, states):
        x = states['x']
        trajectories = states['trajectories']
        B, D = x.c.shape
        for x_idx in range(B):
            cur_x_c, cur_x_delta = x.c[x_idx], x.delta[x_idx]
            input_interval_list = list()
            for idx in self.target_idx:
                input = domain.Box(cur_x_c[idx], cur_x_delta[idx])
                input_interval = input.getInterval()

                assert input_interval.left.data.item() <= input_interval.right.data.item()
                input_interval_list.append(input_interval)
            trajectories[x_idx].append(input_interval_list)
        
        states['trajectories'] = trajectories

        return states




            

            









