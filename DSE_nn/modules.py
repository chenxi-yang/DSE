import torch
import torch.nn.functional as F
import torch.nn as nn


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

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()


'''
Program Statement
'''

def calculate_x_list(target_idx, arg_idx, f, x_list):
    res_list = list()
    for x in x_list:
        input = x.select_from_index(0, arg_idx) # torch.index_select(x, 0, arg_idx)
        res = f(input)
        x.set_from_index(target_idx, res) # x[target_idx[0]] = res
        res_list.append(x)
    return res_list


def calculate_branch(target_idx, test, x, b):
    # print(f"calculate branch -- target_idx: {target_idx}")
    # print(f"x: {x.c, x.delta}")
    target = x.select_from_index(0, target_idx) # target = x[target_idx]
    # print(f"target right: {target.getRight()}")

    if b == '<':
        if target.getRight().data.item() <= test.data.item():
            res = x.clone()
        elif target.getLeft.data.item() > test.data.item():
            res = None
        else:
            res = x.clone()
            c = (target.getLeft() + test) / 2.0
            delta = (test - target.getLeft()) / 2.0
            res.set_from_index(target_idx, Box(c, delta)) # res[target_idx] = Box(c, delta)
    else:
        if target.getRight().data.item() <= test.data.item():
            res = None
        elif target.getLeft.data.item() > test.data.item():
            res = x.clone()
        else:
            res = x.clone()
            c = (target.getRight() + test) / 2.0
            delta = (target.getRight() - test) / 2.0
            res[target_idx] = Box(c, delta)
    
    return res
            

def calculate_branch_list(target_idx, test, x_list, b):
    res_list = list()
    for x in x_list: # for each element, split it. # c, delta
        res = calculate_branch(target_idx, test, x, b)
        if res is None:
            continue
        res_list.append(res)
    return res_list


class Assign(nn.Module):
    def __init__(self, target_idx, arg_idx: list(), f):
        super().__init__()
        self.f = f
        self.target_idx = torch.tensor(target_idx)
        self.arg_idx = torch.tensor(arg_idx)
        if torch.cuda.is_available():
            self.target_idx = self.target_idx.cuda()
            self.arg_idx = self.arg_idx.cuda()
    
    def forward(self, x_list):
        res_list = calculate_x_list(self.target_idx, self.arg_idx, self.f, x_list)
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
    
    def forward(self, x_list):
        res_list = list()
        body_list = calculate_branch_list(self.target_idx, self.f_test(self.test), x_list, '<')
        orelse = calculate_branch_list(self.target_idx, self.f_test(self.test), x_list, '>')

        body_list = self.body(body_list)
        res_list.extend(body_list)

        orelse_list = self.orelse(orelse_list)
        res_list.extend(orelse_list)

        return res_list


class While(nn.Module):
    def __init__(self, target_idx, test, body):
        super().__init__()
        self.target_idx = torch.tensor(target_idx)
        self.test = test
        self.body = body
        if torch.cuda.is_available():
            # print(f"CHECK: cuda")
            self.target_idx = self.target_idx.cuda()
    
    def forward(self, x_list):
        res_list = list()

        while(len(x_list) > 0):
            body_list = calculate_branch_list(self.target_idx, self.test, x_list, '<')
            orelse_list= calculate_branch_list(self.target_idx, self.test, x_list, '>')

            res_list.extend(orelse_list)

            if len(body_list) > 0:
                x_list = self.body(body_list)
        
        return x_list
            
            

            

            









