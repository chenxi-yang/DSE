import torch
import torch.nn.functional as F
import torch.nn as nn

from modules import *
from helper import * 
import domain

# x_list
# i, isOn, x, lin 
# tOff = 62.0
# tOn = 80.0
def initialization_nn(x, width):
    data_list = list()
    data = domain.Box(var_list([x[0], x[0]]), var_list([width, width]))
    data_list.append(data)
    return data_list


def initialization_point_nn(x):
    data_list = list()
    data = domain.Box(var_list([x[0], x[0]]), var_list([0.0, 0.0]))
    data_list.append(data)
    return data_list


def f_isOn(x):
    return x[1].setValue(var(1.0))

def f_notisOn(x):
    return x[1].setValue(var(0.0))

def f_up_temp(x):
    # x = x - 0.1*(x-lin) + 5.0
    return x[2].sub_l((x[2].sub_l(x[3])).mul(var(0.1))).add(var(5.0))

def f_test_first(x):
    return x[0]


class NN_module(nn.Module):
    def __init__(self, target_idx, l):
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
        self.sigmoid = Sigmoid()

    def forward(self, x_list):
        for idx, x in enumerate(x_list):
            res = self.linear1(x[2:3])
            res = self.sigmoid(res)
            res = self.linear2(res)
            x[target_idx[0]] = res
            x_list[idx] = x
        return x_list


# class ThermostatNN(nn.Module):
#     def __init__(self, l):
#         self.linear1 = Linear(in_channels=2, out_channels=l)
#         self.linear2 = Linear(in_channels=l, out_channels=1)
#         self.sigmoid = Sigmoid()
#         self.NN = self.linear2(self.sigmoid(self.linear1))

#         self.assign1 = Assign(target_idx=[1], f_isOn)
#         self.assign2 = Assign(target_idx=[1], f_notisOn)

#         self.ifelse11 = IfElse(target_idx=[2], test=[var(62.0)], f_test=f_test_first, body=self.assign1, orelse=self.assign2)
#         self.block1 = nn.Sequential(
#             self.NN, 
#             self.ifelse11,
#         )

#         self.assign3 = Assign(target_idx=[2], f_up_temp)
#         self.ifelse21 = IfElse(target_idx=[2], test=[var(80.0)], f_test=f_test_first, body=self.assign1, orelse=self.assign2)
#         self.block2 = nn.Sequential(
#             self.assign3,
#             self.ifelse21,
#         )

#         self.ifelse1 = IfElse(target_idx=[1], test=[var(0.5)], f_test=f_test_first, body=self.block1, orelse=self.block2)
#         self.while1 = While(target_idx=[0], test=[var(1.0)], body=self.ifelse1)
    
#     def forward(self, x_list):
#         res_list = self.while1(x_list)
#         return res_list


class ThermostatNN(nn.Module):
    def __init__(self, l):
        super(ThermostatNN, self).__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
    
    def forward(self, x_list):
        for i in range(40):
            res_list = list()
            for x in x_list:
                res = self.linear1(x)
                res = self.linear2(res)
                res_list.append(res)
        return res_list





        
