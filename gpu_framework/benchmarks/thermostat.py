import torch
import torch.nn.functional as F
import torch.nn as nn

from constants import *
import constants
import domain

import os

if constants.status == 'train':
    if mode == 'DSE':
        from gpu_DSE.modules import *
    elif mode == 'only_data':
        # print(f"in only data: import DSE_modules")
        from gpu_DSE.modules import *
    elif mode == 'DiffAI':
        from gpu_DiffAI.modules import *
    elif mode == 'symbol_data_loss_DSE':
        from gpu_symbol_data_loss_DSE.modules import *
    elif mode == 'DiffAI_sps':
        from gpu_DiffAI_sps.modules import *
elif constants.status == 'verify_AI':
    # print(f"in verify_AI: modules_AI")
    from modules_AI import *
elif constants.status == 'verify_SE':
    # print(f"in verify_SE: modules_SE")
    from modules_SE import *

# x_list
# i, isOn, x, lin  = 0.0, 0.0, input, x
# tOff = 62.0
# tOn = 80.0

index0 = torch.tensor(0)
index1 = torch.tensor(1)
index2 = torch.tensor(2)
index3 = torch.tensor(3)

if torch.cuda.is_available():
    index0 = index0.cuda()
    index1 = index1.cuda()
    index2 = index2.cuda()
    index3 = index3.cuda()


def initialize_components(abstract_states):
    #TODO: add batched components to replace the following two 
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
    
    input_center, input_width = center[:, :1], width[:, :1]
    states = {
        'x': domain.Box(torch.cat((padding, padding, input_center, input_center), 1), torch.cat((padding, padding, input_width, input_width), 1)),
        'trajectories': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(1.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states


def f_isOn(x):
    return x[1].setValue(var(1.0))

def f_notisOn(x):
    return x[1].setValue(var(0.0))

def f_up_temp(x):
    # x = x - 0.1*(x-lin) + 5.0
    return x.select_from_index(0, index0).sub_l((x.select_from_index(0, index1).sub_l(x.select_from_index(0, index1))).mul(var(0.1))).add(var(5.0))

def f_test_first(x):
    return x[0]

def f_update_i(x):
    return x + 1

def f_self(x):
    return x


class LinearReLU(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=l)
        self.linear3 = Linear(in_channels=l, out_channels=1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        # self.sigmoid_linear = SigmoidLinear(sig_range=sig_range)

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.relu(res)
        res = self.linear3(res)
        # res = self.sigmoid(res)
        return res


def f_wrap_up_tmp_down_nn(nn):
    def f_tmp_down_nn(x):
        # print(f"nn, before: {x.c, x.delta}")
        plant = nn(x).mul(var(1.0))
        # print(f"nn, after: {plant.c, plant.delta}")
        return x.select_from_index(1, index0).sub_l(plant)
    return f_tmp_down_nn
        

def f_wrap_up_tmp_up_nn(nn):
    def f_tmp_up_nn(x):
        # print(f"nn, before: {x.c, x.delta}")
        plant = nn(x).mul(var(1.0))
        # print(f"nn, after: {plant.c, plant.delta}")
        return x.select_from_index(1, index0).sub_l(plant).add(var(10.0))
    return f_tmp_up_nn


# can not pickle local object
def f_ifelse_tOn_block1(x):
    return x.set_value(var(1.0))

def f_test(x):
    return x

def f_assign2_single(x):
    return f_up_temp(x)

def f_ifelse_tOff_block2(x):
    return x.set_value(var(0.0))

def assign_update(x):
    return x.add(var(1.0))


class Program(nn.Module):
    def __init__(self, l, sig_range=10, nn_mode='all'):
        super(Program, self).__init__()
        self.tOff = var(78.0)
        self.tOn = var(66.0)

        self.nn = LinearReLU(l=l)

        # curL = curL + 10.0 * NN(curL, lin)
        self.assign1 = Assign(target_idx=[2], arg_idx=[2, 3], f=f_wrap_up_tmp_down_nn(self.nn))

        # TODO: empty select index works?
        self.ifelse_tOn_block1 = Assign(target_idx=[1], arg_idx=[], f=f_ifelse_tOn_block1)# f=lambda x: (x.set_value(var(1.0)), var(1.0)))
        self.ifelse_tOn_block2 = Skip()
        self.ifelse_tOn = IfElse(target_idx=[2], test=self.tOn, f_test=f_test, body=self.ifelse_tOn_block1, orelse=self.ifelse_tOn_block2)
        self.ifblock1 = nn.Sequential(
            self.assign1, # DNN
            self.ifelse_tOn, # if x <= tOn: isOn=1.0 else: skip
        )

        self.assign2 = Assign(target_idx=[2], arg_idx=[2, 3], f=f_wrap_up_tmp_up_nn(self.nn))

        self.ifelse_tOff_block1 = Skip()
        self.ifelse_tOff_block2 = Assign(target_idx=[1], arg_idx=[], f=f_ifelse_tOff_block2)# f=lambda x: (x.set_value(var(0.0)), var(1.0)))
        self.ifelse_tOff = IfElse(target_idx=[2], test=self.tOff, f_test=f_test, body=self.ifelse_tOff_block1, orelse=self.ifelse_tOff_block2)

        self.ifblock2 = nn.Sequential(
            self.assign2,
            self.ifelse_tOff,  # if x <= tOff: skip else: isOn=0.0
        )

        self.ifelse_isOn = IfElse(target_idx=[1], test=var(0.5), f_test=f_test, body=self.ifblock1, orelse=self.ifblock2)
        self.assign_update = Assign(target_idx=[0], arg_idx=[0], f=assign_update)
        self.trajectory_update = Trajectory(target_idx=[2])
        self.whileblock = nn.Sequential(
            self.ifelse_isOn,
            self.assign_update,
            self.trajectory_update,
        )
        self.program = While(target_idx=[0], test=var(40.0), body=self.whileblock)
    
    def forward(self, input, version=None):
        # if transition == 'abstract':
        # #     print(f"# of Partitions Before: {len(x_list)}")
        #     for x in x_list:
        #         print(f"x: {x['x'].c}, {x['x'].delta}")
        if version == "single_nn_learning":
            # TODO: add the program version of this benchmark
            # print(input.shape)
            B = input.shape[0]
            isOn = torch.zeros(B, 1)
            lin = input[:, 0].unsqueeze(1)
            x = input[:, 1].unsqueeze(1)
            state = input
            # print(lin.shape, x.shape, isOn.shape)
            trajectory_list = list()
            for i in range(40):
                off_idx = (isOn <= 0.5)
                on_idx = (isOn > 0.5)
                # print(f"off_idx: {off_idx.shape}, x: {x.shape}")
                off_x = x[off_idx].unsqueeze(1)
                # print(f"off_x: {off_x.shape}")
                on_x = x[on_idx].unsqueeze(1)
                off_state = torch.cat((lin[off_idx].unsqueeze(1), off_x), 1)
                on_state = torch.cat((lin[on_idx].unsqueeze(1), on_x), 1)
                isOn_off = isOn[off_idx].unsqueeze(1)
                isOn_on = isOn[on_idx].unsqueeze(1)

                # if isOn <= 0.5: off
                off_x = off_x - self.nn(off_state) * 10.0
                # print(f"off shape: {isOn_off.shape}, {off_x.shape}")
                isOn_off[off_x <= float(self.tOn)] = float(1.0)

                # else  isOn > 0.5: on
                on_x = on_x - self.nn(on_state) * 10.0 + 10.0
                isOn_on[on_x > float(self.tOff)] = float(0.0)

                x = torch.cat((off_x, on_x), 0)
                isOn = torch.cat((isOn_off, isOn_on), 0)
                trajectory_list.append(x)
            
            res = trajectory_list
        else:
            res = self.program(input)
        # if transition == 'abstract':
        #     print(f"# of Partitions After: {len(res_list)}")
        #     # for x in res_list:
        #     #     print(f"x: {x['x'].c}, {x['x'].delta}")
        return res
    
    def clip_norm(self):
        if not hasattr(self, "weight"):
            return
        if not hasattr(self, "weight_g"):
            if torch.__version__[0] == "0":
                nn.utils.weight_norm(self, dim=None)
            else:
                nn.utils.weight_norm(self)



    








        
