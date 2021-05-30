import torch
import torch.nn.functional as F
import torch.nn as nn

from gpu_DiffAI.modules import *

import os

index0 = torch.tensor(0)
index1 = torch.tensor(1)
index2 = torch.tensor(2)
index3 = torch.tensor(3)


if torch.cuda.is_available():
    index0 = index0.cuda()
    index1 = index1.cuda()
    index2 = index2.cuda()
    index3 = index3.cuda()


# input order: h, i
def initialization_nn(batched_center, batched_width):
    B, D = batched_center.shape
    padding = torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
    
    input_center, input_width = batched_center[:, :1], batched_width[:, :1]
    # print(input_center.shape, input_width.shape)
    # right = input_center + input_width 
    # input_center = input_center[right > 4.799].unsqueeze(1)
    # input_width = input_width[right > 4.799].unsqueeze(1)
    # print(input_center.shape, input_width.shape)
    # padding = torch.zeros(1, 1)
    # if torch.cuda.is_available():
    #     padding = padding.cuda()

    symbol_tables = {
        'x': domain.Box(torch.cat((input_center, padding), 1), torch.cat((input_width, padding), 1)),
        'trajectory_list': [[] for i in range(B)],
        'idx_list': [i for i in range(B)], # marks which idx the tensor comes from in the input
    }

    return symbol_tables


def initialization_point_nn(x):
    point_symbol_table_list = list()
    symbol_table = {
        'x': domain.Box(var_list([x[0], 0.0]), var_list([0.0] * 2)),
        'probability': var(1.0),
        'trajectory': list(),
        'branch': '',
    }

    point_symbol_table_list.append(symbol_table)

    # to map to the execution of distribution, add one dimension
    return [point_symbol_table_list]

def f_test(x):
    return x


class LinearNN(nn.Module):
    def __init__(self, l=1):
        super().__init__()
        self.linear1 = Linear(in_channels=1, out_channels=1)
    
    def forward(self, x):
        # print(f"input x: {x.c}, {x.delta}")
        # print(x.shape)
        res = self.linear1(x)
        # print(f"res: {res.c}, {res.delta}")
        return res


class LinearNNComplex(nn.Module):
    def __init__(self, l=4):
        super().__init__()
        self.linear1 = Linear(in_channels=1, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
        self.relu = ReLU()

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        return res


def f_assign_h_update(x):
    return x.add(var(0.2))

def f_assign_i_update(x):
    return x.add(var(1.0))

def f_assign_h_increase(x):
    return x.mul(var(2.0)).add(var(1.0))

# input order: 0:h0, 1:bound, 2:count, 3:tmp_h_1, 4:tmp_h_2
class PathExplosion2(nn.Module):
    def __init__(self, l=1, nn_mode="complex"):
        super(PathExplosion2, self).__init__()
        self.goal_iteration = var(50.0)
        self.bar1 = var(3.0)
        self.bar2 = var(5.0)
        self.bar3 = var(2.5)

        # simple version
        if nn_mode == "simple":
            self.nn = LinearNN(l=l)
        # complex version
        if nn_mode == "complex":
            self.nn = LinearNNComplex(l=l)

        self.assign_i_update = Assign(target_idx=[1], arg_idx=[1], f=f_assign_i_update)
        self.assign_h_update = Assign(target_idx=[0], arg_idx=[0], f=f_assign_h_update)

        self.assign_h_update_nn = Assign(target_idx=[0], arg_idx=[0], f=self.nn)
        self.assign_skip = Skip()

        self.ifelse_h_block2 = IfElse(target_idx=[0], test=self.bar2, f_test=f_test, body=self.assign_h_update_nn, orelse=self.assign_skip)
        self.ifelse_h = IfElse(target_idx=[0], test=self.bar1, f_test=f_test, body=self.assign_h_update, orelse=self.ifelse_h_block2)

        self.assign_h_increase = Assign(target_idx=[0], arg_idx=[0], f=f_assign_h_increase)
        self.ifelse_h2 = IfElse(target_idx=[0], test=self.bar3, f_test=f_test, body=self.assign_h_increase, orelse=self.assign_skip)
        self.trajectory_update = Trajectory(target_idx=[0])
        self.whileblock = nn.Sequential(
            self.assign_i_update,
            self.assign_h_update,
            self.ifelse_h,
            self.ifelse_h2,
            self.trajectory_update,
        )
        self.while_i = While(target_idx=[1], test=self.goal_iteration, body=self.whileblock)
        self.program = nn.Sequential(
            self.trajectory_update, 
            self.while_i,
        )
    
    def forward(self, input, version=None):
        if version == "single_nn_learning":
            # B = input.shape[0]
            # count = torch.zeros(B, 1)
            # if torch.cuda.is_available():
            #     count = count.cuda()

            # # print(input.shape)
            # print(f"input: {input.detach().cpu().numpy().tolist()[0]}")
            res = self.nn(input)
            # print(f"res: {res.detach().cpu().numpy().tolist()[0]}")
            # print(input.shape)
            # for i in range(50):
            #     input += 0.2
            #     b1_idx = (input <= self.bar1)
            #     b2_idx = (torch.logical_and(input > self.bar1, input <= self.bar2))
            #     b3_idx = (input > self.bar2)

            #     b1 = input[b1_idx].unsqueeze(1)
            #     b2 = input[b2_idx].unsqueeze(1)
            #     b3 = input[b3_idx].unsqueeze(1)

            #     b1 = b1 + 0.2
            #     b2 = self.nn(b2)
            #     b3 = b3
            #     input = torch.cat((b1, b2, b3), 0)

            # res = input
        else:
            res = self.program(input)
        return res

    def clip_norm(self):
        if not hasattr(self, "weight"):
            return
        if not hasattr(self, "weight_g"):
            if torch.__version__[0] == "0":
                nn.utils.weight_norm(self, dim=None)
            else:
                nn.utils.weight_norm(self)


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








        
