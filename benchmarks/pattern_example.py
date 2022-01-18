import torch
import torch.nn as nn

import constants
from constants import *
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

index0 = torch.tensor(0)
index1 = torch.tensor(1)
index2 = torch.tensor(2)
index3 = torch.tensor(3)
min_v = torch.tensor(5.0)
max_v = torch.tensor(10.0)

if torch.cuda.is_available():
    index0 = index0.cuda()
    index1 = index1.cuda()
    index2 = index2.cuda()
    index3 = index3.cuda()
    min_v = min_v.cuda()
    max_v = max_v.cuda()


# input order: x, y, z, i
def initialize_components(abstract_states):
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
    
    input_center, input_width = center[:, :1], width[:, :1]
    states = {
        'x': domain.Box(torch.cat((input_center, padding, padding, padding, padding), 1), torch.cat((input_width, padding, padding, padding, padding), 1)),
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states

# input order: x, y, z, i
def initialization_components_point():
    B = 1
    input_center, input_width, padding = torch.zeros(B, 1), torch.zeros(B, 1), torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
        input_center = input_center.cuda()
        input_width = input_width.cuda()
    
    input_center[0], input_width[0] = 1.0, 0.0
    states = {
        'x': domain.Box(torch.cat((input_center, padding, padding, padding, padding), 1), torch.cat((input_width, padding, padding, padding, padding), 1)),
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states

def f_test(x):
    return x

def f_update_i(x):
    return x.add(var(1))


class LinearNN(nn.Module):
    def __init__(self, l=1):
        super().__init__()
        self.linear1 = Linear(in_channels=1, out_channels=1)
    
    def forward(self, x):
        res = self.linear1(x)
        return res


class LinearNNComplex(nn.Module):
    def __init__(self, l=4):
        super().__init__()
        self.linear1 = Linear(in_channels=1, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=l)
        self.linear3 = Linear(in_channels=l, out_channels=l)
        self.linear4 = Linear(in_channels=l, out_channels=1)
        self.relu = ReLU()

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.relu(res)
        res = self.linear3(res)
        res = self.relu(res)
        res = self.linear4(res)

        return res

def f_assign_min_z(x):
    return x.select_from_index(1, index0).sub_l(var(5.0))
    # return x.select_from_index(1, index0)

def f_assign_max_z(x):
    return x.select_from_index(1, index0).add(var(10.0))# .add(x.select_from_index(1, index1)) # mul(var(-1))

# input order: x, y, z, i, acc
class Program(nn.Module):
    def __init__(self, l=1, nn_mode="simple"):
        super(Program, self).__init__()
        self.bar = var(1.0)
        self.steps = var(0) # here the loop condition is <= N, therefore, steps==1 -> 2 steps executed
        if nn_mode == "simple":
            self.nn = LinearNN(l=l)
        if nn_mode == "complex":
            self.nn = LinearNNComplex(l=l)

        self.assign_y = Assign(target_idx=[1], arg_idx=[0], f=self.nn)

        self.assign_min_z = Assign(target_idx=[2], arg_idx=[0, 4], f=f_assign_min_z)
        self.assign_max_z = Assign(target_idx=[2], arg_idx=[0, 4], f=f_assign_max_z)
        self.ifelse_z = IfElse(target_idx=[1], test=self.bar, f_test=f_test, body=self.assign_max_z, orelse=self.assign_min_z)
        self.assign_update_i = Assign(target_idx=[3], arg_idx=[3], f=f_update_i)

        self.trajectory_update = Trajectory(target_idx=[2])
        self.whileblock = nn.Sequential(
            self.assign_y,
            self.ifelse_z,
            self.assign_update_i,
            self.trajectory_update,
        )
        self.program = While(target_idx=[3], test=self.steps, body=self.whileblock)

    def forward(self, input, version=None):
        if version == "single_nn_learning":
            # print(input.shape)
            y = self.nn(input)
            # print(f"y: {y.detach().cpu().numpy().tolist()[:3]}")
            # x = torch.clone(y)
            # x[y <= float(self.bar)] = x[y <= float(self.bar)] + float(max_v)
            # x[y > float(self.bar)] = x[y > float(self.bar)] - float(min_v)
            res = y
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








        
