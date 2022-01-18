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
min_v = torch.tensor(-10.0)
max_v = torch.tensor(10.0)

if torch.cuda.is_available():
    index0 = index0.cuda()
    index1 = index1.cuda()
    index2 = index2.cuda()
    index3 = index3.cuda()
    min_v = min_v.cuda()
    max_v = max_v.cuda()


# x, y, angle, i, star
def initialize_components(abstract_states):
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    star_padding = torch.zeros(B, 1) + 0.5
    if torch.cuda.is_available():
        padding = padding.cuda()
        star_padding = star_padding.cuda()
    
    input_center, input_width = center[:, :1], width[:, :1]
    states = {
        'x': domain.Box(torch.cat((input_center, padding, padding, padding, star_padding), 1), torch.cat((input_width, padding, padding, padding, star_padding), 1)),
        'trajectories': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states

# input order: x, y, angle, i, star
def initialization_components_point():
    B = 1
    input_center, input_width, padding = torch.zeros(B, 1), torch.zeros(B, 1), torch.zeros(B, 1)
    star_padding = torch.zeros(B, 1) + 0.5
    if torch.cuda.is_available():
        padding = padding.cuda()
        star_padding = star_padding.cuda()
        input_center = input_center.cuda()
        input_width = input_width.cuda()
    
    input_center[0], input_width[0] = 1.0, 0.0
    states = {
        'x': domain.Box(torch.cat((input_center, padding, padding, padding, star_padding), 1), torch.cat((input_width, padding, padding, padding, star_padding), 1)),
        'trajectories': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states

def f_test(x):
    return x


class LinearNN(nn.Module):
    def __init__(self, l=1):
        super().__init__()
        self.linear1 = Linear(in_channels=3, out_channels=1)
    
    def forward(self, x):
        res = self.linear1(x)
        return res


class LinearNNComplex(nn.Module):
    def __init__(self, l=4):
        super().__init__()
        self.linear1 = Linear(in_channels=3, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.sigmoid(res)
        return res


def f_move_down(x):
    return x.sub_l(var(1.0))
def f_move_right(x):
    return x
def f_move_up(x):
    return x.add(var(1.0))

def f_angle_down(x):
    return x.set_value(var(1))
def f_angle_up(x):
    return x.set_value(var(0))
def f_angle_straight(x):
    return x.set_value(var(0.5))

def f_forward(x): # y += 1
    return x.add(var(1.0))
def f_step_update(x):
    return x.add(var(1.0))

# input order: x, y, angle, i, star
class Program(nn.Module):
    def __init__(self, l=1, nn_mode="simple"):
        super(Program, self).__init__()
        self.step = var(19)
        self.down_bar = var(0.4)
        self.up_bar = var(0.6)
        self.star_bar = var(0.5)
        self.control_straight_bar = var(9)
        self.control_down_bar = var(12)

        if nn_mode == "simple":
            self.nn_straight = LinearNN(l=l)
            self.nn_up = LinearNN(l=l)
            self.nn_down = LinearNN(l=l)
        if nn_mode == "complex":
            self.nn_straight = LinearNNComplex(l=l)
            self.nn_up = LinearNNComplex(l=l)
            self.nn_down = LinearNNComplex(l=l)

        self.move_down = Assign(target_idx=[0], arg_idx=[0, 1, 2], f=self.nn_down)
        self.move_right = Assign(target_idx=[0], arg_idx=[0, 1, 2], f=self.nn_straight)
        self.move_up = Assign(target_idx=[0], arg_idx=[0, 1, 2], f=self.nn_up)

        self.assign_angle_down = Assign(target_idx=[2], arg_idx=[2], f=f_angle_down) # update the angle
        self.assign_angle_up = Assign(target_idx=[2], arg_idx=[2], f=f_angle_up)
        self.assign_angle_straight = Assign(target_idx=[2], arg_idx=[2], f=f_angle_straight)
        self.ifelse_car_control_2= IfElse(target_idx=[1], test=self.control_down_bar, f_test=f_test, body=self.assign_angle_down, orelse=self.assign_angle_up)
        self.ifelse_car_control_1 = IfElse(target_idx=[1], test=self.control_straight_bar, f_test=f_test, body=self.assign_angle_straight, orelse=self.ifelse_car_control_2)
        
        self.star_block = IfElse(target_idx=[4], test=self.star_bar, f_test=f_test, body=self.move_up, orelse=self.move_down)
        self.ifelse_angle_2 = IfElse(target_idx=[2], test=self.down_bar, f_test=f_test, body=self.star_block, orelse=self.move_right)
        self.ifelse_angle = IfElse(target_idx=[2], test=self.up_bar, f_test=f_test, body=self.ifelse_angle_2, orelse=self.star_block)

        self.forward_update = Assign(target_idx=[1], arg_idx=[1], f=f_forward)
        self.step_update = Assign(target_idx=[3], arg_idx=[3], f=f_step_update)
        self.trajectory_update = Trajectory(target_idx=[0, 1, 2]) # x, y, angle

        self.whileblock = nn.Sequential(
            self.ifelse_car_control_1, # x, y -> angle
            self.ifelse_angle, # update x, y according to angle
            self.forward_update,  # update y: y += 1
            self.step_update, # update step i
            self.trajectory_update, 
        )
        self.program = While(target_idx=[3], test=self.step, body=self.whileblock)
    
    def forward(self, input, version=None):
        if version == "single_nn_learning":
            # model the car-controller
            B, D = input.shape
            res = torch.zeros(B, 1) # placeholder for res
            if torch.cuda.is_available():
                res = res.cuda()
            straight_index = input[:, 1] == 0
            up_index = input[:, 1] == 1
            down_index = input[:, 1] == 2
            res[straight_index] = self.nn_straight(input[straight_index][:, :-1])
            res[up_index] = self.nn_up(input[up_index][:, :-1])
            res[down_index] = self.nn_down(input[down_index][:, :-1])
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









        
