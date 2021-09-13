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


# x1, y1, x2, y2, p00, p01, p02, p10, p11, p12, i, a1, b1, c1, a2, b2, c2, d
def initialize_components(abstract_states):
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
    
    input_center, input_width = center[:, :1], width[:, :1]
    states = {
        'x': domain.Box(torch.cat((input_center, padding, input_center, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding), 1), \
             torch.cat((input_width, padding, input_width, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding), 1)),
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states

# input order: x, y, z
def initialization_components_point(x_l=None, x_r=None):
    B = 4
    input_center = torch.rand(B, 1) * (x_r[0] - x_l[0]) + x_l[0]
    input_center[0] = x_r[0]
    input_center[1] = x_l[0]

    input_width, padding = torch.zeros(B, 1), torch.zeros(B, 1)
    
    if torch.cuda.is_available():
        padding = padding.cuda()
        input_center = input_center.cuda()
        input_width = input_width.cuda()
    
    # input_center[0], input_width[0] = 4.1, 0.0
    # input_center[3], input_width[3] = 4.0005, 0.0005 
    # input_center[1], input_width[1] = 5.0, 0.0
    # input_center[2], input_width[2] = 5.0, 0.0005
    # input_center[4], input_width[4] = 4.001, 0.0

    states = {
        'x': domain.Box(torch.cat((input_center, padding, input_center, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding), 1), \
            torch.cat((input_width, padding, input_width, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding, padding), 1)),
        # 'trajectories': [[] for i in range(B)],
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
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
        self.linear1 = Linear(in_channels=2, out_channels=3)
    
    def forward(self, x):
        res = self.linear1(x)
        return res


class LinearNNComplex(nn.Module):
    def __init__(self, l=4):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=l)
        self.linear3 = Linear(in_channels=l, out_channels=3)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.relu(res)
        res = self.linear3(res)
        # res = self.sigmoid(res) # to increase the volume
        return res


def f_move_down(x): #  x -= 1
    return x.sub_l(var(1.0))
def f_move_right(x):
    return x
def f_move_up(x): # x += 1
    return x.add(var(1.0))

def f_forward(x): # y += 1
    return x.add(var(1.0))
def f_step_update(x):
    return x.add(var(1.0))

def f_assign_a(x):
    p0 = x.select_from_index(1, index0)
    p1 = x.select_from_index(1, index1)
    return p1.sub_l(p0)
def f_assign_b(x):
    p0 = x.select_from_index(1, index0)
    p2 = x.select_from_index(1, index1)
    return p2.sub_l(p0)
def f_assign_c(x):
    p1 = x.select_from_index(1, index0)
    p2 = x.select_from_index(1, index1)
    return p2.sub_l(p1)
def f_assign_distance(x):
    p0 = x.select_from_index(1, index0)
    p1 = x.select_from_index(1, index1)
    res = p0.sub_l(p1)
    # return p0.sub_l(p1)
    # print(f"p0: {p0.c.detach().cpu().numpy().tolist(), p0.delta.detach().cpu().numpy().tolist()}")
    # print(f"p1: {p1.c.detach().cpu().numpy().tolist(), p1.delta.detach().cpu().numpy().tolist()}")
    # print(f"res: {res.c.detach().cpu().numpy().tolist(), res.delta.detach().cpu().numpy().tolist()}")
    return res

# x y p0, p1, p2
# x1, y1, x2, y2, p00, p01, p02, p10, p11, p12, i, a1, b1, c1, a2, b2, c2, d, d_me
# 0,   1,  2,  3,  4,    5,   6,   7,   8,   9,10, 11, 12, 13, 14, 15, 16, 17, 
class Program(nn.Module):
    def __init__(self, l=1, nn_mode="simple"):
        super(Program, self).__init__()
        self.step = var(19)
        self.comparison_bar = var(0.0)

        self.agent1 = LinearNNComplex(l=l)
        self.agent2 = LinearNNComplex(l=l)
        # the update of agent1
        self.move_down_1 = Assign(target_idx=[0], arg_idx=[0], f=f_move_down)
        self.move_right_1 = Assign(target_idx=[0], arg_idx=[0], f=f_move_right)
        self.move_up_1 = Assign(target_idx=[0], arg_idx=[0], f=f_move_up)

        self.assign_a_1 = Assign(target_idx=[11], arg_idx=[4, 5], f=f_assign_a)
        self.assign_b_1 = Assign(target_idx=[12], arg_idx=[4, 6], f=f_assign_b)
        self.assign_c_1 = Assign(target_idx=[13], arg_idx=[5, 6], f=f_assign_c)

        self.assign_probability_1 = Assign(target_idx=[4, 5, 6], arg_idx=[0, 1], f=self.agent1)
        self.assign_comparison_block_1 = nn.Sequential(
            self.assign_a_1,
            self.assign_b_1,
            self.assign_c_1,
        )

        self.ifelse_angle_b_1 = IfElse(target_idx=[12], test=self.comparison_bar, f_test=f_test, body=self.move_down_1, orelse=self.move_up_1)
        self.ifelse_angle_c_1 = IfElse(target_idx=[13], test=self.comparison_bar, f_test=f_test, body=self.move_right_1, orelse=self.move_up_1)
        # if a <= 0: if b <= 0: index = 0; else: index = 2; else: if c <= 0: index = 1; else: index = 2;
        self.ifelse_angle_1 = IfElse(target_idx=[11], test=self.comparison_bar, f_test=f_test, body=self.ifelse_angle_b_1, orelse=self.ifelse_angle_c_1)

        self.forward_update_1 = Assign(target_idx=[1], arg_idx=[1], f=f_forward)
    
        # the update of agent2
        self.move_down_2 = Assign(target_idx=[2], arg_idx=[2], f=f_move_down)
        self.move_right_2 = Assign(target_idx=[2], arg_idx=[2], f=f_move_right)
        self.move_up_2 = Assign(target_idx=[2], arg_idx=[2], f=f_move_up)

        self.assign_a_2 = Assign(target_idx=[14], arg_idx=[7, 8], f=f_assign_a)
        self.assign_b_2 = Assign(target_idx=[15], arg_idx=[7, 9], f=f_assign_b)
        self.assign_c_2 = Assign(target_idx=[16], arg_idx=[8, 9], f=f_assign_c)

        self.assign_probability_2 = Assign(target_idx=[7, 8, 9], arg_idx=[2, 3], f=self.agent2)
        self.assign_comparison_block_2 = nn.Sequential(
            self.assign_a_2,
            self.assign_b_2,
            self.assign_c_2,
        )

        self.ifelse_angle_b_2 = IfElse(target_idx=[15], test=self.comparison_bar, f_test=f_test, body=self.move_down_2, orelse=self.move_up_2)
        self.ifelse_angle_c_2 = IfElse(target_idx=[16], test=self.comparison_bar, f_test=f_test, body=self.move_right_2, orelse=self.move_up_2)
        # if a <= 0: if b <= 0: index = 0; else: index = 2; else: if c <= 0: index = 1; else: index = 2;
        self.ifelse_angle_2 = IfElse(target_idx=[14], test=self.comparison_bar, f_test=f_test, body=self.ifelse_angle_b_2, orelse=self.ifelse_angle_c_2)

        self.forward_update_2 = Assign(target_idx=[3], arg_idx=[3], f=f_forward)

        self.step_update = Assign(target_idx=[10], arg_idx=[10], f=f_step_update)
        self.assign_distance = Assign(target_idx=[17], arg_idx=[0, 2], f=f_assign_distance)
        
        self.trajectory_update = Trajectory(target_idx=[17, 0, 2, 1, 3, 4, 6, 7, 8, 9]) # d, x1, x2, y1, y2, p00, p01, p02, p10, p11, p12,
        # self.trajectory_update = Trajectory(target_idx=[0, 1])

        self.whileblock = nn.Sequential(
            self.assign_probability_1, # x1, y1 -> p00, p01, p02
            self.assign_comparison_block_1, # p00, p01, p02 -> a1, b1, c1
            self.ifelse_angle_1, # update x1, y1 according to a1, b1, c1
            self.forward_update_1,  # update y1: y1 += 1
            self.assign_probability_2, # x2, y2 -> p10, p11, p12
            self.assign_comparison_block_2, # p10, p11, p12 -> a2, b2, c2
            self.ifelse_angle_2, # update x2, y2 according to a2, b2, c2
            self.forward_update_2,  # update y2: y2 += 1
            self.step_update, # update step i
            self.assign_distance,
            self.trajectory_update, 
        )
        self.whilecase = While(target_idx=[10], test=self.step, body=self.whileblock)
        self.program = nn.Sequential(
            self.trajectory_update, # update the first state  
            self.whilecase,
        )
    
    def forward(self, input, version=None):
        if version == "single_nn_learning":
            # model the car-controller
            B, D = input.shape
            res = torch.zeros(B, 3) # placeholder for res, three outputs: p0, p1, p2
            if torch.cuda.is_available():
                res = res.cuda()
            agent1_index = input[:, 2] == 0
            agent2_index = input[:, 2] == 1
            res[agent1_index] = self.agent1(input[agent1_index][:, :-1])
            res[agent2_index] = self.agent2(input[agent2_index][:, :-1])
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









        
