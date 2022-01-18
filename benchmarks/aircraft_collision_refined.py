import torch
import torch.nn as nn

from constants import *
import constants
import domain


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

if torch.cuda.is_available():
    index0 = index0.cuda()
    index1 = index1.cuda()
    index2 = index2.cuda()
    index3 = index3.cuda()

# i, x1, y1, x2, y2, distance, step, stage, x_co
# stage: [0, 1, 2, 3] == ['CRUISE', 'LEFT', 'STRAIGHT', 'RIGHT']
def initialize_components(abstract_states):
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    padding_y1 = torch.zeros(B, 1) - 15.0
    if torch.cuda.is_available():
        padding = padding.cuda()
        padding_y1 = padding_y1.cuda()
    
    input_center, input_width = center[:, :1], width[:, :1]
    states = {
        'x': domain.Box(torch.cat((padding, input_center, padding_y1, padding, padding, padding, padding, padding, padding), 1), \
            torch.cat((padding, input_width, padding, padding, padding, padding, padding, padding, padding), 1)),
        'trajectories': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # use the log_p here, so start from 0.0
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states


def initialization_components_point():
    B = 2
    input_center, input_width, padding, padding_y1 = torch.zeros(B, 1), torch.zeros(B, 1), torch.zeros(B, 1), torch.zeros(B, 1) - 15.0
    if torch.cuda.is_available():
        padding = padding.cuda()
        padding_y1 = padding_y1.cuda()
        input_center = input_center.cuda()
        input_width = input_width.cuda()
    
    input_center[0], input_width[0] = 12.0, 0.0
    input_center[1], input_width[1] = 12.5, 0.001
    states = {
        'x': domain.Box(torch.cat((padding, input_center, padding_y1, padding, padding, padding, padding, padding, padding), 1), \
            torch.cat((padding, input_width, padding, padding, padding, padding, padding, padding, padding), 1)),
        'trajectories': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states


def f_self(x):
    return x


class LinearReLU(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.linear1 = Linear(in_channels=5, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=l)
        self.linear3 = Linear(in_channels=l, out_channels=1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.relu(res)
        res = self.linear3(res)
        res = self.sigmoid(res) # for the range, the value would be more accurate
        return res

def f_test(x):
    return x

def f_update_i(x):
    return x.add(var(1))

def f_update_y1(x):
    return x.add(var(5.0))

def f_update_x2(x):
    return x.add(var(5.0))

def compute_distance(x):
    x1 = x.select_from_index(1, index0)
    y1 = x.select_from_index(1, index1)
    x2 = x.select_from_index(1, index2)
    y2 = x.select_from_index(1, index3)
    return ((x1.sub_l(x2)).mul(x1.sub_l(x2))).add((y1.sub_l(y2)).mul(y1.sub_l(y2)))

def f_assign_update_x1_left(x):
    return x.sub_l(var(5.0))

def f_assign_update_x1_right(x):
    return x.add(var(5.0))

# i, x1, y1, x2, y2, distance, step, stage, x_co
class Program(nn.Module):
    def __init__(self, l, nn_mode='all'):
        super(Program, self).__init__()
        self.steps = var(15)
        self.straight_speed = var(5.0)
        self.cruise_bar = var(0.25)
        self.left_bar = var(0.5)
        self.straight_bar = var(0.75)
        self.right_bar = var(1.0)

        self.nn_control = LinearReLU(l=l)
        self.skip = Skip()

        self.assign_update_x1_right = Assign(target_idx=[1], arg_idx=[1], f=f_assign_update_x1_right)
        self.assign_update_x1_left = Assign(target_idx=[1], arg_idx=[1], f=f_assign_update_x1_left)
        self.assign_distance = Assign(target_idx=[5], arg_idx=[1, 2, 3, 4], f=compute_distance)

        self.update_left_block = nn.Sequential(
            self.assign_update_x1_left,
        )
        self.update_right_block = nn.Sequential(
            self.assign_update_x1_right,
        )
        self.ifelse_stage_straight = IfElse(target_idx=[7], test=self.straight_bar, f_test=f_test, body=self.skip, orelse=self.update_right_block)
        self.ifelse_stage_left = IfElse(target_idx=[7], test=self.left_bar, f_test=f_test, body=self.update_left_block, orelse=self.ifelse_stage_straight)
        self.ifelse_stage = IfElse(target_idx=[7], test=self.cruise_bar, f_test=f_test, body=self.skip, orelse=self.ifelse_stage_left)
        self.assign_stage = Assign(target_idx=[7], arg_idx=[1,2,3, 4, 7], f=self.nn_control)

        self.assign_y1 = Assign(target_idx=[2], arg_idx=[2], f=f_update_y1)
        self.assign_x2 = Assign(target_idx=[3], arg_idx=[3], f=f_update_x2)
        self.assign_update_i = Assign(target_idx=[0], arg_idx=[0], f=f_update_i)
        self.trajectory_update = Trajectory(target_idx=[5]) # update the distance
        self.whileblock = nn.Sequential(
            self.assign_distance, 
            self.assign_stage,
            self.ifelse_stage,
            self.assign_y1,
            self.assign_x2,
            self.assign_update_i,
            self.trajectory_update,
        )
        self.program = While(target_idx=[0], test=self.steps, body=self.whileblock)
    
    def forward(self, input, version=None): # version describing data loss or safety loss
        if version == "single_nn_learning":
            # TODO: add update of input(seperate data loss)
            # input[:, 1] == 0: nn_cool
            # input[:, 1] == 1: nn_heat
            # use a placeholder for the results
            # update the placeholder in a piece-wise way
            # B, D = input.shape
            # res = torch.zeros(B, 1) # placeholder for x1_co
            # if torch.cuda.is_available():
            #     res = res.cuda()
            # left_index = input[:, 1] == 0
            # right_index = input[:, 1] == 1
            # res[left_index] = self.nn_left(input[left_index][:, :-1])
            # res[right_index] = self.nn_right(input[right_index][:, :-1])
            res = self.nn_control(input)
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



    








        
