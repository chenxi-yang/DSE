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

# i, x, heat, isOn
def initialize_components(abstract_states):
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
    
    input_center, input_width = center[:, :1], width[:, :1]
    states = {
        'x': domain.Box(torch.cat((padding, input_center, input_center, padding), 1), torch.cat((padding, input_width, input_width, padding), 1)),
        # 'trajectories': [[] for i in range(B)],
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states


def initialization_components_point(x_l=None, x_r=None):
    B = 100
    input_center = torch.rand(B, 1) * (x_r[0] - x_l[0]) + x_l[0]
    input_center[0] = x_r[0]
    input_center[1] = x_l[0]

    input_width, padding = torch.zeros(B, 1), torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
        input_center = input_center.cuda()
        input_width = input_width.cuda()
    
    # input_center[0], input_width[0] = 60.5, 0.0
    # input_center[1], input_width[1] = 60.5, 0.001
    states = {
        'x': domain.Box(torch.cat((padding, input_center, input_center, padding), 1), torch.cat((padding, input_width, input_width, padding), 1)),
        # 'trajectories': [[] for i in range(B)],
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
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
        self.linear1 = Linear(in_channels=1, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=l)
        self.linear3 = Linear(in_channels=l, out_channels=2)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.relu(res)
        res = self.linear3(res)
        res = self.sigmoid(res)
        return res

# can not pickle local object
def f_ifelse_tOn_block1(x):
    return x.set_value(var(1.0))

def f_test(x):
    return x

def assign_update(x):
    return x.add(var(1.0))

def f_cooling(x):
    k = var(0.1)
    dt = var(0.5)
    return x.mul(var(1.0) - k*dt)

def f_warming(x):
    k = var(0.1)
    dt = var(0.5)
    return x.select_from_index(1, index0).mul(var(1-0.1*0.5)).add(x.select_from_index(1, index1))

def f_update_heat(x):
    return x.mul(var(15.0))

# i, x, h, isOn
class Program(nn.Module):
    def __init__(self, l, nn_mode='all'):
        super(Program, self).__init__()
        self.h_unit = var(15.0)
        self.steps = var(39)
        # balance temperature: 70.0

        self.nn_cool = LinearReLU(l=l)
        self.nn_heat = LinearReLU(l=l)

        self.assign_cool_nn = Assign(target_idx=[3, 2], arg_idx=[1], f=self.nn_cool)
        self.assign_cooling = Assign(target_idx=[1], arg_idx=[1], f=f_cooling)
        self.cool_block = nn.Sequential(
            self.assign_cool_nn,
            self.assign_cooling,
        )

        # assign to (h, isOn)
        self.assign_heat_nn = Assign(target_idx=[3, 2], arg_idx=[1], f=self.nn_heat)
        self.assign_update_heat = Assign(target_idx=[2], arg_idx=[2], f=f_update_heat)
        self.assign_warming = Assign(target_idx=[1], arg_idx=[1, 2], f=f_warming)
        self.heat_block = nn.Sequential(
            self.assign_heat_nn,
            self.assign_update_heat,
            self.assign_warming,
        )
        self.ifelse_isOn = IfElse(target_idx=[3], test=var(0.5), f_test=f_test, body=self.cool_block, orelse=self.heat_block)
        
        self.assign_update = Assign(target_idx=[0], arg_idx=[0], f=assign_update)
        self.trajectory_update = Trajectory(target_idx=[1]) # update the temperature
        self.whileblock = nn.Sequential(
            self.ifelse_isOn,
            self.assign_update,
            self.trajectory_update,
        )
        self.while_statement = While(target_idx=[0], test=self.steps, body=self.whileblock)
        self.program = nn.Sequential(
            self.trajectory_update,
            self.while_statement,
        )
    
    def forward(self, input, version=None): # version describing data loss or safety loss
        if version == "single_nn_learning":
            # TODO: add update of input(seperate data loss)
            # input[:, 1] == 0: nn_cool
            # input[:, 1] == 1: nn_heat
            # use a placeholder for the results
            # update the placeholder in a piece-wise way
            B, D = input.shape
            res = torch.zeros(B, 2) # placeholder for res
            if torch.cuda.is_available():
                res = res.cuda()
            cool_index = input[:, 1] == 0
            heat_index = input[:, 1] == 1
            res[cool_index] = self.nn_cool(input[cool_index][:, :-1])
            res[heat_index] = self.nn_heat(input[heat_index][:, :-1])
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



    








        
