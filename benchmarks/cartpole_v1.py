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

# i, x, x_dot, theta, theta_dot, costheta, sintheta, action, force, temp, thetaacc, xacc 
# stage: [0, 1, 2, 3] == ['CRUISE', 'LEFT', 'STRAIGHT', 'RIGHT']
def initialize_components(abstract_states):
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    # padding_y1 = torch.zeros(B, 1) - 15.0
    if torch.cuda.is_available():
        padding = padding.cuda()
        # padding_y1 = padding_y1.cuda()
    # TODO
    x_center, x_width = center[:, :1], width[:, :1]
    x_dot_center, x_dot_width = center[:, 1:2], width[:, 1:2]
    theta_center, theta_width = center[:, 2:3], width[:, 2:3]
    theta_dot_center, theta_dot_width = center[:, 3:], width[:, 3:]
    states = {
        'x': domain.Box(torch.cat((padding, x_center, x_dot_center, theta_center, theta_dot_center, padding, padding, padding, padding, padding, padding, padding), 1), \
            torch.cat((padding, x_width, x_dot_width, theta_width, theta_dot_width, padding, padding, padding, padding, padding, padding, padding), 1)),
        # 'trajectories': [[] for i in range(B)],
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # use the log_p here, so start from 0.0
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states


def initialization_components_point(x_l=None, x_r=None):
    B = 100
    x_center = torch.rand(B, 1) * (x_r[0] - x_l[0]) + x_l[0]
    x_center[0] = x_r[0]
    x_center[1] = x_l[0]
    x_dot_center = torch.rand(B, 1) * (x_r[1] - x_l[1]) + x_l[1]
    x_dot_center[0] = x_r[1]
    x_dot_center[1] = x_l[1]
    theta_center = torch.rand(B, 1) * (x_r[2] - x_l[2]) + x_l[2]
    theta_center[0] = x_r[2]
    theta_center[1] = x_l[2]
    theta_dot_center = torch.rand(B, 1) * (x_r[3] - x_l[3]) + x_l[3]
    theta_dot_center[0] = x_r[3]
    theta_dot_center[1] = x_l[3]

    x_width, x_dot_width, theta_width, theta_dot_width, padding = torch.zeros(B, 1), torch.zeros(B, 1)
    
    if torch.cuda.is_available():
        padding = padding.cuda()
        x_center, x_dot_center, theta_center, theta_dot_center = x_center.cuda(), x_dot_center.cuda(), theta_center.cuda(), theta_dot_center.cuda()
        x_width, x_dot_width, theta_width, theta_dot_width = x_width.cuda(), x_dot_width.cuda(), theta_width.cuda(), theta_dot_width.cuda()
    
    # input_center[0], input_width[0] = 12.0, 0.0
    # input_center[1], input_width[1] = 12.5, 0.001
    states = {
        'x': domain.Box(torch.cat((padding, x_center, x_dot_center, theta_center, theta_dot_center, padding, padding, padding, padding, padding, padding, padding), 1), \
            torch.cat((padding, x_width, x_dot_width, theta_width, theta_dot_width, padding, padding, padding, padding, padding, padding, padding), 1)),
        # 'trajectories': [[] for i in range(B)],
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
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
        # x1, y1, x2, y2, stage, step
        self.linear1 = Linear(in_channels=4, out_channels=l)
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
        res = self.sigmoid(res) # For classifier
        return res

def f_test(x):
    return x

def f_update_i(x):
    return x.add(var(1))

def f_assign_neg_force(x):
    return x.set_value(var(-10.0))
def f_assign_pos_force(x):
    return x.set_value(var(10.0))
def f_assign_costheta(x):
    return x.cos()
def f_assign_sintheta(x):
    return x.sin()
def f_assign_temp(x):
    force = x.select_from_index(1, index0)
    theta_dot = x.select_from_index(1, index1)
    sintheta = x.select_from_index(1, index2)
    return (force.add(sintheta.mul(theta_dot).mul(theta_dot).mul(var(0.05)))).mul(var(1.0/1.1))
def f_assign_thetaacc(x):
    sintheta = x.select_from_index(1, index0)
    costheta = x.select_from_index(1, index1)
    temp = x.select_from_index(1, index2)
    costheta_square = costheta.abs().mul(costheta.abs())
    upper = (sintheta.mul(var(9.8)).sub_l(costheta.mul(temp))).mul(var(2))
    down = costheta_square.mul(var(0.1/1.1)).sub_r(var(4/3))
    return upper.mul(down.div(var(1.0)))
def f_assign_xacc(x):
    temp = x.select_from_index(1, index0)
    thetaacc = x.select_from_index(1, index1)
    costheta = x.select_from_index(1, index2)
    return temp.sub_l(thetaacc.mul(costheta).mul(var(0.05/1.1)))
def f_assign_x(x):
    a = x.select_from_index(1, index0)
    x_dot = x.select_from_index(1, index1)
    return a.add(x_dot.mul(var(0.02)))
def f_assign_x_dot(x):
    x_dot = x.select_from_index(1, index0)
    xacc = x.select_from_index(1, index1)
    return x_dot.add(xacc.mul(var(0.02)))
def f_assign_theta(x):
    theta = x.select_from_index(1, index0)
    theta_dot = x.select_from_index(1, index1)
    return theta.add(theta_dot.mul(var(0.02)))
def f_assign_theta_dot(x):
    theta_dot = x.select_from_index(1, index0)
    thetaacc = x.select_from_index(1, index1)
    return theta_dot.add(thetaacc.mul(var(0.02)))


# i, x, x_dot, theta, theta_dot, costheta, sintheta, action, force, temp, thetaacc, xacc 
# 0,  1,  2,   3,     4,           5,          6,     7,      8,   9,      10,      11,       12,13,14,15,16,   17
class Program(nn.Module):
    def __init__(self, l, nn_mode='all'):
        super(Program, self).__init__()
        self.steps = var(10)
        self.comparison_bar = var(0.5)
        self.nn_classifier = LinearReLU(l=l)

        self.skip = Skip()

        self.assign_update_i = Assign(target_idx=[0], arg_idx=[0], f=f_update_i)

        self.assign_action = Assign(target_idx=[7], arg_idx=[1, 2, 3, 4], f=self.nn_classifier)
        
        self.assign_neg_force = Assign(target_idx=[8], arg_idx=[8], f=f_assign_neg_force)
        self.assign_pos_force = Assign(target_idx=[8], arg_idx=[8], f=f_assign_pos_force)
        self.ifelse_force = IfElse(target_idx=[7], test=self.comparison_bar, f_test=f_test, body=self.assign_neg_force, orelse=self.assign_pos_force)
        
        self.assign_costheta = Assign(target_idx=[5], arg_idx=[3], f=f_assign_costheta)
        self.assign_sintheta = Assign(target_idx=[6], arg_idx=[3], f=f_assign_sintheta)
        self.assign_temp = Assign(target_idx=[9], arg_idx=[8, 4, 6], f=f_assign_temp)
        self.assign_thetaacc = Assign(target_idx=[10], arg_idx=[6, 5, 9], f=f_assign_thetaacc)
        self.assign_xacc = Assign(target_idx=[11], arg_idx=[9, 10, 5], f=f_assign_xacc)
        self.assign_middle_states_block = nn.Sequential(
            self.assign_costheta,
            self.assign_sintheta,
            self.assign_temp,
            self.assign_thetaacc,
            self.assign_xacc,
        )
        self.assign_x = Assign(target_idx=[1], arg_idx=[1, 2], f=f_assign_x)
        self.assign_x_dot = Assign(target_idx=[2], arg_idx=[2, 11], f=f_assign_x_dot)
        self.assign_theta = Assign(target_idx=[3], arg_idx=[3, 4], f=f_assign_theta)
        self.assign_theta_dot = Assign(target_idx=[4], arg_idx=[4, 10], f=f_assign_theta_dot)
        self.assign_states_block = nn.Sequential(
            self.assign_x,
            self.assign_x_dot,
            self.assign_theta,
            self.assign_theta_dot,
        )
        self.trajectory_update = Trajectory(target_idx=[3, 1, 2, 4]) # update theta, x, x_dot, theta_dot
        self.whileblock = nn.Sequential(
            # self.assign_distance, 
            self.assign_action,  
            self.ifelse_force, 
            self.assign_middle_states_block,
            self.assign_states_block,
            self.trajectory_update,
            self.assign_update_i,
        )
        self.while_statement = While(target_idx=[0], test=self.steps, body=self.whileblock)
        self.program = nn.Sequential(
            self.trajectory_update,
            self.while_statement,
        )
    
    def forward(self, input, version=None): # version describing data loss or safety loss
        if version == "single_nn_learning":
            # TODO: add update of input(seperate data loss)
            res = self.nn_classifier(input)
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



    








        
