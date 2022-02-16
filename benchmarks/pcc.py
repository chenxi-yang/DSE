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

# i, loss, sender_rate, latency_gradient, latency_ratio, sending_ratio
def initialize_components(abstract_states):
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
    # TODO
    loss_center, loss_width = center[:, :1], width[:, :1]
    sender_rate_center, sender_rate_width = center[:, 1:2], width[:, 1:2]
    states = {
        'x': domain.Box(torch.cat((padding, loss_center, sender_rate_center, padding, padding, padding, padding), 1), \
            torch.cat((padding, loss_width, sender_rate_width, padding, padding, padding, padding), 1)),
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
    loss_center = torch.rand(B, 1) * (x_r[0] - x_l[0]) + x_l[0]
    loss_center[0] = x_r[0]
    loss_center[1] = x_l[0]
    sender_rate_center = torch.rand(B, 1) * (x_r[1] - x_l[1]) + x_l[1]
    sender_rate_center[0] = x_r[1]
    sender_rate_center[1] = x_l[1]

    loss_width, sender_rate_width, padding = torch.zeros(B, 1), torch.zeros(B, 1), torch.zeros(B, 1)
    
    if torch.cuda.is_available():
        padding = padding.cuda()
        variable = variable.cuda()
        loss_center, sender_rate_center = loss_center.cuda(), sender_rate_center.cuda()
        loss_width, sender_rate_width = loss_width.cuda(), sender_rate_width.cuda()
    
    states = {
        'x': domain.Box(torch.cat((padding, loss_center, sender_rate_center, padding, padding, padding, padding), 1), \
            torch.cat((padding, loss_width, sender_rate_width, padding, padding, padding, padding), 1)),
        # 'trajectories': [[] for i in range(B)],
        'trajectories_l': [[] for i in range(B)],
        'trajectories_r': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(0.0) for i in range(B)], # might be changed to batch
        'alpha_list': [var(1.0) for i in range(B)],
    }

    return states

class LinearReLU(nn.Module):
    def __init__(self, l):
        super().__init__()
        # x1, y1, x2, y2, stage, step
        self.linear1 = Linear(in_channels=3, out_channels=l)
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
    return x.add(var(1.0))

def f_assign_upper_sender_ratio(x):
    sender_ratio = x.select_from_index(1, index0)
    action = x.select_from_index(1, index1)
    return sender_ratio.mul(action.add(var(1.0)))

def f_assign_down_sender_ratio(x):
    sender_ratio = x.select_from_index(1, index0)
    action = x.select_from_index(1, index1)
    return sender_ratio.mul((action.sub_l(var(1.0))).div(var(1.0)))

def f_assign_latency_ratio(x):
    return x.add(var(5*0.03/200 + 0.06))

def f_assign_sending_ratio(x):
    return x.mul(var(2.0/3))

def f_assign_latency_gradient(x):
    return x.mul(var(1/0.06))

# i, loss, sender_rate, latency_gradient, latency_ratio, sending_ratio, action, 
# 0,  1,    2,             3,               4,             5,  , 6
class Program(nn.Module):
    def __init__(self, l, nn_mode='all'):
        super(Program, self).__init__()
        self.steps = var(10)
        self.bw = var(200)
        self.lat = var(0.03)
        self.queue = var(5)
        self.comparison_bar = var(0.0)

        self.nn = LinearReLU(l=l)
        self.skip = Skip()

        self.assign_upper_sender_ratio = Assign(target_idx=[2], arg_idx=[2, 6], f=f_assign_upper_sender_ratio)
        self.assign_down_sender_ratio = Assign(target_idx=[2], arg_idx=[2, 6], f=f_assign_down_sender_ratio)
        self.assign_latency_ratio = Assign(target_idx=[4], arg_idx=[2], f=f_assign_latency_ratio)
        self.assign_sending_ratio = Assign(target_idx=[5], arg_idx=[2], f=f_assign_sending_ratio)
        self.assign_latency_gradient = Assign(target_idx=[3], arg_idx=[4], f=f_assign_latency_gradient)

        self.assign_update_action = Assign(target_idx=[2], arg_idx=[3, 4, 5], f=self.nn)
        
        self.assign_update_i = Assign(target_idx=[0], arg_idx=[0], f=f_update_i)

        self.ifelse_action = IfElse(target_idx=[6], test=self.comparison_bar, f_test=f_test, body=self.assign_upper_sender_ratio, orelse=self.assign_down_sender_ratio)
        self.trajectory_update = Trajectory(target_idx=[2]) # update the sender rate
        self.whileblock = nn.Sequential(
            self.ifelse_action,
            self.assign_latency_ratio,
            self.assign_sending_ratio,
            self.assign_latency_gradient,
            self.assign_update_action,
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
            res = self.nn(input)
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



    








        
