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
        from gpu_DSE.modules import *
    elif mode == 'DiffAI':
        from gpu_DiffAI.modules import *
elif constants.status == 'verify_AI':
    from modules_AI import *
elif constants.status == 'verify_SE':
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


def initialize_components(abstract_states):
    #TODO: add batched components to replace the following two 
    center, width = abstract_states['center'], abstract_states['width']
    B, D = center.shape
    padding = torch.zeros(B, 1)
    if torch.cuda.is_available():
        padding = padding.cuda()
    
    input_center, input_width = center[:, :1], width[:, :1]
    states = {
        'x': domain.Box(torch.cat((input_center, padding, padding, padding), 1), torch.cat((input_width, padding, padding, padding), 1)),
        'trajectories': [[] for i in range(B)],
        'idx_list': [i for i in range(B)],
        'p_list': [var(1.0) for i in range(B)], # might be changed to batch
    }

    return states


def f_self(x):
    return x

def f_test(x):
    return x

class LinearSig(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # print(f"LinearSig, before: {x.c, x.delta}")
        res = self.linear1(x)
        # print(f"LinearSig, after linear1: {res.c, res.delta}")
        res = self.sigmoid(res)
        # print(f"LinearSig, after sigmoid: {res.c, res.delta}")
        res = self.linear2(res)
        # print(f"LinearSig, after linear2: {res.c, res.delta}")
        res = self.sigmoid(res)
        # print(f"LinearSig, after sigmoid: {res.c, res.delta}")
        # exit(0)
        return res


class LinearReLU(nn.Module):
    def __init__(self, l, sig_range):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
        self.relu = ReLU()
        # self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        # self.sigmoid_linear = SigmoidLinear(sig_range=sig_range)

    def forward(self, x):
        # start_time = time.time()
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        # res = self.sigmoid(res)
        # !!!!!!! between [-1.0, 1.0]
        res = self.tanh(res)
        
        # print(f"time in LinearReLU: {time.time() - start_time}")
        return res


class LinearReLUNoAct(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=l)
        self.linear3 = Linear(in_channels=l, out_channels=2)
        self.linear_output = Linear(in_channels=2, out_channels=1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        # self.sigmoid_linear = SigmoidLinear(sig_range=sig_range)

    def forward(self, x):
        # final layer is not activation
        res = self.linear1(x)
        # res = self.relu(res)
        # # res = self.Sigmoid()
        # res = self.linear2(res)
        
        res = self.relu(res)
        res = self.linear3(res)
        res = self.relu(res)
        res = self.linear_output(res)
        # res = self.sigmoid(res)
        # !!!!!!! between [-1.0, 1.0]
        # print(f"in Linear")
        # res = res.mul(var(10.0))
        # print(f"time in LinearReLU: {time.time() - start_time}")
        return res


def reward_reach(x):
    return x.add(var(100.0)) 

def f_assign_min_p(x):
    return x.set_value(var(-1.2))

def f_assign_min_v(x):
    return x.set_value(var(0.0))

def f_assign_min_speed(x):
    return x.set_value(var(-0.07))

def f_assign_max_speed(x):
    return x.set_value(var(0.07))

def f_assign_max_acc(x):
    return x.set_value(var(1.2))

def f_assign_min_acc(x):
    return x.set_value(var(-1.2))

def f_assign_left_acc(x):
    return x.set_value(var(-0.5))

def f_assign_right_acc(x):
    return x.set_value(var(0.5))

def f_assign_reset_acc(x):
    return x.set_value(var(0.0))

def f_assign_update_p(x):
    return x.select_from_index(0, index0).add(x.select_from_index(0, index1))

def f_assign_reward_update(x):
    return x.select_from_index(0, index1).add(x.select_from_index(0, index0).mul(x.select_from_index(0, index0)).mul(var(-0.1)))

def f_assign_v(x):
    # x: p, v, u
    p = x.select_from_index(0, index0)
    v = x.select_from_index(0, index1)
    u = x.select_from_index(0, index2)
    # TODO: cos
    return v.add(u.mul(var(0.0015))).add(p.mul(var(3.0)).cos().mul(var(-0.0025)))


class Program(nn.Module):
    def __init__(self, l, nn_mode='all'):
        super(Program, self).__init__()
        self.goal_position = var(0.5)
        self.min_position = var(-1.2)
        self.min_speed = var(-0.07)
        self.initial_speed = var(0.0)
        self.max_speed = var(0.07)
        self.min_acc = var(-1.2)
        self.max_acc = var(1.2)
        self.change_acc = var(0.0)

        self.add_value = var(0.3)
        self.min_abs_acc = var(0.3)
        self.neg_min_abs_acc = - self.min_abs_acc
        
        self.nn = LinearReLUNoAct(l=l)
        
        ####
        self.assign_min_p = Assign(target_idx=[0], arg_idx=[0], f=f_assign_min_p)
        self.assign_min_v = Assign(target_idx=[1], arg_idx=[1], f=f_assign_min_v)
        self.ifelse_p_block1 = nn.Sequential(
            self.assign_min_p, 
            self.assign_min_v,
        )
        self.ifelse_p_block2 = Skip()
        self.ifelse_p = IfElse(target_idx=[0], test=self.min_position, f_test=f_test, body=self.ifelse_p_block1, orelse=self.ifelse_p_block2)

        self.assign_acceleration = Assign(target_idx=[2], arg_idx=[0, 1], f=self.nn)
        
        self.assign_min_acc = Assign(target_idx=[2], arg_idx=[2], f=f_assign_min_acc)  

        # use continuous acc
        self.ifelse_max_acc_block1 = Skip()
       
        self.assign_max_acc = Assign(target_idx=[2], arg_idx=[2], f=f_assign_max_acc)
        self.ifelse_max_acc = IfElse(target_idx=[2], test=self.max_acc, f_test=f_test, body=self.ifelse_max_acc_block1, orelse=self.assign_max_acc)

        # self.assign_min_acc = Assign(target_idx=[2], arg_idx=[2], f=f_assign_min_acc)
        self.ifelse_acceleration = IfElse(target_idx=[2], test=self.min_acc, f_test=f_test, body=self.assign_min_acc, orelse=self.ifelse_max_acc)

        self.assign_reward_update = Assign(target_idx=[3], arg_idx=[2, 3], f=f_assign_reward_update)
        self.assign_v = Assign(target_idx=[1], arg_idx=[0, 1, 2], f=f_assign_v)
        self.assign_block = nn.Sequential(
            self.assign_acceleration,
            self.ifelse_acceleration, # truncate acceleration
            self.assign_reward_update,
            self.assign_v,
        )

        self.ifelse_max_speed_block1 = Skip()
        self.assign_max_speed = Assign(target_idx=[1], arg_idx=[1], f=f_assign_max_speed)
        self.ifelse_max_speed = IfElse(target_idx=[1], test=self.max_speed, f_test=f_test, body=self.ifelse_max_speed_block1, orelse=self.assign_max_speed)
        
        self.assign_min_speed = Assign(target_idx=[1], arg_idx=[1], f=f_assign_min_speed)
        self.ifelse_v = IfElse(target_idx=[1], test=self.min_speed, f_test=f_test, body=self.assign_min_speed, orelse=self.ifelse_max_speed)
        
        self.assign_update_p = Assign(target_idx=[0], arg_idx=[0, 1], f=f_assign_update_p)
        self.trajectory_update_1 = Trajectory(target_idx=[2, 0])
        self.whileblock = nn.Sequential(
            self.ifelse_p,
            self.assign_block, 
            self.ifelse_v,
            self.assign_update_p, 
            self.trajectory_update_1, 
        )
        self.while1 = While(target_idx=[0], test=self.goal_position, body=self.whileblock)

        self.check_non = Skip() # nothing changes
        self.check_reach = Assign(target_idx=[3], arg_idx=[3], f=reward_reach)
        self.check_position = IfElse(target_idx=[0], test=self.goal_position-var(1e-5), f_test=f_test, body=self.check_non, orelse=self.check_reach)
        
        self.trajectory_update_2 = Trajectory(target_idx=[2, 0])
        self.program = nn.Sequential(
            self.while1,
            self.check_position,
            self.trajectory_update_2, # only use the final reward
        )


    def forward(self, input, transition='interval', version=None):
        if version == "single_nn_learning":
            # print(input.detach().cpu().numpy().tolist()[:3])
            res = self.nn(input)
            # print(res.detach().cpu().numpy().tolist()[:3])
            # res *= 10
            # res[res <= self.min_acc] = float(self.min_acc)
            # res[res > self.max_acc] = float(self.max_acc)

            # res[torch.abs(res) <= self.min_abs_acc] = float(self.min_acc)
            # res[torch.logical_and(res <= self.max_acc, res > self.min_acc)] = torch.sign(res[torch.logical_and(res <= self.max_acc, res > self.min_acc)]) *  0.5
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
    








        
