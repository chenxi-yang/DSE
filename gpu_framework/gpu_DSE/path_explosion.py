import torch
import torch.nn.functional as F
import torch.nn as nn

from gpu_DSE.modules import *

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


# input order: h0, bound, count, tmp_h_1, tmp_h_2
def initialization_abstract_state(component_list):
    abstract_state_list = list()
    # we assume there is only one abstract distribtion, therefore, one component list is one abstract state
    abstract_state = list()
    for component in component_list:
        center, width, p = component['center'], component['width'], component['p']
        symbol_table = {
            'x': domain.Box(var_list([center[0], 0.0, 0.0, 0.0, 0.0]), var_list([width[0], 0.0, 0.0, 0.0, 0.0])),
            'probability': var(p),
            'trajectory': list(),
            'branch': '',
        }

        abstract_state.append(symbol_table)
    abstract_state_list.append(abstract_state)
    return abstract_state_list


def f_test(x):
    return x


class LinearNN(nn.Module):
    def __init__(self, l=1):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        res = self.linear1(x)
        res = self.sigmoid(res)
        return res


class LinearNNComplex(nn.Module):
    def __init__(self, l=4):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

    def forward(self, x):
        res = self.linear1(x)
        res = self.relu(res)
        res = self.linear2(res)
        res = self.sigmoid(res)
        return res


def f_assign_update_h(x):
    return x.add(var(0.01))

def f_assign_tmp_h_1(x):
    # x[0] - x[1]
    x0 = x.select_from_index(0, index1)
    x1 = x.select_from_index(0, index2)
    return x0.sub_l(x1)

def f_assign_tmp_h_2(x):
    # x[0] - x[1]
    x0 = x.select_from_index(0, index1)
    x1 = x.select_from_index(0, index2)
    return x0.sub_l(x1.mul(var(3.0)))

def f_assign_2h(x):
    return x.mul(var(2.0))

def f_assign_3h(x):
    return x.mul(var(3.0))

def f_assign_10h(x):
    return x.mul(var(10.0))

def f_update_count(x):
    return x.add(var(1.0))

# input order: 0:h0, 1:bound, 2:count, 3:tmp_h_1, 4:tmp_h_2
class PathExplosion(nn.Module):
    def __init__(self, l=1, nn_mode="complex"):
        super(PathExplosion, self).__init__()
        self.goal_h = var(10.0)
        self.tmp_h_1_bound = var(0.0)
        self.tmp_h_2_bound = var(-0.001)

        # simple version
        if nn_mode == "simple":
            self.nn = LinearNN(l=l)
        # complex version
        if nn_mode == "complex":
            self.nn = LinearNNComplex(l=l)

        # new
        self.assign_bound = Assign(target_idx=[1], arg_idx=[0], f=self.nn)
        self.trajectory_update = Trajectory(target_idx=[0])

        self.assign_basic_h = Assign(target_idx=[1], arg_idx=[0], f=f_assign_update_h)
        self.assign_tmp_h_1 = Assign(target_idx=[3], arg_idx=[0, 1], f=f_assign_tmp_h_1)

        self.assign_2h = Assign(target_idx=[0], arg_idx=[0], f=f_assign_2h)
        self.assign_3h = Assign(target_idx=[0], arg_idx=[0], f=f_assign_3h)
        self.ifelse_h_in_2 = IfElse(target_idx=[3], test=self.tmp_h2_bound, f_test=f_test, body=self.assign_2h, orelse=self.assign_3h)
        self.h_skip = Skip()
        self.ifelse_h_in = IfElse(target_idx=[3], test=self.tmp_h_1_bound, f_test=f_test, body=self.ifelse_h_in_2, orelse=self.h_skip)

        self.assign_count = Assign(target_id=[2], arg_idx=[2], f=f_update_count)
        self.whileblock = nn.Sequential(
            self.assign_basic_h,
            self.assign_tmp_h_1,
            self.ifelse_h_in,
            self.assign_count,
            self.trajectory_update,
        )

        self.while_h = While(target_idx=[0], test=self.goal_h, body=self.whileblock)

        self.assign_tmp_h_2 = Assign(target_idx=[4], arg_idx=[0, 1], f=f_assign_tmp_h_2)

        self.assign_10h = Assign(target_idx=[0], arg_idx=[0], f=f_assign_10h)
        self.ifelse_h_out = IfElse(target_idx=[4], test=self.tmp_h_2_bound, f_test=f_test, body=self.h_skip, orelse=self.assign_10h)

        self.program = nn.Sequential(
            self.assign_bound,
            self.while_h,
            self.assign_tmp_h_2,
            self.ifelse_h_out,
            self.trajectory_update,
        )
    
    def forward(self, input, version=None):
        if version == "single_nn_learning":
            B = input.shape[0]
            count = torch.zeros(B, 1)

            bound = self.nn(input)
            while_idx = (input <= 10.0)
            while_condition_res = torch.any(while_idx)

            while(while_condition_res == True): # there is tensor in the batch <= 10.0
                input[while_idx] = input[while_idx] + 0.01
                bound_idx = (input <= bound)
                bound2_idx = (input <= (bound - 0.001))
                bound3_idx = (input > (bound - 0.001))
                input[torch.logical_and(bound_idx, bound2_idx)] = input[torch.logical_and(bound_idx, bound2_idx)] * 2.0
                input[torch.logical_and(bound_idx, bound3_idx)] = input[torch.logical_and(bound_idx, bound3_idx)] * 3.0

                count[while_idx] += 1
                while_idx = (input <= 10.0)
                while_condition_res = torch.any(while_idx)
            
            input[input > (3 * bound - 0.001)] = input[input > (3 * bound - 0.001)] * 10.0
            res = count
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








        
