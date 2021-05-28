import torch
import torch.nn.functional as F
import torch.nn as nn

from modules_sound import *

import os

index0 = torch.tensor(0)
index1 = torch.tensor(1)
index2 = torch.tensor(2)
index3 = torch.tensor(3)
min_v = torch.tensor(1.0)
max_v = torch.tensor(10.0)

if torch.cuda.is_available():
    index0 = index0.cuda()
    index1 = index1.cuda()
    index2 = index2.cuda()
    index3 = index3.cuda()
    min_v = min_v.cuda()
    max_v = max_v.cuda()


# input order: x, y, z
def initialization_abstract_state(component_list):
    abstract_state_list = list()
    # we assume there is only one abstract distribtion, therefore, one component list is one abstract state
    abstract_state = list()
    for component in component_list:
        center, width, p = component['center'], component['width'], component['p']
        symbol_table = {
            'x': domain.Box(var_list([center[0], 0.0, 0.0]), var_list([width[0], 0.0, 0.0])),
            'probability': var(p),
            'trajectory': list(),
            'branch': '',
        }

        abstract_state.append(symbol_table)
    abstract_state_list.append(abstract_state)
    return abstract_state_list


def initialization_point_nn(x):
    point_symbol_table_list = list()
    symbol_table = {
        'x': domain.Box(var_list([x[0], 0.0, 0.0]), var_list([0.0] * 3)),
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
        res = self.linear1(x)
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


def f_assign_min_z(x):
    return x.set_value(min_v)

def f_assign_max_z(x):
    return x.set_value(max_v)


class Unsound_1(nn.Module):
    def __init__(self, l=1, nn_mode="simple"):
        super(Unsound_1, self).__init__()
        self.bar = var(1.0)
        if nn_mode == "simple":
            self.nn = LinearNN(l=l)
        # complex version
        if nn_mode == "complex":
            self.nn = LinearNNComplex(l=l)

        self.assign_y = Assign(target_idx=[1], arg_idx=[0], f=self.nn)

        self.assign_min_z = Assign(target_idx=[2], arg_idx=[2], f=f_assign_min_z)
        self.assign_max_z = Assign(target_idx=[2], arg_idx=[2], f=f_assign_max_z)
        self.ifelse_z = IfElse(target_idx=[1], test=self.bar, f_test=f_test, body=self.assign_max_z, orelse=self.assign_min_z)

        self.trajectory_update = Trajectory(target_idx=[2])
        self.program = nn.Sequential(
            self.assign_y,
            self.ifelse_z,
            self.trajectory_update,
        )
    
    def forward(self, input, version=None):
        if version == "single_nn_learning":
            # TODO: use a batch-wise way
            y = self.nn(input)
            # print(f"y: {y.detach().cpu().numpy().tolist()[:3]}")
            x = torch.clone(y)
            x[y <= float(self.bar)] = 10.0
            x[y > float(self.bar)] = 1.0
            res = x
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








        
