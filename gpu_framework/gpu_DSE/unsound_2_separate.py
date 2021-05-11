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


def f_test(x):
    return x


class LinearAssign(nn.Module):
    def __init__(self, l=1):
        super().__init__()
        self.linear = Linear(in_channels=1, out_channels=1)
    
    def forward(self, x):
        res = self.linear(x)
        return res


def f_assign_min_z(x):
    return x.set_value(var(1.0))

def f_assign_max_z(x):
    return x.set_value(var(10.0))


class Unsound_2_Separate(nn.Module):
    def __init__(self, l=1):
        super(Unsound_2_Separate, self).__init__()
        self.bar = var(1.0)
        self.nn = LinearAssign(l=l)

        self.assign_y = Assign(target_idx=[1], arg_idx=[0], f=self.nn)

        self.assign_min_z = Assign(target_idx=[2], arg_idx=[2], f=f_assign_min_z)
        self.assign_max_z = Assign(target_idx=[2], arg_idx=[2], f=f_assign_max_z)
        self.ifelse_z = IfElse(target_idx=[1], test=self.bar, f_test=f_test, body=self.assign_max_z, orelse=self.assign_min_z)

        self.assign_y_2 = Assign(target_idx=[1], arg_idx=[2], f=self.nn)
        self.trajectory_update_1 = Trajectory(target_idx=[1])
        self.trajectory_update_2 = Trajectory(target_idx=[1])
        self.program = nn.Sequential(
            self.assign_y,
            self.trajectory_update_1,
            self.ifelse_z,
            self.assign_y_2, 
            self.trajectory_update_2,
        )
    
    def forward(self, input, version=None):
        if version == "single_nn_learning":
            # TODO: use a batch-wise way
            y = self.nn(input)
            print(f"y: {y.detach().cpu().numpy().tolist()[:3]}")
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








        
