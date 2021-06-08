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


# input order: yExp, p_gender, not_p_gender, gender, p_colRank, not_p_colRank, colRank, expRank, hire, m, n, g_f, g_m
def initialization_abstract_state(component_list):
    abstract_state_list = list()
    # we assume there is only one abstract distribtion, therefore, one component list is one abstract state
    abstract_state = list()
    for component in component_list:
        center, width, p = component['center'], component['width'], component['p']
        symbol_table = {
            'x': domain.Box(var_list([center[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                var_list([width[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
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
        # res = self.sigmoid(res)
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
        # res = self.sigmoid(res)
        return res


def f_assign_p_gender(x):
    return x.set_value(var(0.5))

def f_assign_not_p_gender(x):
    return x.set_value(var(0.5))

def f_assign_non_colRank(x):
    return x.set_value(var(1.0))

def f_assign_p_colRank(x):
    return x.set_value(var(0.5))

def f_assign_not_p_colRank(x):
    return x.set_value(var(0.5))

def f_assign_real_hire(x):
    return x.set_value(var(1.0))

def f_assign_false_hire(x):
    return x.set_value(var(0.0))

def f_assign_g_f(x):
    return x.set_value(var(1.0))

def f_assign_g_m(x):
    return x.set_value(var(1.0))

def f_assign_m(x):
    return x.set_value(var(1.0))

def f_assign_n(x):
    return x.set_value(var(1.0))


# input order: 
# 0:yExp, 
# 1:p_gender, 
# 2:not_p_gender, 
# 3:gender, 
# 4:p_colRank, 
# 5:not_p_colRank, 
# 6:colRank, 
# 7:expRank, 
# 8:hire
# 9: m (hire == 1, gender==0)
# 10: n (hire == 1, gender==1)
# 11: g_f (gender==0)
# 12: g_m (gender==1)
# assert(p(m==1)*p(g_f==1) >= p(n==1)*p(g_m==1))
class Fairness_1(nn.Module):
    def __init__(self, l=1, nn_mode="simple"):
        super(Fairness_1, self).__init__()
        self.p_gender = var(0.5)
        self.not_p_gender = 1 - self.p_gender
        self.p_colRank = var(0.5)
        self.not_p_colRank = 1 - self.p_colRank
        self.gender_bar = var(0.5)
        self.colRank_bar = var(0.5)
        self.expRank_bar = var(1.8)
        self.hire_bar = var(0.5)
        self.sample_population = [0, 1]
        if nn_mode == "simple":
            self.nn = LinearNN(l=l)
        if nn_mode == "complex":
            self.nn = LinearNNComplex(l=l)

        self.assign_p_gender = Assign(target_idx=[1], arg_idx=[1], f=f_assign_p_gender)
        self.assign_not_p_gender = Assign(target_idx=[2], arg_idx=[2], f=f_assign_not_p_gender)
        self.sample_gender = Sampler(target_idx=[3], sample_population=self.sample_population, weights_arg_idx=[1, 2])

        self.assign_p_colRank = Assign(target_idx=[4], arg_idx=[4], f=f_assign_p_colRank)
        self.assign_not_p_colRank = Assign(target_idx=[5], arg_idx=[5], f=f_assign_not_p_colRank)
        self.sample_colRank = Sampler(target_idx=[6], sample_population=self.sample_population, weights_arg_idx=[4, 5])

        self.assign_skip = Skip()
        self.assign_non_colRank = Assign(target_idx=[6], arg_idx=[6], f=f_assign_non_colRank)
        self.ifelse_gender = IfElse(target_idx=[3], test=self.gender_bar, f_test=f_test, body=self.assign_non_colRank, orelse=self.assign_skip)

        self.assign_expRank = Assign(target_idx=[7], arg_idx=[0, 6], f=self.nn)

        self.assign_real_hire = Assign(target_idx=[8], arg_idx=[8], f=f_assign_real_hire)
        self.assign_false_hire = Assign(target_idx=[8], arg_idx=[8], f=f_assign_false_hire)
        self.ifelse_expRank = IfElse(target_idx=[7], test=self.expRank_bar, f_test=f_test, body=self.assign_real_hire, orelse=self.assign_false_hire)
        self.ifelse_colRank = IfElse(target_idx=[6], test=self.colRank_bar, f_test=f_test, body=self.assign_real_hire, orelse=self.ifelse_expRank)

        self.assign_g_f = Assign(target_idx=[11], arg_idx=[11], f=f_assign_g_f)
        self.assign_g_m = Assign(target_idx=[12], arg_idx=[12], f=f_assign_g_m)
        self.ifelse_fairness_gender_1 = IfElse(target_idx=[3], test=self.gender_bar, f_test=f_test, body=self.assign_g_f, orelse=self.assign_g_m)
        
        self.assign_m = Assign(target_idx=[9], arg_idx=[9], f=f_assign_m)
        self.assign_n = Assign(target_idx=[10], arg_idx=[10], f=f_assign_n)
        self.ifelse_fairness_gender_2 = IfElse(target_idx=[3], test=self.gender_bar, f_test=f_test, body=self.assign_m, orelse=self.assign_n)

        self.ifelse_fairness_hire = IfElse(target_idx=[8], test=self.gender_bar, f_test=f_test, body=self.assign_skip, orelse=self.ifelse_fairness_gender_2)
        self.fairness_extract = nn.Sequential(
            self.ifelse_fairness_gender_1,
            self.ifelse_fairness_hire,
        )
            
        self.trajectory_update = Trajectory(target_idx=[9, 10, 11, 12])

        self.program = nn.Sequential(
            self.assign_p_gender,
            self.assign_not_p_gender,
            self.sample_gender,
            self.assign_p_colRank,
            self.assign_not_p_colRank,
            self.sample_colRank,
            self.ifelse_gender,
            self.assign_expRank,
            self.ifelse_colRank,
            self.fairness_extract,
            self.trajectory_update,
        )
    
    def forward(self, input, version=None):
        if version == "single_nn_learning":
            # TODO: use a batch-wise way
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








        
