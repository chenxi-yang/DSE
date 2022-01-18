import torch
import torch.nn.functional as F
import torch.nn as nn

from modules import *
from helper import * 
from constants import *
import domain

import os

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


# DiffAI version
def initialization_nn(center_list, width_list, p_list=None):
    # print(f"in initialization_nn")
    symbol_table_list = list()
    for idx, center in enumerate(center_list):
        width = width_list[idx]
        p = var(1.0) if p_list is None else p_list[i]
        symbol_table = {
            'x': domain.Box(var_list([0.0, 0.0, center[0], center[0]]), var_list([0.0, 0.0, width[0], width[0]])),
            # 'safe_range': domain.Interval(P_INFINITY, N_INFINITY),
            'probability': p,
            'trajectory': list(),
            'branch': '',
        }

        symbol_table_list.append(symbol_table)

    return symbol_table_list


def initialization_one_point_nn(x):
    return domain.Box(var_list([0.0, 0.0, x[0], x[0]]), var_list([0.0] * 4))


def initialization_point_nn(x):
    point_symbol_table_list = list()
    symbol_table = dict()
    symbol_table['x'] = domain.Box(var_list([0.0, 0.0, x[0], x[0]]), var_list([0.0] * 4))
    symbol_table['safe_range'] = domain.Interval(P_INFINITY, N_INFINITY)
    symbol_table['probability'] = var(1.0)
    symbol_table['explore_probability'] = var(1.0)
    symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY, N_INFINITY)])
    symbol_table['point_cloud'] = list()
    symbol_table['counter'] = var(0.0)
    point_symbol_table_list.append(symbol_table)

    return point_symbol_table_list


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
        return res, var(1.0)


class LinearReLU(nn.Module):
    def __init__(self, l, sig_range):
        super().__init__()
        self.linear1 = Linear(in_channels=2, out_channels=l)
        self.linear2 = Linear(in_channels=l, out_channels=1)
        self.relu = ReLU()
        self.sigmoid_linear = SigmoidLinear(sig_range=sig_range)

    def forward(self, x):
        # start_time = time.time()
        # print(f"LinearSig, before: {x.c, x.delta}")
        res = self.linear1(x)
        # print(f"LinearSig, after linear1: {res.c, res.delta}")
        res, q1 = self.relu(res)
        # print(f"LinearSig, after sigmoid: {res.c, res.delta}")
        res = self.linear2(res)
        # print(f"LinearSig, after linear2: {res.c, res.delta}")
        res, q2 = self.sigmoid_linear(res)
        # print(f"LinearSig, after sigmoid: {res.c, res.delta}")
        # exit(0)
        # print(f"time in LinearReLU: {time.time() - start_time}")
        return res, q1.mul(q2)


class ThermostatNN(nn.Module):
    def __init__(self, l, sig_range=10, nn_mode='all', module='linearrelu'):
        super(ThermostatNN, self).__init__()
        self.tOff = var(78.0)
        self.tOn = var(66.0)
        if module == 'linearsig':
            self.nn = LinearSig(l=l)
        if module == 'linearrelu':
            self.nn = LinearReLU(l=l, sig_range=sig_range)

        def f_tmp_down_nn(x):
            plant, p = self.nn(x)
            return x.select_from_index(0, index0).add(plant.mul(var(10.0))), p
        
        def f_tmp_up_nn(x):
            plant, p = self.nn(x)
            return x.select_from_index(0, index0).add(plant.mul(var(10.0))).add(var(5.0)), p

        # curL = curL + 10.0 * NN(curL, lin)
        self.assign1 = Assign(target_idx=[2], arg_idx=[2, 3], f=f_tmp_down_nn)

        # TODO: empty select index works?
        self.ifelse_tOn_block1 = Assign(target_idx=[1], arg_idx=[], f=lambda x: (x.set_value(var(1.0)), var(1.0)))
        self.ifelse_tOn_block2 = Skip()
        self.ifelse_tOn = IfElse(target_idx=[2], test=self.tOn, f_test=lambda x: x, body=self.ifelse_tOn_block1, orelse=self.ifelse_tOn_block2)
        self.ifblock1 = nn.Sequential(
            self.assign1, # DNN
            self.ifelse_tOn, # if x <= tOn: isOn=1.0 else: skip
        )

        if nn_mode == "single":
            # curL = curL + 0.1(curL - lin) + 5.0
            self.assign2 = Assign(target_idx=[2], arg_idx=[2, 3], f=lambda x: (f_up_temp(x), var(1.0)))
        if nn_mode == "all":
            # curL = curL + 10.0 * NN(curL, lin) + 5.0
            self.assign2 = Assign(target_idx=[2], arg_idx=[2, 3], f=f_tmp_up_nn)

        self.ifelse_tOff_block1 = Skip()
        self.ifelse_tOff_block2 = Assign(target_idx=[1], arg_idx=[], f=lambda x: (x.set_value(var(0.0)), var(1.0)))
        self.ifelse_tOff = IfElse(target_idx=[2], test=self.tOff, f_test=lambda x: x, body=self.ifelse_tOff_block1, orelse=self.ifelse_tOff_block2)

        self.ifblock2 = nn.Sequential(
            self.assign2,
            self.ifelse_tOff,  # if x <= tOff: skip else: isOn=0.0
        )

        self.ifelse_isOn = IfElse(target_idx=[1], test=var(0.5), f_test=lambda x: x, body=self.ifblock1, orelse=self.ifblock2)
        self.assign_update = Assign(target_idx=[0], arg_idx=[0], f=lambda x: (x.add(var(1.0)), var(1.0)))
        self.trajectory_update = Trajectory(target_idx=[2])
        self.whileblock = nn.Sequential(
            self.ifelse_isOn,
            self.assign_update,
            self.trajectory_update,
        )
        self.while1 = While(target_idx=[0], test=var(40.0), body=self.whileblock)
    
    def forward(self, x_list, transition='interval'):
        # if transition == 'abstract':
        # #     print(f"# of Partitions Before: {len(x_list)}")
        #     for x in x_list:
        #         print(f"x: {x['x'].c}, {x['x'].delta}")
        res_list = self.while1(x_list)
        # if transition == 'abstract':
        #     print(f"# of Partitions After: {len(res_list)}")
        #     # for x in res_list:
        #     #     print(f"x: {x['x'].c}, {x['x'].delta}")
        return res_list
    
    def clip_norm(self):
        if not hasattr(self, "weight"):
            return
        if not hasattr(self, "weight_g"):
            if torch.__version__[0] == "0":
                nn.utils.weight_norm(self, dim=None)
            else:
                nn.utils.weight_norm(self)


def load_model(folder, name, epoch=None):
    if os.path.isfile(folder):
        return None, torch.loader(folder)
    model_dir = os.path.join(folder, f"model_{name}")
    if not os.path.exists(model_dir):
        return None, None
    if epoch is None and os.listdir(model_dir):
        epoch = max(os.listdir(model_dir), key=int)
    path = os.path.join(model_dir, str(epoch))
    if not os.path.exists(path):
        return None, None
    return int(epoch), torch.load(path)


def save_model(model, folder, name, epoch):
    path = os.path.join(folder, f"model_{name}", str(epoch))
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    torch.save(model, path)
    








        