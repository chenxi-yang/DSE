import torch
import random
from torch.autograd import Variable
import copy
# import domain

def var(i, requires_grad=False):
    # print(i)
    return Variable(torch.tensor(i, dtype=torch.double), requires_grad=requires_grad)

PI = var((3373259426.0 + 273688.0 / (1 << 21)) / (1 << 30))
PI_TWICE = PI.mul(var(2.0))
PI_HALF = PI.div(var(2.0))

# print(PI)
# print(torch.sin(PI))
# print(torch.sin(PI_TWICE))
# print(torch.sin(PI_HALF))

def split_symbol_table(symbol_table, argument_list, partition=10):
    symbol_table_list = list()
    if len(argument_list) == 1:
        target = argument_list[0]
        original_length = symbol_table[target].getVolumn()
        domain_list = symbol_table[target].split(partition) # interval/zonotope split
        for domain in domain_list:
            # print('domain', domain.left, domain.right)
            new_symbol_table = dict()
            for key in symbol_table:
                # print('OUT key, target', key, target)
                if key == target:
                    # print('key, target', key, target)
                    new_symbol_table[key] = domain
                elif 'probability' in key:
                    new_symbol_table[key] = symbol_table[key].mul(var(1.0/partition))
                else:
                    new_symbol_table[key] = copy.deepcopy(symbol_table[key])
            symbol_table_list.append(new_symbol_table)
    elif len(argument_list) == 0:
        return [symbol_table]
    else:
        return [symbol_table]

    return symbol_table_list