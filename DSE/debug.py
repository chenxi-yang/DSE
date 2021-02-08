import domain
from helper import *
import copy


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



# symbol_table = dict()
# symbol_table['x'] = domain.Interval(2.0, 3.0)

# # new_symbol_table = dict()
# # for key in symbol_table:
# #     new_symbol_table[key] = copy.deepcopy(symbol_table[key])

# # symbol_table['x'].left = symbol_table['x'].left.sub(var(3.0))
# # print('symbol_table', symbol_table['x'].left, symbol_table['x'].right)
# # print('new_symbol_table', new_symbol_table['x'].left, new_symbol_table['x'].right)

# symbol_table_list = split_symbol_table(symbol_table, ['x'], 10)
# for table in symbol_table_list:
#     print(table['x'].left, table['x'].right)