
import domain
from constants import *
from helper import * 
from point_interpretor import *

# Thermostat
# safe: x
# return(y): res

if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    if DOMAIN == "interval":
        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['i'] = domain.Interval(0, 0)
            symbol_table['x'] = domain.Interval(x_l[0], x_r[0])
            symbol_table['isOn'] = domain.Interval(0.0, 0.0)
            symbol_table['probability'] = var(1.0)

            symbol_table['res'] = domain.Interval(0.0, 0.0)
            symbol_table['x_min'] = domain.Interval(P_INFINITY.data.item(), P_INFINITY.data.item())
            symbol_table['x_max'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
            symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())])
            symbol_table['probability'] = var(1.0)
            symbol_table['explore_probability'] = var(1.0)

            symbol_table_list.append(symbol_table)

            return symbol_table_list


if MODE == 2:
    from disjunction_of_intervals_interpretor import *

    def initialization(x_l, x_r):

        symbol_table_list = list()

        symbol_table = dict()
        symbol_table['i'] = domain.Interval(0, 0)
        symbol_table['x'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['isOn'] = domain.Interval(0.0, 0.0)
        symbol_table['probability'] = var(1.0)

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        symbol_table_list.append(symbol_table)
        
        return symbol_table_list    

if MODE == 3:
    from partial_disjunction_of_intervals_interpretor import *

    def initialization(x_l, x_r):

        symbol_table_list = list()

        symbol_table = dict()
        symbol_table['i'] = domain.Interval(0, 0)
        symbol_table['x'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['isOn'] = domain.Interval(0.0, 0.0)
        symbol_table['probability'] = var(1.0)

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        symbol_table_list.append(symbol_table)
        
        return symbol_table_list 


if MODE == 1:
    from interval_interpretor import *
    
    def initialization(x_l, x_r):

        symbol_table = dict()
        symbol_table['i'] = domain.Interval(0, 0)
        symbol_table['x'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['isOn'] = domain.Interval(0.0, 0.0)
        symbol_table['probability'] = var(1.0)

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        
        return symbol_table    


def initialization_point(x):
    symbol_table_point = dict()
    symbol_table_point['i'] = var(0.0)
    symbol_table_point['x'] = var(x[0])
    symbol_table_point['isOn'] = var(0.0)

    symbol_table_point['res'] = var(0.0)
    symbol_table_point['x_min'] = P_INFINITY
    symbol_table_point['x_max'] = N_INFINITY
    symbol_table_point['probability'] = var(1.0)
    symbol_table_point['explore_probability'] = var(1.0)

    return symbol_table_point


def f6(x):
    return x[0].sub(var(0.1).mul(x[0].sub(var(60))))
def f6_domain(x):
    return x[0].sub_l(x[0].sub(var(60)).mul(var(0.1)))
def f18(x):
    return x[0].add(var(1.0))
def f18_domain(x):
    return x[0].add(var(1.0))
def f8(x):
    return var(1.0)
def f8_domain(x):
    return x[0].setValue(var(1.0))
def f10(x):
    return var(0.0)
def f10_domain(x):
    return x[0].setValue(var(0.0))
def f12(x):
    return x[0].sub(var(0.1).mul(x[0].sub(var(60)))).add(var(5.0))
def f12_domain(x):
    return x[0].sub(x[0].sub(var(60)).mul(var(0.1))).add(var(5.0))
def f19(x):
    return x[1]
def f19_domain(x):
    return x[1]
def f_max(x):
    return torch.max(x[0], x[1])
def f_max_domain(x):
    return x[0].max(x[1])
def f_min(x):
    return torch.min(x[0], x[1])
def f_min_domain(x):
    return x[0].min(x[1])


def construct_syntax_tree(Theta):
    
    l21 = Assign(['x_max', 'x'], f_max_domain, None)
    l20 = Assign(['x_min', 'x'], f_min_domain, l21)
    l18 = Assign('i', f18, l20)

    l8 = Assign(['isOn'], f8, None)
    l10 = Assign(['isOn'], f10, None)
    l7 = Ifelse('x', Theta, l8, l10, None)

    l6 = Assign(['x'], f6, l7)

    l14 = Assign(['isOn'], f8, None)
    l16 = Assign(['isOn'], f10, None)
    l13 = Ifelse('x', var(80.0), l14, l16, None)

    l12 = Assign(['x'], f12, l13)
    l5 = Ifelse('isOn', var(0.5), l6, l12, l18)

    l19 = Assign(['res', 'x'], f19, None)
    l4 = WhileSimple('i', var(40.0), l5, l19)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    l21 = AssignPoint(['x_max', 'x'], f_max, None)
    l20 = AssignPoint(['x_min', 'x'], f_min, l21)
    l18 = AssignPoint(['i'], f18, l20)

    l8 = AssignPoint(['isOn'], f8, None)
    l10 = AssignPoint(['isOn'], f10, None)
    l7 = IfelsePoint('x', Theta, l8, l10, None)

    l6 = AssignPoint(['x'], f6, l7)

    l14 = AssignPoint(['isOn'], f8, None)
    l16 = AssignPoint(['isOn'], f10, None)
    l13 = IfelsePoint('x', var(80.0), l14, l16, None)

    l12 = AssignPoint(['x'], f12, l13)
    l5 = IfelsePoint('isOn', var(0.5), l6, l12, l18)

    l19 = AssignPoint(['res', 'x'], f19, None)
    l4 = WhileSimplePoint('i', var(40), l5, l19)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    l21 = AssignPointSmooth(['x_max', 'x'], f_max, None)
    l20 = AssignPointSmooth(['x_min', 'x'], f_min, l21)
    l18 = AssignPointSmooth(['i'], f18, l20)

    l8 = AssignPointSmooth(['isOn'], f8, None)
    l10 = AssignPointSmooth(['isOn'], f10, None)
    l7 = IfelsePointSmooth('x', Theta, l8, l10, None)

    l6 = AssignPointSmooth(['x'], f6, l7)

    l14 = AssignPointSmooth(['isOn'], f8, None)
    l16 = AssignPointSmooth(['isOn'], f10, None)
    l13 = IfelsePointSmooth('x', var(80.0), l14, l16, None)

    l12 = AssignPointSmooth(['x'], f12, l13)
    l5 = IfelsePointSmooth('isOn', var(0.5), l6, l12, l18)


    l19 = AssignPoint(['res', 'x'], f19, None)
    l4 = WhileSimplePointSmooth('i', var(40.0), l5, l19)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict
