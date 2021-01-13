
import domain
from constants import *
from helper import * 
from point_interpretor import *

# Thermostat sample loop
# safe: x
# return(y): res

if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    if DOMAIN == "interval":

        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['x_neg'] = domain.Interval(-x_r[0], -x_l[0])
            symbol_table['x'] = domain.Interval(x_l[0], x_r[0])
            symbol_table['isOn'] = domain.Interval(0.0, 0.0)

            symbol_table['res'] = domain.Interval(0.0, 0.0)
            symbol_table['x_min'] = domain.Interval(P_INFINITY.data.item(), P_INFINITY.data.item())
            symbol_table['x_max'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
            symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())])
            symbol_table['probability'] = var(1.0)
            symbol_table['explore_probability'] = var(1.0)
            symbol_table_list.append(symbol_table)

            return symbol_table_list
    
    if DOMAIN == "zonotope":

        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['x_neg'] = domain.Interval(-x_r[0], -x_l[0]).getZonotope()
            symbol_table['x'] = domain.Interval(x_l[0], x_r[0]).getZonotope()
            symbol_table['isOn'] = domain.Interval(0.0, 0.0).getZonotope()

            symbol_table['res'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['x_min'] = domain.Interval(P_INFINITY.data.item(), P_INFINITY.data.item())
            symbol_table['x_max'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
            symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())])
            symbol_table['probability'] = var(1.0)
            symbol_table['explore_probability'] = var(1.0)
            symbol_table_list.append(symbol_table)

            return symbol_table_list
    

def initialization_point(x):
    symbol_table = dict()

    symbol_table['x_neg'] = var(-x[0])
    symbol_table['x'] = var(x[0])
    symbol_table['isOn'] = var(0.0)

    symbol_table['res'] = var(0.0)
    symbol_table['x_min'] = P_INFINITY
    symbol_table['x_max'] = N_INFINITY
    symbol_table['probability'] = var(1.0)
    symbol_table['explore_probability'] = var(1.0)

    return symbol_table


def f6(x):
    return x[0].sub(var(0.1).mul(x[0].sub(var(60))))
def f6_domain(x):
    return x[0].sub_l((x[0].sub_l(var(60.0))).mul(var(0.1)))
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
    return x[0].sub_l((x[0].sub_l(var(60))).mul(var(0.1))).add(var(5.0))
def f19(x):
    return x[1]
def f19_domain(x):
    return x[1]
def f_neg(x):
    return x[1].mul(var(-1.0))
def f_neg_domain(x):
    return x[1].mul(var(-1.0))

def f_max(x):
    return torch.max(x[0], x[1])
def f_max_domain(x):
    return x[0].max(x[1])
def f_min(x):
    y = torch.min(x[0], x[1])
    return y
def f_min_domain(x):
    return x[0].min(x[1])

# for if condition
def fself(x):
    return x


def construct_syntax_tree(Theta):

    l18_0 = Assign(['x_neg', 'x'], f_neg_domain, None)
    l18_0_min = Assign(['x_min', 'x'], f_min_domain, l18_0)
    l18_0_max = Assign(['x_max', 'x'], f_max_domain, l18_0_min)

    l8_0 = Assign(['isOn'], f8_domain, None)
    l10_0 = Assign(['isOn'], f10_domain, None)
    l7_0 = Ifelse('x', Theta, fself, l8_0, l10_0, None)

    l6_0 = Assign(['x'], f6_domain, l7_0)

    l14_0 = Assign(['isOn'], f8_domain, None)
    l16_0 = Assign(['isOn'], f10_domain, None)
    l13_0 = Ifelse('x', var(77.0), fself, l14_0, l16_0, None)

    l12_0 = Assign(['x'], f12_domain, l13_0)
    l5_0 = Ifelse('isOn', var(0.5), fself, l6_0, l12_0, l18_0_max)

    l19 = Assign(['res', 'x'], f19_domain, None)
    l4_0 = WhileSample('x_neg', var(-78.0), l5_0, l19)
    
    # l18 = Assign('i', f18, None)
    l18 = Assign(['x_neg', 'x'], f_neg_domain, None)
    l18_min = Assign(['x_min', 'x'], f_min_domain, l18)
    l18_max = Assign(['x_max', 'x'], f_max_domain, l18_min)

    l8 = Assign(['isOn'], f8_domain, None)
    l10 = Assign(['isOn'], f10_domain, None)
    l7 = Ifelse('x', Theta, fself, l8, l10, None)

    l6 = Assign(['x'], f6_domain, l7)

    l14 = Assign(['isOn'], f8_domain, None)
    l16 = Assign(['isOn'], f10_domain, None)
    l13 = Ifelse('x', var(77.0), fself, l14, l16, None)

    l12 = Assign(['x'], f12_domain, l13)
    l5 = Ifelse('isOn', var(0.5), fself, l6, l12, l18_max)

    # l19 = Assign(['res', 'x'], f19, None)
    l4 = WhileSample('x', var(74.0), l5, l4_0)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    # l18_0 = AssignPoint(['x_neg', 'x'], f_neg, None)
    l18_0 = AssignPoint(['x_neg', 'x'], f_neg, None)
    l18_0_min = AssignPoint(['x_min', 'x'], f_min, l18_0)
    l18_0_max = AssignPoint(['x_max', 'x'], f_max, l18_0_min)

    l8_0 = AssignPoint(['isOn'], f8, None)
    l10_0 = AssignPoint(['isOn'], f10, None)
    l7_0 = IfelsePoint('x', Theta, fself, l8_0, l10_0, None)

    l6_0 = AssignPoint(['x'], f6, l7_0)

    l14_0 = AssignPoint(['isOn'], f8, None)
    l16_0 = AssignPoint(['isOn'], f10, None)
    l13_0 = IfelsePoint('x', var(77.0), fself, l14_0, l16_0, None)

    l12_0 = AssignPoint(['x'], f12, l13_0)
    l5_0 = IfelsePoint('isOn', var(0.5), fself, l6_0, l12_0, l18_0_max)

    l19 = AssignPoint(['res', 'x'], f19, None)
    l4_0 = WhilePoint('x_neg', var(-78.0), l5_0, l19)

    # l18 = AssignPoint('i', f18, None)
    # l18 = AssignPoint(['x_neg', 'x'], f_neg, None)
    l18 = AssignPoint(['x_neg', 'x'], f_neg, None)
    l18_min = AssignPoint(['x_min', 'x'], f_min, l18)
    l18_max = AssignPoint(['x_max', 'x'], f_max, l18_min)

    l8 = AssignPoint(['isOn'], f8, None)
    l10 = AssignPoint(['isOn'], f10, None)
    l7 = IfelsePoint('x', Theta, fself, l8, l10, None)

    l6 = AssignPoint(['x'], f6, l7)

    l14 = AssignPoint(['isOn'], f8, None)
    l16 = AssignPoint(['isOn'], f10, None)
    l13 = IfelsePoint('x', var(77.0), fself, l14, l16, None)

    l12 = AssignPoint(['x'], f12, l13)
    l5 = IfelsePoint('isOn', var(0.5), fself, l6, l12, l18_max)

    # l19 = Assign(['res', 'x'], f19, None)
    l4 = WhilePoint('x', var(74.0), l5, l4_0)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    # l18_0 = AssignPointSmooth(['x_neg', 'x'], f_neg, None)
    l18_0 = AssignPointSmooth(['x_neg', 'x'], f_neg, None)
    l18_0_min = AssignPointSmooth(['x_min', 'x'], f_min, l18_0)
    l18_0_max = AssignPointSmooth(['x_max', 'x'], f_max, l18_0_min)

    l8_0 = AssignPointSmooth(['isOn'], f8, None)
    l10_0 = AssignPointSmooth(['isOn'], f10, None)
    l7_0 = IfelsePointSmooth('x', Theta, fself, l8_0, l10_0, None)

    l6_0 = AssignPointSmooth(['x'], f6, l7_0)

    l14_0 = AssignPointSmooth(['isOn'], f8, None)
    l16_0 = AssignPointSmooth(['isOn'], f10, None)
    l13_0 = IfelsePointSmooth('x', var(77.0), fself, l14_0, l16_0, None)

    l12_0 = AssignPointSmooth(['x'], f12, l13_0)
    l5_0 = IfelsePointSmooth('isOn', var(0.5), fself, l6_0, l12_0, l18_0_max)

    l19 = AssignPointSmooth(['res', 'x'], f19, None)
    l4_0 = WhilePointSmooth('x_neg', var(-78.0), l5_0, l19)

    # l18 = AssignPointSmooth(['i'], f18, None)
    # l18 = AssignPointSmooth(['x_neg', 'x'], f_neg, None)
    l18 = AssignPointSmooth(['x_neg', 'x'], f_neg, None)
    l18_min = AssignPointSmooth(['x_min', 'x'], f_min, l18)
    l18_max = AssignPointSmooth(['x_max', 'x'], f_max, l18_min)

    l8 = AssignPointSmooth(['isOn'], f8, None)
    l10 = AssignPointSmooth(['isOn'], f10, None)
    l7 = IfelsePointSmooth('x', Theta, fself, l8, l10, None)

    l6 = AssignPointSmooth(['x'], f6, l7)

    l14 = AssignPointSmooth(['isOn'], f8, None)
    l16 = AssignPointSmooth(['isOn'], f10, None)
    l13 = IfelsePointSmooth('x', var(77.0), fself, l14, l16, None)

    l12 = AssignPointSmooth(['x'], f12, l13)
    l5 = IfelsePointSmooth('isOn', var(0.5), fself, l6, l12, l18_max)

    # l19 = Assign(['res', 'x'], f19, None)
    l4 = WhilePointSmooth('x', var(74.0), l5, l4_0)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict
