
import domain
from constants import *
from helper import * 
from point_interpretor import *

# Thermostat deep branch
# safe: x
# return(y): res

if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    if DOMAIN == "interval":
        def initialization(x_l, x_r, X_train, y_train):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['i'] = domain.Interval(0, 0)
            symbol_table['x'] = domain.Interval(x_l[0], x_r[0])
            symbol_table['isOn'] = domain.Interval(0.0, 0.0)

            symbol_table['res'] = domain.Interval(0.0, 0.0)
            symbol_table['x_min'] = domain.Interval(P_INFINITY.data.item(), P_INFINITY.data.item())
            symbol_table['x_max'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
            symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())])
            symbol_table['probability'] = var(1.0)
            symbol_table['explore_probability'] = var(1.0)

            # symbol_table['counter'] = var(len(X_train))
            #  ['point_cloud']
            # symbol_table_list.append(symbol_table)

            symbol_table = build_point_cloud(symbol_table, X_train, y_train, initialization_smooth_point)
            symbol_table_list.append(symbol_table)

            symbol_table_list = split_symbol_table(symbol_table, ['x'], partition=10)

            return symbol_table_list
    
    if DOMAIN == "zonotope":
        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['i'] = domain.Interval(0, 0).getZonotope()
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


def initialization_smooth_point(x, y):
    if DOMAIN == "interval":
        symbol_table = dict()
        symbol_table['i'] = domain.Interval(var(0.0), var(0.0))
        symbol_table['x'] = domain.Interval(var(x[0]),  var(x[0]))
        symbol_table['isOn'] = domain.Interval(var(0.0), var(0.0))

        symbol_table['res'] = domain.Interval(var(0.0), var(0.0))
        symbol_table['x_min'] = domain.Interval(P_INFINITY, P_INFINITY)
        symbol_table['x_max'] = domain.Interval(N_INFINITY, N_INFINITY)
        symbol_table['probability'] = var(1.0)
        symbol_table['explore_probability'] = var(1.0)

        symbol_table['target_res'] = var(y)

    return symbol_table


def initialization_point(x):
    symbol_table = dict()
    symbol_table['i'] = var(0.0)
    symbol_table['x'] = var(x[0])
    symbol_table['isOn'] = var(0.0)

    symbol_table['res'] = var(0.0)
    symbol_table['x_min'] = P_INFINITY
    symbol_table['x_max'] = N_INFINITY
    symbol_table['probability'] = var(1.0)
    symbol_table['explore_probability'] = var(1.0)

    return symbol_table

# function in ifelse condition
def fself(x):
    return x
def fself_add(x):
    return x.add(var(2.0))

# function in assign
def f_max(x):
    return torch.max(x[0], x[1])
def f_max_domain(x):
    return x[0].max(x[1])
def f_min(x):
    y = torch.min(x[0], x[1])
    return y
def f_min_domain(x):
    return x[0].min(x[1])


def f6(x):
    return x[0].sub(var(0.1).mul(x[0].sub(var(60))))
def f6_domain(x):
    return x[0].sub_l((x[0].sub_l(var(60))).mul(var(0.1)))
def f6_theta(theta):
    def res(x):
        return x[0].sub(var(0.1).mul(x[0].sub(theta)))
    return res
def f6_theta_domain(theta):
    def res(x):
        return x[0].sub_l((x[0].sub_l(theta)).mul(theta))
    return res

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
    return x[0].sub_l((x[0].sub_l(var(60.0))).mul(var(0.1))).add(var(5.0))
def f12_theta(theta):
    def res(x):
        return x[0].sub(var(0.1).mul(x[0].sub(theta))).add(var(5.0))
    return res
def f12_theta_domain(theta):
    def res(x):
        return x[0].sub_l((x[0].sub_l(theta)).mul(var(0.1))).add(var(5.0))
    return res

def f19(x):
    return x[1]
def f19_domain(x):
    return x[1]


def construct_syntax_tree(Theta):
    
    l18_2 = Assign(['x_max', 'x'], f_max_domain, None)
    l18_1 = Assign(['x_min', 'x'], f_min_domain, l18_2)
    l18 = Assign('i', f18_domain, l18_1)

    l8 = Assign(['isOn'], f8_domain, None)
    l10 = Assign(['isOn'], f10_domain, None)
    l7 = Ifelse('x', Theta, fself_add, l8, l10, None)

    l6 = Assign(['x'], f6_domain, l7)

    l14 = Assign(['isOn'], f8_domain, None)
    l16 = Assign(['isOn'], f10_domain, None)
    l13 = Ifelse('x', var(80.0), fself, l14, l16, None)

    l12 = Assign(['x'], f12_domain, l13)
    l5 = Ifelse('isOn', var(0.5), fself, l6, l12, l18)

    l19 = Assign(['res', 'x'], f19_domain, None)
    l4 = WhileSample('i', var(40.0), l5, l19)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    l18_2 = AssignPoint(['x_max', 'x'], f_max, None)
    l18_1 = AssignPoint(['x_min', 'x'], f_min, l18_2)
    l18 = AssignPoint('i', f18, l18_1)

    l8 = AssignPoint(['isOn'], f8, None)
    l10 = AssignPoint(['isOn'], f10, None)
    l7 = IfelsePoint('x', Theta, fself_add, l8, l10, None)

    l6 = AssignPoint(['x'], f6, l7)

    l14 = AssignPoint(['isOn'], f8, None)
    l16 = AssignPoint(['isOn'], f10, None)
    l13 = IfelsePoint('x', var(80.0), fself, l14, l16, None)

    l12 = AssignPoint(['x'], f12, l13)
    l5 = IfelsePoint('isOn', var(0.5), fself, l6, l12, l18)

    l19 = AssignPoint(['res', 'x'], f19, None)
    l4 = WhilePoint('i', var(40), l5, l19)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    l18_2 = AssignPointSmooth(['x_max', 'x'], f_max, None)
    l18_1 = AssignPointSmooth(['x_min', 'x'], f_min, l18_2)
    l18 = AssignPointSmooth('i', f18, l18_1)

    l8 = AssignPointSmooth(['isOn'], f8, None)
    l10 = AssignPointSmooth(['isOn'], f10, None)
    l7 = IfelsePointSmooth('x', Theta, fself_add, l8, l10, None)

    l6 = AssignPointSmooth(['x'], f6, l7)

    l14 = AssignPointSmooth(['isOn'], f8, None)
    l16 = AssignPointSmooth(['isOn'], f10, None)
    l13 = IfelsePointSmooth('x', var(80.0), fself, l14, l16, None)

    l12 = AssignPointSmooth(['x'], f12, l13)
    l5 = IfelsePointSmooth('isOn', var(0.5), fself, l6, l12, l18)

    l19 = AssignPoint(['res', 'x'], f19, None)
    l4 = WhilePointSmooth('i', var(40.0), l5, l19)

    tree_dict = dict()
    tree_dict['entry'] = l4
    tree_dict['para'] = Theta

    return tree_dict
