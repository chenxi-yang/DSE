import domain
from constants import *
from helper import * 
from point_interpretor import *

if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    def initialization(x_l, x_r):
        symbol_table_list = list()
        symbol_table = dict()
        symbol_table['h'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['count'] = domain.Interval(0.0, 0.0) 

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        symbol_table['x'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
        symbol_table['probability'] = var(1.0)

        symbol_table_list.append(symbol_table)

        return symbol_table_list


if MODE == 4:
    from disjunction_of_intervals_interpretor_loop import *

    def initialization(x_l, x_r):
        symbol_table_list = list()
        symbol_table = dict()
        symbol_table['h'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['count'] = domain.Interval(0.0, 0.0) 

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        symbol_table['x'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
        symbol_table['probability'] = var(1.0)

        symbol_table_list.append(symbol_table)

        return symbol_table_list


if MODE == 2:
    from disjunction_of_intervals_interpretor import *

    def initialization(x_l, x_r):
        symbol_table_list = list()
        symbol_table = dict()
        symbol_table['h'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['count'] = domain.Interval(0.0, 0.0)

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        symbol_table['x'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
        symbol_table['probability'] = var(1.0)

        symbol_table_list.append(symbol_table)

        return symbol_table_list


if MODE == 3:
    from partial_disjunction_of_intervals_interpretor import *

    def initialization(x_l, x_r):
        symbol_table_list = list()
        symbol_table = dict()
        symbol_table['h'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['count'] = domain.Interval(0.0, 0.0) 

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        symbol_table['x'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
        symbol_table['probability'] = var(1.0)

        symbol_table_list.append(symbol_table)

        return symbol_table_list


if MODE == 1:
    from interval_interpretor import *

    def initialization(x_l, x_r):
        symbol_table = dict()
        symbol_table['h'] = domain.Interval(x_l[0], x_r[0])
        symbol_table['count'] = domain.Interval(0.0, 0.0) 

        symbol_table['res'] = domain.Interval(0.0, 0.0)
        symbol_table['x'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
        symbol_table['probability'] = var(1.0)

        return symbol_table


def initialization_point(x):

    symbol_table = dict()
    symbol_table['h'] = var(x[0])
    symbol_table['count'] = var(0.0) 

    symbol_table['res'] = var(0.0)
    symbol_table['x'] = N_INFINITY
    symbol_table['probability'] = var(1.0)

    return symbol_table


def f1(x):
    return x[0].add(var(0.01))
def f_double(x):
    return x[0].mul(var(2.0))
def f_identity(x):
    return x[0]
def f_add(x):
    return x[0].add(var(1.0))
def f_max(x):
    return torch.max(x[0], x[1])
def f_equal(x):
    return x[1].div(var(25.0))


def construct_syntax_tree(Theta):

    l8 = Assign(['x', 'h'], f_max, None)
    l7 = Assign(['res', 'count'], f_equal, l8)

    l6 = Assign(['x', 'h'], f_max, None)
    l5 = Assign(['count'], f_add, l6)

    l4 = Assign(['h'], f_identity, None)
    l3 = Assign(['h'], f_double, None)
    l2 = Ifelse('h', Theta, l3, l4, l5)

    l1 = Assign(['h'], f1, l2)
    l0 = WhileSample('h', var(10), l1, l7)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    l8 = AssignPoint(['x', 'h'], f_max, None)
    l7 = AssignPoint(['res', 'count'], f_equal, l8)

    l6 = AssignPoint(['x', 'h'], f_max, None)
    l5 = AssignPoint(['count'], f_add, l6)

    l4 = AssignPoint(['h'], f_identity, None)
    l3 = AssignPoint(['h'], f_double, None)
    l2 = IfelsePoint('h', Theta, l3, l4, l5)

    l1 = AssignPoint(['h'], f1, l2)
    l0 = WhilePoint('h', var(10), l1, l7)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    l8 = AssignPointSmooth(['x', 'h'], f_max, None)
    l7 = AssignPointSmooth(['res', 'count'], f_equal, l8)
    l6 = AssignPointSmooth(['x', 'h'], f_max, None)
    l5 = AssignPointSmooth(['count'], f_add, l6)

    l4 = AssignPointSmooth(['h'], f_identity, None)
    l3 = AssignPointSmooth(['h'], f_double, None)
    l2 = IfelsePointSmooth('h', Theta, l3, l4, l5)

    l1 = AssignPointSmooth(['h'], f1, l2)
    l0 = WhilePointSmooth('h', var(10), l1, l7)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict