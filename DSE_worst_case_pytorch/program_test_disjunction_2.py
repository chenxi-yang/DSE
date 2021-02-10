import domain
from constants import *
from helper import * 
from point_interpretor import *


if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    if DOMAIN == "interval":
        def initialization(x_l, x_r, X_train, y_train):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['h'] = domain.Interval(x_l[0], x_r[0])
            symbol_table['count'] = domain.Interval(0.0, 0.0) 

            symbol_table['res'] = domain.Interval(0.0, 0.0)
            symbol_table['x_min'] = domain.Interval(P_INFINITY.data.item(), P_INFINITY.data.item())
            symbol_table['x_max'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
            symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())])
            symbol_table['probability'] = var(1.0)
            symbol_table['explore_probability'] = var(1.0)

            symbol_table = build_point_cloud(symbol_table, X_train, y_train, initialization_smooth_point)
            symbol_table_list.append(symbol_table)

            return symbol_table_list
    
    if DOMAIN == "zonotope":
        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['h'] = domain.Interval(x_l[0], x_r[0]).getZonotope()
            symbol_table['count'] = domain.Interval(0.0, 0.0) .getZonotope()

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
        symbol_table['h'] = domain.Interval(var(x[0]),  var(x[0]))
        symbol_table['count'] = domain.Interval(var(0.0), var(0.0))

        symbol_table['res'] = domain.Interval(var(0.0), var(0.0))
        symbol_table['x_min'] = domain.Interval(P_INFINITY, P_INFINITY)
        symbol_table['x_max'] = domain.Interval(N_INFINITY, N_INFINITY)
        symbol_table['probability'] = var(1.0)
        symbol_table['explore_probability'] = var(1.0)

        symbol_table['target_res'] = var(y)

    return symbol_table

def initialization_point(x):

    symbol_table = dict()
    symbol_table['h'] = var(x[0])
    symbol_table['count'] = var(0.0) 

    symbol_table['res'] = var(0.0)
    symbol_table['x_min'] = P_INFINITY
    symbol_table['x_max'] = N_INFINITY
    symbol_table['probability'] = var(1.0)
    symbol_table['explore_probability'] = var(1.0)

    return symbol_table


def f1(x):
    return x[0].add(var(0.01))
def f1_domain(x):
    return x[0].add(var(0.01))
def f_double(x):
    return x[0].mul(var(2.0))
def f_double_domain(x):
    return x[0].mul(var(2.0))
def f_triple(x):
    # print('in triple', x[0])
    return x[0].mul(var(3.0))
def f_triple_domain(x):
    return x[0].mul(var(3.0))
def f_identity(x):
    return x[0]
def f_identity_domain(x):
    return x[0]
def f_add(x):
    return x[0].add(var(1.0))
def f_add_domain(x):
    return x[0].add(var(1.0))
def f_add_more(x):
    return x[0].add(var(10.0))
def f_add_more_domain(x):
    return x[0].add(var(10.0))
def f_max(x):
    return torch.max(x[0], x[1])
def f_max_domain(x):
    return x[0].max(x[1])
def f_min(x):
    y = torch.min(x[0], x[1])
    return y
def f_min_domain(x):
    return x[0].min(x[1])

def f_equal(x):
    return x[1].div(var(25.0))
def f_equal_domain(x):
    return x[1].mul(var(1/25.0))

# for if condition
def f_triple_sub(x):
    return x.mul(var(3.0)).sub(var(0.001))
def fsub(x):
    return x.sub(var(0.001))

def fself(x):
    return x



def construct_syntax_tree(Theta):

    l12 = Assign(['x_max', 'h'], f_max_domain, None)
    l11 = Assign(['x_min', 'h'], f_min_domain, l12)

    l10 = Assign(['h'], f_add_more_domain, l11)
    l9 = Assign(['h'], f_identity_domain, l11)
    l8 = Ifelse('h', Theta, f_triple_sub, l9, l10, l11)
    l7 = Assign(['res', 'count'], f_equal_domain, l8)

    l6_1 = Assign(['x_max', 'h'], f_max_domain, None)
    l6 = Assign(['x_min', 'h'], f_min_domain, l6_1)
    l5 = Assign(['count'], f_add_domain, l6)

    l4 = Assign(['h'], f_identity_domain, l11)
    l3_1 = Assign(['h'], f_triple_domain, l11)
    l3_0 = Assign(['h'], f_double_domain, l11)
    l3 = Ifelse('h', Theta, fsub, l3_0, l3_1, None)
    l2 = Ifelse('h', Theta, fself, l3, l4, l5)

    l1 = Assign(['h'], f1_domain, l2)
    l0 = WhileSample('h', var(10), l1, l7)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    l12 = AssignPoint(['x_max', 'h'], f_max, None)
    l11 = AssignPoint(['x_min', 'h'], f_min, l12)

    l10 = AssignPoint(['h'], f_add_more, l11)
    l9 = AssignPoint(['h'], f_identity, l11)
    l8 = IfelsePoint('h', Theta, f_triple_sub, l9, l10, l11)
    l7 = AssignPoint(['res', 'count'], f_equal, l8)

    l6_1 = AssignPoint(['x_max', 'h'], f_max, None)
    l6 = AssignPoint(['x_min', 'h'], f_min, l6_1)
    l5 = AssignPoint(['count'], f_add, l6)

    l4 = AssignPoint(['h'], f_identity, l11)
    l3_1 = AssignPoint(['h'], f_triple, l11)
    l3_0 = AssignPoint(['h'], f_double, l11)
    l3 = IfelsePoint('h', Theta, fsub, l3_0, l3_1, None)
    l2 = IfelsePoint('h', Theta, fself, l3, l4, l5)

    l1 = AssignPoint(['h'], f1, l2)
    l0 = WhilePoint('h', var(10), l1, l7)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    l12 = AssignPointSmooth(['x_max', 'h'], f_max, None)
    l11 = AssignPointSmooth(['x_min', 'h'], f_min, l12)

    l10 = AssignPointSmooth(['h'], f_add_more, l11)
    l9 = AssignPointSmooth(['h'], f_identity, l11)
    l8 = IfelsePointSmooth('h', Theta, f_triple_sub, l9, l10, l11)
    l7 = AssignPointSmooth(['res', 'count'], f_equal, l8)

    l6_1 = AssignPointSmooth(['x_max', 'h'], f_max, None)
    l6 = AssignPointSmooth(['x_min', 'h'], f_min, l6_1)
    l5 = AssignPointSmooth(['count'], f_add, l6)

    l4 = AssignPointSmooth(['h'], f_identity, l11)
    l3_1 = AssignPointSmooth(['h'], f_triple, l11)
    l3_0 = AssignPointSmooth(['h'], f_double, l11)
    l3 = IfelsePointSmooth('h', Theta, fsub, l3_0, l3_1, None)
    l2 = IfelsePointSmooth('h', Theta, fself, l3, l4, l5)

    l1 = AssignPointSmooth(['h'], f1, l2)
    l0 = WhilePointSmooth('h', var(10), l1, l7)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict