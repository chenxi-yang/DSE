import domain
from constants import *
from helper import * 
from point_interpretor import *

# aircraft collision deep branch
    
if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    if DOMAIN == "interval":
        def initialization(x_l, x_r):
            # print('here')

            symbol_table_list = list()

            symbol_table = dict()
            symbol_table['i'] = domain.Interval(0, 0)
            symbol_table['v1'] = domain.Interval(x_l[0], x_r[0])
            symbol_table['v2'] = domain.Interval(x_l[1], x_r[1])
            symbol_table['stage'] = domain.Interval(0.5, 0.5)
            symbol_table['steps'] = domain.Interval(0.0, 0.0)
            symbol_table['x1'] = domain.Interval(6.0, 6.0)
            symbol_table['y1'] = domain.Interval(0.0, 0.0)
            symbol_table['x2'] = domain.Interval(0.0, 0.0)
            symbol_table['y2'] = domain.Interval(4.0, 4.0)
            symbol_table['dist'] = domain.Interval(0.0, 0.0)
            symbol_table['x_dist'] = domain.Interval(0.0, 0.0)
            symbol_table['y_dist'] = domain.Interval(0.0, 0.0)
            symbol_table['x'] = domain.Interval(0.0, 0.0)

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
            symbol_table['i'] = domain.Interval(0, 0).getZonotope()
            symbol_table['v1'] = domain.Interval(x_l[0], x_r[0]).getZonotope()
            symbol_table['v2'] = domain.Interval(x_l[1], x_r[1]).getZonotope()
            symbol_table['stage'] = domain.Interval(0.5, 0.5).getZonotope()
            symbol_table['steps'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['x1'] = domain.Interval(6.0, 6.0).getZonotope()
            symbol_table['y1'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['x2'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['y2'] = domain.Interval(4.0, 4.0).getZonotope()
            symbol_table['dist'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['x_dist'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['y_dist'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['x'] = domain.Interval(0.0, 0.0).getZonotope()

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
    symbol_table['i'] = var(0.0)
    symbol_table['v1'] = var(x[0])
    symbol_table['v2'] = var(x[1])
    symbol_table['stage'] = var(0.5)
    symbol_table['steps'] = var(0.0)
    symbol_table['x1'] = var(6.0)
    symbol_table['y1'] = var(0.0)
    symbol_table['x2'] = var(0.0)
    symbol_table['y2'] = var(4.0)
    symbol_table['dist'] = var(0.0)
    symbol_table['x_dist'] = var(0.0)
    symbol_table['y_dist'] = var(0.0)
    symbol_table['x'] = var(0.0)

    symbol_table['res'] = var(0.0)
    symbol_table['x_min'] = P_INFINITY
    symbol_table['x_max'] = N_INFINITY
    symbol_table['probability'] = var(1.0)
    symbol_table['explore_probability'] = var(1.0)

    return symbol_table


# function in ifelse condition
def fself(x):
    return x

def f_max(x):
    return torch.max(x[0], x[1])
def f_max_domain(x):
    return x[0].max(x[1])
def f_min(x):
    y = torch.min(x[0], x[1])
    return y
def f_min_domain(x):
    return x[0].min(x[1])

def f_cal_dist(x):
    # print('f_cal_dist', x[1], x[2])
    return torch.sqrt(x[1].add(x[2]))
def f_cal_dist_domain(x):
    return (x[1].add(x[2])).sqrt()
def f_update_dist(x):
    return torch.sqrt(x[1].add(x[2]))
def f_update_dist_domain(x):
    return (x[1].add(x[2])).sqrt()
def f4(x):
    return var(1.5)
def f4_domain(x):
    return x[0].setValue(var(1.5))
def f5(x):
    return var(0.0)
def f5_domain(x):
    return x[0].setValue(var(0.0))
def f7(x):
    return var(0.5)
def f7_domain(x):
    return x[0].setValue(var(0.5))
def f13(x):
    return var(2.5)
def f13_domain(x):
    return x[0].setValue(var(2.5))
def f14(x):
    return var(0.0)
def f14_domain(x):
    return x[0].setValue(var(0.0))
def f20(x):
    return var(3.5)
def f20_domain(x):
    return x[0].setValue(var(3.5))
def f_sub(x):
    return x[0].sub(x[1])
def f_sub_domain(x):
    return x[0].sub_l(x[1])
def f_add(x):
    return x[0].add(x[1])
def f_add_domain(x):
    return x[0].add(x[1])
def f_add_one(x):
    return x[0].add(var(1.0))
def f_add_one_domain(x):
    return x[0].add(var(1.0))
def f_theta(theta):
    def f_ret(x):
        return theta.mul(var(2.0)).add(var(10.0))
    return f_ret
def f_theta_domain(theta):
    def f_ret(x):
        return x[0].setValue(theta.mul(var(2.0)).add(var(10.0)))
    return f_ret

def f_dist_sub(x):
    return x[1].sub(x[2])
def f_dist_sub_domain(x):
    return x[1].sub_l(x[2])
def f_square(x):
    return x[0].mul(x[0])
def f_square_domain(x):
    return x[0].mul(x[0])


def construct_syntax_tree(Theta):

    l7 = Assign(['stage'], f7_domain, None)
    l5 = Assign(['steps'], f5_domain, None)
    l4 = Assign(['stage'], f4_domain, l5) # stage = 1.5, LEFT

    l3 = Ifelse('dist', Theta, fself, l4, l7, None)

    l2_1 = Assign(['dist', 'x_dist', 'y_dist'], f_cal_dist_domain, l3) # cal dist(safe property) # TODO:use all left/right
    l2_04 = Assign(['y_dist'], f_square_domain, l2_1)
    l2_03 = Assign(['x_dist'], f_square_domain, l2_04)
    l2_02 = Assign(['y_dist', 'y1', 'y2'], f_dist_sub_domain, l2_03)
    l2_01 = Assign(['x_dist', 'x1', 'x2'], f_dist_sub_domain, l2_02)
    l2_0 = Assign(['x2', 'v2'], f_add_domain, l2_01)
    l2 = Assign(['y1', 'v1'], f_add_domain, l2_0)

    l26 = Assign(['stage'], f7_domain, None)
    l25 = Assign(['stage'], f20_domain, None)
    l24 = Ifelse('steps', var(2.0), fself, l25, l26, None)

    l23 = Assign(['steps'], f_add_one_domain, l24)
    l22_3 = Assign(['y2', 'v2'], f_sub_domain, l23)
    l22_2 = Assign(['x2', 'v2'], f_add_domain, l22_3)
    l22_1 = Assign(['y1', 'v1'], f_add_domain, l22_2)
    l22 = Assign(['x1', 'v1'], f_add_domain, l22_1)

    l21 = Assign(['steps'], f14_domain, None)
    l20 = Assign(['stage'], f20_domain, l21)
    l19 = Assign(['stage'], f13_domain, None)
    l18 = Ifelse('steps', var(3.1), fself, l19, l20, None)

    l17 = Assign(['steps'], f_add_one_domain, l18)
    l16_1 = Assign(['x2', 'v2'], f_add_domain, l17)
    l16 = Assign(['y1', 'v1'], f_add_domain, l16_1)
    l15 = Ifelse('stage', var(3), fself, l16, l22, None)

    l14 = Assign(['steps'], f14_domain, None)
    l13 = Assign(['stage'], f13_domain, l14)
    l12 = Assign(['stage'], f4_domain, None)
    l11 = Ifelse('steps', var(2.0), fself, l12, l13, None)
    l10 = Assign(['steps'], f_add_one_domain, l11)
    l9_3 = Assign(['y2', 'v2'], f_add_domain, l10)
    l9_2 = Assign(['x2', 'v2'], f_add_domain, l9_3)
    l9_1 = Assign(['y1', 'v1'], f_add_domain, l9_2)
    l9 = Assign(['x1', 'v1'], f_sub_domain, l9_1)

    l8 = Ifelse('stage', var(2.0), fself, l9, l15, None)

    l28 = Assign(['i'], f_add_one_domain, None)
    l28_max = Assign(['x_max', 'x'], f_max_domain, l28)
    l28_min = Assign(['x_min', 'x'], f_min_domain, l28_max)

    l27 = Assign(['x', 'x_dist', 'y_dist'], f_update_dist_domain, l28_min) # cal dist(safe property) # TODO:use all left/right
    l27_4 = Assign(['y_dist'], f_square_domain, l27)
    l27_3 = Assign(['x_dist'], f_square_domain, l27_4)
    l27_2 = Assign(['y_dist', 'y1', 'y2'], f_dist_sub_domain, l27_3)
    l27_1 = Assign(['x_dist', 'x1', 'x2'], f_dist_sub_domain, l27_2)

    l1 = Ifelse('stage', var(1.0), fself, l2, l8, l27_1)
    l00 = Assign(['res'], f_theta_domain(Theta), None)
    l0 = WhileSample('i', var(50), l1, l00)
    
    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    l7 = AssignPoint(['stage'], f7, None)
    l5 = AssignPoint(['steps'], f5, None)
    l4 = AssignPoint(['stage'], f4, l5) # stage = 1.5, LEFT

    l3 = IfelsePoint('dist', Theta, fself, l4, l7, None)

    l2_1 = AssignPoint(['dist', 'x_dist', 'y_dist'], f_cal_dist, l3) # cal dist(safe property) # TODO:use all left/right
    l2_04 = AssignPoint(['y_dist'], f_square, l2_1)
    l2_03 = AssignPoint(['x_dist'], f_square, l2_04)
    l2_02 = AssignPoint(['y_dist', 'y1', 'y2'], f_dist_sub, l2_03)
    l2_01 = AssignPoint(['x_dist', 'x1', 'x2'], f_dist_sub, l2_02)
    l2_0 = AssignPoint(['x2', 'v2'], f_add, l2_01)
    l2 = AssignPoint(['y1', 'v1'], f_add, l2_0)

    l26 = AssignPoint(['stage'], f7, None)
    l25 = AssignPoint(['stage'], f20, None)
    l24 = IfelsePoint('steps', var(2.0), fself, l25, l26, None)

    l23 = AssignPoint(['steps'], f_add_one, l24)
    l22_3 = AssignPoint(['y2', 'v2'], f_sub, l23)
    l22_2 = AssignPoint(['x2', 'v2'], f_add, l22_3)
    l22_1 = AssignPoint(['y1', 'v1'], f_add, l22_2)
    l22 = AssignPoint(['x1', 'v1'], f_add, l22_1)

    l21 = AssignPoint(['steps'], f14, None)
    l20 = AssignPoint(['stage'], f20, l21)
    l19 = AssignPoint(['stage'], f13, None)
    l18 = IfelsePoint('steps', var(3.1), fself, l19, l20, None)

    l17 = AssignPoint(['steps'], f_add_one, l18)
    l16_1 = AssignPoint(['x2', 'v2'], f_add, l17)
    l16 = AssignPoint(['y1', 'v1'], f_add, l16_1)
    l15 = IfelsePoint('stage', var(3), fself, l16, l22, None)

    l14 = AssignPoint(['steps'], f14, None)
    l13 = AssignPoint(['stage'], f13, l14)
    l12 = AssignPoint(['stage'], f4, None)
    l11 = IfelsePoint('steps', var(2.0), fself, l12, l13, None)
    l10 = AssignPoint(['steps'], f_add_one, l11)
    l9_3 = AssignPoint(['y2', 'v2'], f_add, l10)
    l9_2 = AssignPoint(['x2', 'v2'], f_add, l9_3)
    l9_1 = AssignPoint(['y1', 'v1'], f_add, l9_2)
    l9 = AssignPoint(['x1', 'v1'], f_sub, l9_1)

    l8 = IfelsePoint('stage', var(2.0), fself, l9, l15, None)

    l28 = AssignPoint(['i'], f_add_one, None)
    l28_max = AssignPoint(['x_max', 'x'], f_max, l28)
    l28_min = AssignPoint(['x_min', 'x'], f_min, l28_max)

    l27 = AssignPoint(['x', 'x_dist', 'y_dist'], f_update_dist, l28_min) # cal dist(safe property) # TODO:use all left/right
    l27_4 = AssignPoint(['y_dist'], f_square, l27)
    l27_3 = AssignPoint(['x_dist'], f_square, l27_4)
    l27_2 = AssignPoint(['y_dist', 'y1', 'y2'], f_dist_sub, l27_3)
    l27_1 = AssignPoint(['x_dist', 'x1', 'x2'], f_dist_sub, l27_2)

    l1 = IfelsePoint('stage', var(1.0), fself, l2, l8, l27_1)

    l00 = AssignPoint(['res'], f_theta(Theta), None)
    l0 = WhileSimplePoint('i', var(50), l1, l00)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    l7 = AssignPointSmooth(['stage'], f7, None)
    l5 = AssignPointSmooth(['steps'], f5, None)
    l4 = AssignPointSmooth(['stage'], f4, l5) # stage = 1.5, LEFT

    l3 = IfelsePointSmooth('dist', Theta, fself, l4, l7, None)

    l2_1 = AssignPointSmooth(['dist', 'x_dist', 'y_dist'], f_cal_dist, l3) # cal dist(safe property) # TODO:use all left/right
    l2_04 = AssignPointSmooth(['y_dist'], f_square, l2_1)
    l2_03 = AssignPointSmooth(['x_dist'], f_square, l2_04)
    l2_02 = AssignPointSmooth(['y_dist', 'y1', 'y2'], f_dist_sub, l2_03)
    l2_01 = AssignPointSmooth(['x_dist', 'x1', 'x2'], f_dist_sub, l2_02)
    l2_0 = AssignPointSmooth(['x2', 'v2'], f_add, l2_01)
    l2 = AssignPointSmooth(['y1', 'v1'], f_add, l2_0)

    l26 = AssignPointSmooth(['stage'], f7, None)
    l25 = AssignPointSmooth(['stage'], f20, None)
    l24 = IfelsePointSmooth('steps', var(2.0), fself, l25, l26, None)

    l23 = AssignPointSmooth(['steps'], f_add_one, l24)
    l22_3 = AssignPointSmooth(['y2', 'v2'], f_sub, l23)
    l22_2 = AssignPointSmooth(['x2', 'v2'], f_add, l22_3)
    l22_1 = AssignPointSmooth(['y1', 'v1'], f_add, l22_2)
    l22 = AssignPointSmooth(['x1', 'v1'], f_add, l22_1)

    l21 = AssignPointSmooth(['steps'], f14, None)
    l20 = AssignPointSmooth(['stage'], f20, l21)
    l19 = AssignPointSmooth(['stage'], f13, None)
    l18 = IfelsePointSmooth('steps', var(3.1), fself, l19, l20, None)

    l17 = AssignPointSmooth(['steps'], f_add_one, l18)
    l16_1 = AssignPointSmooth(['x2', 'v2'], f_add, l17)
    l16 = AssignPointSmooth(['y1', 'v1'], f_add, l16_1)
    l15 = IfelsePointSmooth('stage', var(3), fself, l16, l22, None)

    l14 = AssignPointSmooth(['steps'], f14, None)
    l13 = AssignPointSmooth(['stage'], f13, l14)
    l12 = AssignPointSmooth(['stage'], f4, None)
    l11 = IfelsePointSmooth('steps', var(2.0), fself, l12, l13, None)
    l10 = AssignPointSmooth(['steps'], f_add_one, l11)
    l9_3 = AssignPointSmooth(['y2', 'v2'], f_add, l10)
    l9_2 = AssignPointSmooth(['x2', 'v2'], f_add, l9_3)
    l9_1 = AssignPointSmooth(['y1', 'v1'], f_add, l9_2)
    l9 = AssignPointSmooth(['x1', 'v1'], f_sub, l9_1)

    l8 = IfelsePointSmooth('stage', var(2.0), fself, l9, l15, None)

    l28 = AssignPointSmooth(['i'], f_add_one, None)
    l28_max = AssignPointSmooth(['x_max', 'x'], f_max_domain, l28)
    l28_min = AssignPointSmooth(['x_min', 'x'], f_min_domain, l28_max)

    l27 = AssignPointSmooth(['x', 'x_dist', 'y_dist'], f_update_dist, l28_min) # cal dist(safe property) # TODO:use all left/right
    l27_4 = AssignPointSmooth(['y_dist'], f_square, l27)
    l27_3 = AssignPointSmooth(['x_dist'], f_square, l27_4)
    l27_2 = AssignPointSmooth(['y_dist', 'y1', 'y2'], f_dist_sub, l27_3)
    l27_1 = AssignPointSmooth(['x_dist', 'x1', 'x2'], f_dist_sub, l27_2)

    l1 = IfelsePointSmooth('stage', var(1.0), fself, l2, l8, l27_1)
    l00 = AssignPointSmooth(['res'], f_theta(Theta), None)
    l0 = WhileSimplePointSmooth('i', var(50), l1, l00)
    # l00 = Assign(['res'], f_theta(Theta), None)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


