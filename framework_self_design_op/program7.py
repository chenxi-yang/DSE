import domain
from constants import *
from helper import * 
from point_interpretor import *

if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    if DOMAIN == "interval":
        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['x1'] = domain.Interval(x_l[0], x_r[0])
            symbol_table['y'] = domain.Interval(x_l[1], x_r[1])
            symbol_table['z'] = domain.Interval(x_l[2], x_r[2])
            symbol_table['omega1'] = domain.Interval(x_l[3], x_r[3])
            symbol_table['omega2'] = domain.Interval(x_l[4], x_r[4])
            symbol_table['tau'] = domain.Interval(x_l[5], x_r[5])
            symbol_table['stage'] = domain.Interval(0.5, 0.5)
            symbol_table['t'] = domain.Interval(0.0, 0.0) 

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
            symbol_table['x1'] = domain.Interval(x_l[0], x_r[0]).getZonotope()
            symbol_table['y'] = domain.Interval(x_l[1], x_r[1]).getZonotope()
            symbol_table['z'] = domain.Interval(x_l[2], x_r[2]).getZonotope()
            symbol_table['omega1'] = domain.Interval(x_l[3], x_r[3]).getZonotope()
            symbol_table['omega2'] = domain.Interval(x_l[4], x_r[4]).getZonotope()
            symbol_table['tau'] = domain.Interval(x_l[5], x_r[5]).getZonotope()
            symbol_table['stage'] = domain.Interval(0.5, 0.5).getZonotope()
            symbol_table['t'] = domain.Interval(0.0, 0.0) .getZonotope()

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
    symbol_table['x1'] = var(x[0])
    symbol_table['y'] = var(x[1])
    symbol_table['z'] = var(x[2])
    symbol_table['omega1'] = var(x[3])
    symbol_table['omega2'] = var(x[4])
    symbol_table['tau'] = var(x[5])
    symbol_table['stage'] = var(0.5)
    symbol_table['t'] = var(0.0)

    symbol_table['res'] = var(0.0)
    symbol_table['x_min'] = P_INFINITY
    symbol_table['x_max'] = N_INFINITY
    symbol_table['probability'] = var(1.0)
    symbol_table['explore_probability'] = var(1.0)

    return symbol_table


omega = var(3.14)

ax = var(1.0)
ay = var(1.2)
az = var(0.8)

delta_t = var(0.1)

# function in ifelse condition
def fself(x):
    return x

# function in assign
def f_stage_1(x):
    # print('f_stage_1')
    return var(0.5)
def f_stage_1_domain(x):
    return x[0].setValue(var(0.5))
def f_stage_2(x):
    # print('f_stage_2')
    return var(1.5)
def f_stage_2_domain(x):
    return x[0].setValue(var(1.5))
def f_stage_3(x):
    return var(2.5)
def f_stage_3_domain(x):
    return x[0].setValue(var(2.5))
def f_stage_4(x):
    return var(3.5)
def f_stage_4_domain(x):
    return x[0].setValue(var(3.5))
def f_add_one(x):
    return x[0].add(var(1.0))
def f_add_one_domain(x):
    return x[0].add(var(1.0))
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
    return x[1]
def f_equal_domain(x):
    return x[1]

# in stage 1
def f2(x):
    return x[0].add(var(-1.0).mul(ax).mul(torch.sin(omega.mul(x[1]))).mul(delta_t))
def f2_domain(x):
    return x[0].add(((x[1].mul(omega).sin()).mul(var(-1.0).mul(ax))).mul(delta_t))
def f3(x):
    return x[0].add(var(-1.0).mul(ay).mul(torch.sin((x[1].add(var(1.0))).mul(x[2])).mul(torch.sin(x[3]).mul(var(2.0)))).mul(delta_t))
def f3_domain(x):
    return x[0].add(((((x[1].add(var(1.0))).mul(x[2])).sin().mul((x[3]).sin())).mul(var(-2.0).mul(ay))).mul(delta_t))
def f4(x):
    return x[0].add(var(-1.0).mul(az).mul(torch.sin((x[1].add(var(1.0))).mul(x[2])).mul(torch.cos(x[3]).mul(var(2.0)))).mul(delta_t))
def f4_domain(x):
    return x[0].add(((((x[1].add(var(1.0))).mul(x[2])).sin().mul((x[3]).cos())).mul(var(-2.0).mul(az))).mul(delta_t))
def f5(x):
    return x[0].add((var(-1.0).mul(var(0.5).mul(x[0]))).mul(delta_t))
def f5_domain(x):
    return x[0].add((x[0].mul(var(-0.5))).mul(delta_t))
def f6(x):
    return x[0].add((var(-1.0).mul(x[0])).mul(delta_t))
def f6_domain(x):
    return x[0].add((x[0].mul(var(-1.0))).mul(delta_t))
def f7(x):
    return x[0].add(var(1.0).mul(delta_t))
def f7_domain(x):
    return x[0].add(var(1.0).mul(delta_t))
# in stage1 jmp
def f11(x):
    return var(0.0)
def f11_domain(x):
    return x[0].setValue(var(0.0))
def f12(x):
    return x[0]
def f12_domain(x):
    return x[0]
def f13(x):
    return x[0].mul(var(0.2))
def f13_domain(x):
    return x[0].mul(var(0.2))
def f14(x):
    return var(1.5)
def f14_domain(x):
    return x[0].setValue(var(1.5))
def f15(x):
    return var(1.0)
def f15_domain(x):
    return x[0].setValue(var(1.0))


# in stage 2
def f17(x):
    y = x[0].add((var(-1.0).mul(ax).mul(torch.sin(omega.mul(x[1])))).mul(delta_t))
    return y
def f17_domain(x):
    return x[0].add(((x[1].mul(omega).sin()).mul(var(-1.0).mul(ax))).mul(delta_t))
def f18(x):
    return x[0].add((var(-1.0).mul(ay).mul(torch.sin((x[1].add(var(1.0))).mul(x[2]))).mul(torch.sin(x[3]).mul(var(2.0)))).mul(delta_t))
def f18_domain(x):
    return x[0].add(((((x[1].add(var(1.0))).mul(x[2])).sin().mul((x[3]).sin())).mul(var(-2.0).mul(ay))).mul(delta_t))
def f19(x):
    return x[0].add((var(-1.0).mul(az).mul(torch.sin((var(2.0).sub(x[1])).mul(x[2]))).mul(torch.sin(x[3])).mul(var(2.0))).mul(delta_t))
def f19_domain(x):
    return x[0].add(((((x[1].sub_r(var(2.0))).mul(x[2])).sin().mul((x[3]).sin())).mul(var(-2.0).mul(az))).mul(delta_t))
def f20(x):
    return x[0].add((var(-1.0).mul(x[0])).mul(delta_t))
def f20_domain(x):
    return x[0].add((x[0].mul(var(-1.0))).mul(delta_t))
def f21(x):
    return x[0].add((var(-1.0).mul(x[0])).mul(delta_t))
def f21_domain(x):
    return x[0].add((x[0].mul(var(-1.0))).mul(delta_t))
def f22(x):
    return x[0].add(var(1.0).mul(delta_t))
def f22_domain(x):
    return x[0].add(var(1.0).mul(delta_t))
# in stage2 jmp
def f26(x):
    return var(0.0)
def f26_domain(x):
    return x[0].setValue(var(0.0))
def f27(x):
    return var(0.2).mul(x[0])
def f27_domain(x):
    return x[0].mul(var(0.2))
def f28(x):
    return var(0.5).mul(x[0])
def f28_domain(x):
    return x[0].mul(var(0.5))
def f29(x):
    return x[0]
def f29_domain(x):
    return x[0]
def f30(x):
    return torch.sin(x[0])
def f30_domain(x):
    return x[0].sin()
def f31(x):
    return var(-1.0).mul(x[0])
def f31_domain(x):
    return x[0].mul(var(-1.0))


# in stage3
def f32(x):
    # print('f32')
    return x[0].add((var(-1.0).mul(ax.mul(torch.sin(omega.mul(x[1]))))).mul(delta_t))
def f32_domain(x):
    return x[0].add(((x[1].mul(omega).sin()).mul(var(-1.0).mul(ax))).mul(delta_t))
def f33(x):
    return x[0].add((var(-1.0).mul(ay.mul(torch.sin((x[1].add(var(1.0))).mul(x[2])).mul(torch.sin(x[3])).mul(var(2.0))))).mul(delta_t))
def f33_domain(x):
    return x[0].add(((((x[1].add(var(1.0))).mul(x[2])).sin().mul((x[3]).sin())).mul(var(-2.0).mul(ay))).mul(delta_t))
def f34(x):
    return x[0].add((var(-1.0).mul(az).mul(torch.sin((x[1].add(var(2.0))).mul(x[2])).mul(torch.cos(x[3]).mul(var(2.0))))).mul(delta_t))
def f34_domain(x):
    return x[0].add(((((x[1].add(var(2.0))).mul(x[2])).sin().mul((x[3]).cos())).mul(var(-2.0).mul(az))).mul(delta_t))
def f35(x):
    return x[0].add((var(-0.5).mul(x[0])).mul(delta_t))
def f35_domain(x):
    return x[0].add((x[0].mul(var(-0.5))).mul(delta_t))
def f36(x):
    return x[0].add((var(-1.0).mul(x[0])).mul(delta_t))
def f36_domain(x):
    return x[0].add((x[0].mul(var(-1.0))).mul(delta_t))
def f37(x):
    return x[0].add(var(1.0).mul(delta_t))
def f37_domain(x):
    return  x[0].add(var(1.0).mul(delta_t))
# in stage3 jmp
def f41(x):
    return var(0.0)
def f41_domain(x):
    return x[0].setValue(var(0.0))
def f42(x):
    return var(1.0)
def f42_domain(x):
    return x[0].setValue(var(1.0))
def f43(x):
    return var(1.0)
def f43_domain(x):
    return x[0].setValue(var(1.0))


def construct_syntax_tree(Theta):
    l47 = Assign(['res', 'y'], f_equal_domain, None)
    l46 = Assign(['x_min', 'x1'], f_min_domain, l47)
    l45 = Assign(['x_max', 'x1'], f_max_domain, l46)
    l44 = Assign(['t'], f_add_one_domain, l45)

    l43 = Assign(['omega2'], f43_domain, None)
    l42 = Assign(['omega1'], f42_domain, l43)
    l41 = Assign(['tau'], f41_domain, l42)
    l40 = Assign(['stage'], f_stage_1_domain, l41)
    l39 = Assign(['stage'], f_stage_3_domain, None)
    l38 = Ifelse('tau', var(5.0), fself, l39, l40, None)
    l38_1 = Assign(['stage'], f_stage_4_domain, None)
    l38_0 = Ifelse('tau', Theta, fself, l38, l38_1, None)
    l37 = Assign(['tau'], f37_domain, l38_0)
    l36 = Assign(['omega2'], f36_domain, l37)
    l35 = Assign(['omega1'], f35_domain, l36)
    l34 = Assign(['z', 'omega2', 'tau', 'omega1'], f34_domain, l35)
    l33 = Assign(['y', 'omega1', 'tau', 'omega2'], f33_domain, l34)
    l32 = Assign(['x1', 'tau'], f32_domain, l33)

    l31 = Assign(['omega2'], f31_domain, None)
    l30 = Assign(['omega1'], f30_domain, l31)
    l29 = Assign(['z'], f29_domain, l30)
    l28 = Assign(['y'], f28_domain, l29)
    l27 = Assign(['x1'], f27_domain, l28)
    l26 = Assign(['tau'], f26_domain, l27)
    l25 = Assign(['stage'], f_stage_3_domain, l26)
    l24 = Assign(['stage'], f_stage_2_domain, None)
    l23 = Ifelse('tau', var(8.0), fself, l24, l25, None)
    l22 = Assign(['tau'], f22_domain, l23)
    l21 = Assign(['omega2'], f21_domain, l22)
    l20 = Assign(['omega1'], f20_domain, l21)
    l19 = Assign(['z', 'omega2', 'tau', 'omega1'], f19_domain, l20)
    l18 = Assign(['y', 'omega1', 'tau', 'omega2'], f18_domain, l19)
    l17 = Assign(['x1', 'tau'], f17_domain, l18)
    l16 = Ifelse('stage', var(2.0), fself, l17, l32, None)

    l15 = Assign(['omega2'], f15_domain, None)
    l14 = Assign(['omega1'], f14_domain, l15)
    l13 = Assign(['y'], f13_domain, l14)
    l12 = Assign(['x1'], f12_domain, l13)
    l11 = Assign(['tau'], f11_domain, l12)
    l10 = Assign(['stage'], f_stage_2_domain, l11)
    l9 = Assign(['stage'], f_stage_1_domain, None)
    l8 = Ifelse('tau', Theta, fself, l9, l10, None)
    l7 = Assign(['tau'], f7_domain, l8)
    l6 = Assign(['omega2'], f6_domain, l7)
    l5 = Assign(['omega1'], f5_domain, l6)
    l4 = Assign(['z', 'omega2', 'tau', 'omega1'], f4_domain, l5)
    l3 = Assign(['y', 'omega1', 'tau', 'omega2'], f3_domain, l4)
    l2 = Assign(['x1', 'tau'], f2_domain, l3)
    l1 = Ifelse('stage', var(1.0), fself, l2, l16, l44)

    l0 = WhileSample('stage', var(3.0), l1, None)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):
    l47 = AssignPoint(['res', 'y'], f_equal, None)
    l46 = AssignPoint(['x_min', 'x1'], f_min, l47)
    l45 = AssignPoint(['x_max', 'x1'], f_max, l46)
    l44 = AssignPoint(['t'], f_add_one, l45)

    l43 = AssignPoint(['omega2'], f43, None)
    l42 = AssignPoint(['omega1'], f42, l43)
    l41 = AssignPoint(['tau'], f41, l42)
    l40 = AssignPoint(['stage'], f_stage_1, l41)
    l39 = AssignPoint(['stage'], f_stage_3, None)
    l38 = IfelsePoint('tau', var(5.0), fself, l39, l40, None)
    l38_1 = AssignPoint(['stage'], f_stage_4, None)
    l38_0 = IfelsePoint('tau', Theta, fself, l38, l38_1, None)
    l37 = AssignPoint(['tau'], f37, l38_0)
    l36 = AssignPoint(['omega2'], f36, l37)
    l35 = AssignPoint(['omega1'], f35, l36)
    l34 = AssignPoint(['z', 'omega2', 'tau', 'omega1'], f34, l35)
    l33 = AssignPoint(['y', 'omega1', 'tau', 'omega2'], f33, l34)
    l32 = AssignPoint(['x1', 'tau'], f32, l33)

    l31 = AssignPoint(['omega2'], f31, None)
    l30 = AssignPoint(['omega1'], f30, l31)
    l29 = AssignPoint(['z'], f29, l30)
    l28 = AssignPoint(['y'], f28, l29)
    l27 = AssignPoint(['x1'], f27, l28)
    l26 = AssignPoint(['tau'], f26, l27)
    l25 = AssignPoint(['stage'], f_stage_3, l26)
    l24 = AssignPoint(['stage'], f_stage_2, None)
    l23 = IfelsePoint('tau', var(8.0), fself, l24, l25, None)
    l22 = AssignPoint(['tau'], f22, l23)
    l21 = AssignPoint(['omega2'], f21, l22)
    l20 = AssignPoint(['omega1'], f20, l21)
    l19 = AssignPoint(['z', 'omega2', 'tau', 'omega1'], f19, l20)
    l18 = AssignPoint(['y', 'omega1', 'tau', 'omega2'], f18, l19)
    l17 = AssignPoint(['x1', 'tau'], f17, l18)
    l16 = IfelsePoint('stage', var(2.0), fself, l17, l32, None)

    l15 = AssignPoint(['omega2'], f15, None)
    l14 = AssignPoint(['omega1'], f14, l15)
    l13 = AssignPoint(['y'], f13, l14)
    l12 = AssignPoint(['x1'], f12, l13)
    l11 = AssignPoint(['tau'], f11, l12)
    l10 = AssignPoint(['stage'], f_stage_2, l11)
    l9 = AssignPoint(['stage'], f_stage_1, None)
    l8 = IfelsePoint('tau', Theta, fself, l9, l10, None)
    l7 = AssignPoint(['tau'], f7, l8)
    l6 = AssignPoint(['omega2'], f6, l7)
    l5 = AssignPoint(['omega1'], f5, l6)
    l4 = AssignPoint(['z', 'omega2', 'tau', 'omega1'], f4, l5)
    l3 = AssignPoint(['y', 'omega1', 'tau', 'omega2'], f3, l4)
    l2 = AssignPoint(['x1', 'tau'], f2, l3)
    l1 = IfelsePoint('stage', var(1.0), fself, l2, l16, l44)

    l0 = WhilePoint('stage', var(3.0), l1, None)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):
    l47 = AssignPointSmooth(['res', 'y'], f_equal, None)
    l46 = AssignPointSmooth(['x_min', 'x1'], f_min, l47)
    l45 = AssignPointSmooth(['x_max', 'x1'], f_max, l46)
    l44 = AssignPointSmooth(['t'], f_add_one, l45)

    l43 = AssignPointSmooth(['omega2'], f43, None)
    l42 = AssignPointSmooth(['omega1'], f42, l43)
    l41 = AssignPointSmooth(['tau'], f41, l42)
    l40 = AssignPointSmooth(['stage'], f_stage_1, l41)
    l39 = AssignPointSmooth(['stage'], f_stage_3, None)
    l38 = IfelsePointSmooth('tau', var(5.0), fself, l39, l40, None)
    l38_1 = AssignPointSmooth(['stage'], f_stage_4, None)
    l38_0 = IfelsePointSmooth('tau', Theta, fself, l38, l38_1, None)
    l37 = AssignPointSmooth(['tau'], f37, l38_0)
    l36 = AssignPointSmooth(['omega2'], f36, l37)
    l35 = AssignPointSmooth(['omega1'], f35, l36)
    l34 = AssignPointSmooth(['z', 'omega2', 'tau', 'omega1'], f34, l35)
    l33 = AssignPointSmooth(['y', 'omega1', 'tau', 'omega2'], f33, l34)
    l32 = AssignPointSmooth(['x1', 'tau'], f32, l33)

    l31 = AssignPointSmooth(['omega2'], f31, None)
    l30 = AssignPointSmooth(['omega1'], f30, l31)
    l29 = AssignPointSmooth(['z'], f29, l30)
    l28 = AssignPointSmooth(['y'], f28, l29)
    l27 = AssignPointSmooth(['x1'], f27, l28)
    l26 = AssignPointSmooth(['tau'], f26, l27)
    l25 = AssignPointSmooth(['stage'], f_stage_3, l26)
    l24 = AssignPointSmooth(['stage'], f_stage_2, None)
    l23 = IfelsePointSmooth('tau', var(8.0), fself, l24, l25, None)
    l22 = AssignPointSmooth(['tau'], f22, l23)
    l21 = AssignPointSmooth(['omega2'], f21, l22)
    l20 = AssignPointSmooth(['omega1'], f20, l21)
    l19 = AssignPointSmooth(['z', 'omega2', 'tau', 'omega1'], f19, l20)
    l18 = AssignPointSmooth(['y', 'omega1', 'tau', 'omega2'], f18, l19)
    l17 = AssignPointSmooth(['x1', 'tau'], f17, l18)
    l16 = IfelsePointSmooth('stage', var(2.0), fself, l17, l32, None)

    l15 = AssignPointSmooth(['omega2'], f15, None)
    l14 = AssignPointSmooth(['omega1'], f14, l15)
    l13 = AssignPointSmooth(['y'], f13, l14)
    l12 = AssignPointSmooth(['x1'], f12, l13)
    l11 = AssignPointSmooth(['tau'], f11, l12)
    l10 = AssignPointSmooth(['stage'], f_stage_2, l11)
    l9 = AssignPointSmooth(['stage'], f_stage_1, None)
    l8 = IfelsePointSmooth('tau', Theta, fself, l9, l10, None)
    l7 = AssignPointSmooth(['tau'], f7, l8)
    l6 = AssignPointSmooth(['omega2'], f6, l7)
    l5 = AssignPointSmooth(['omega1'], f5, l6)
    l4 = AssignPointSmooth(['z', 'omega2', 'tau', 'omega1'], f4, l5)
    l3 = AssignPointSmooth(['y', 'omega1', 'tau', 'omega2'], f3, l4)
    l2 = AssignPointSmooth(['x1', 'tau'], f2, l3)
    l1 = IfelsePointSmooth('stage', var(1.0), fself, l2, l16, l44)

    l0 = WhilePointSmooth('stage', var(3.0), l1, None)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict