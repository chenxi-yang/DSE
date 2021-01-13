import domain
from constants import *
from helper import * 
from point_interpretor import *

# atrial fibrillation sample loop

if MODE == 5:
    from disjunction_of_intervals_interpretor_loop_importance_sampling import *

    if DOMAIN == "interval":
        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['u'] = domain.Interval(x_l[0], x_r[0])
            symbol_table['w'] = domain.Interval(x_l[1], x_r[1])
            symbol_table['v'] = domain.Interval(x_l[2], x_r[2])
            symbol_table['s'] = domain.Interval(x_l[3], x_r[3])
            symbol_table['stage'] = domain.Interval(0.5, 0.5)
            symbol_table['t'] = domain.Interval(0.0, 0.0) 

            symbol_table['res'] = domain.Interval(0.0, 0.0)
            symbol_table['x_max'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())
            symbol_table['x_min'] = domain.Interval(P_INFINITY.data.item(), P_INFINITY.data.item())
            symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())])
            symbol_table['probability'] = var(1.0)
            symbol_table['explore_probability'] = var(1.0)

            symbol_table_list.append(symbol_table)

            return symbol_table_list
    
    if DOMAIN == "zonotope":
        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['u'] = domain.Interval(x_l[0], x_r[0]).getZonotope()
            symbol_table['w'] = domain.Interval(x_l[1], x_r[1]).getZonotope()
            symbol_table['v'] = domain.Interval(x_l[2], x_r[2]).getZonotope()
            symbol_table['s'] = domain.Interval(x_l[3], x_r[3]).getZonotope()
            symbol_table['stage'] = domain.Interval(0.5, 0.5).getZonotope()
            symbol_table['t'] = domain.Interval(0.0, 0.0).getZonotope()

            symbol_table['res'] = domain.Interval(0.0, 0.0).getZonotope()
            symbol_table['x_max'] = domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item()).getZonotope()
            symbol_table['x_min'] = domain.Interval(P_INFINITY.data.item(), P_INFINITY.data.item()).getZonotope()

            # calculate in the domain of interval
            symbol_table['x_memo_list'] = list([domain.Interval(N_INFINITY.data.item(), N_INFINITY.data.item())])
            symbol_table['probability'] = var(1.0)
            symbol_table['explore_probability'] = var(1.0)

            symbol_table_list.append(symbol_table)

            return symbol_table_list
        

 
def initialization_point(x):

    symbol_table = dict()
    symbol_table['u'] = var(x[0])
    symbol_table['w'] = var(x[1])
    symbol_table['v'] = var(x[2])
    symbol_table['s'] = var(x[3])
    symbol_table['stage'] = var(0.5)
    symbol_table['t'] = var(0.0)

    symbol_table['res'] = var(0.0)
    symbol_table['x_max'] = N_INFINITY
    symbol_table['x_min'] = P_INFINITY
    symbol_table['probability'] = var(1.0)
    symbol_table['explore_probability'] = var(1.0)

    return symbol_table


epi_tvp = var(1.4506)
epi_tv1m = var(60.0)
epi_tv2m = var(1150.0)
epi_twp = var(200.0)
epi_tw1m = var(60.0)
epi_tw2m = var(15.0)
epi_ts1 = var(2.7342)
epi_ts2 = var(16.0)
epi_tfi = var(0.11)
epi_to1 = var(0.0055)# var(0.0055)
epi_to2 = var(6)
epi_tso1 = var(30.0181)
epi_tso2 = var(0.9957)
epi_tsi = var(1.8875)
epi_twinf = var(0.07)
epi_thv = var(0.3)
epi_thvm = var(0.006)
epi_thvinf = var(0.006)
epi_thw = var(0.13)
epi_thwinf = var(0.006)
epi_thso = var(0.13)
epi_thsi = var(0.13)
epi_tho = var(0.006)

epi_kwm = var(65.0)
epi_ks = var(2.0994)
epi_kso = var(2.0458)
epi_uwm = var(0.03)
epi_us = var(0.9087)
epi_uo = var(0.0)
epi_uu = var(1.55)
epi_uso = var(0.65)

delta_t = var(0.001)

jfi1 = var(0.0)
jsi1 = var(0.0)
jfi2 = var(0.0)
jsi2 = var(0.0)
jfi3 = var(0.0)
stim = var(1.0)

def jso1(u):
    return u.div(epi_to1)
def jso1_domain(u):
    return u.mul(var(1.0).div(epi_to1))

def jso2(u):
    return u.div(epi_to2)
def jso2_domain(u):
    return u.mul(var(1.0).div(epi_to2))

def jso3(u):
    return var(1.0).div(epi_tso1.add((epi_tso2.sub(epi_tso1).mul(var(1.0).div(var(1.0).add(torch.exp(var(-1.0).mul(var(2.0).mul(epi_kso).mul(u.sub(epi_uso))))))))))
def jso3_domain(u):
    return (((((u.sub_l(epi_uso)).mul(var(-1.0).mul(var(2.0).mul(epi_kso)))).exp()).add(var(1.0))).div(epi_tso2.sub(epi_tso1).mul(var(1.0))).add(epi_tso1)).div(var(1.0))
def jsi3(w, s):
    return var(-1.0).mul(w.mul(s)).div(epi_tsi)
def jsi3_domain(w, s):
    return w.mul(s).mul(var(-1.0)).mul(var(1.0).div(epi_tsi))

def jfi4(u, v):
    return var(-1.0).mul(v.mul(u.sub(epi_thv))).mul(epi_uu.sub(u)).div(epi_tfi)
def jfi4_domain(u, v):
    return v.mul(u.sub_l(epi_thv)).mul(u.sub_r(epi_uu)).mul(var(-1.0).div(epi_tfi))
def jso4(u):
    return var(1.0).div(epi_tso1.add((epi_tso2.sub(epi_tso1)).mul(var(1.0).div(var(1.0).add(torch.exp(var(-2.0).mul(epi_kso).mul(u.sub(epi_uso))))))))
def jso4_domain(u):
    return (((((((u.sub_l(epi_uso)).mul(var(-2.0).mul(epi_kso))).exp()).add(var(1.0))).div(var(1.0))).mul(epi_tso2.sub(epi_tso1))).add(epi_tso1)).div(var(1.0))

def jsi4(w, s):
    return var(-1.0).mul(w.mul(s)).div(epi_tsi)
def jsi4_domain(w, s):
    return w.mul(s).mul(var(-1.0).div(epi_tsi))

# function in ifelse condition
# get a value
def fself(x):
    return x

# function in assign
def f_stage_1(x):
    return var(0.5)
def f_stage_1_domain(x):
    return x[0].setValue(var(0.5))
def f_stage_2(x):
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
    return torch.min(x[0], x[1])
def f_min_domain(x):
    return x[0].min(x[1])
def f_equal(x):
    return x[1]
def f_equal_domain(x):
    return x[1]

# in stage 1
def f2(x):
    return x[0].add(stim.sub(jfi1).sub(jso1(x[0]).add(jsi1)).mul(delta_t))
def f2_domain(x):
    # print(type(x[0]))
    # print(type(jso1_domain(x[0])))
    # exit(0)
    return x[0].add(jso1_domain(x[0]).add(jsi1).sub_r(stim.sub(jfi1)).mul(delta_t))
def f3(x):
    return x[0].add((var(1.0).sub(x[1].div(epi_twinf)).sub(x[0])).div(epi_tw1m.add((epi_tw2m.sub(epi_tw1m)).mul(var(1.0).div(var(1.0).add(torch.exp(var(-2.0).mul(epi_kwm).mul(x[1].sub(epi_uwm)))))))).mul(delta_t))
def f3_domain(x):
    return x[0].add(((((((((x[1].sub_l(epi_uwm)).mul(var(-2.0).mul(epi_kwm))).exp()).add(var(1.0))).div(var(1.0))).mul(epi_tw2m.sub(epi_tw1m))).add(epi_tw1m)).div(x[0].sub_r(x[1].mul(var(1.0).mul(epi_twinf)).sub_r(var(1.0))))).mul(delta_t))
def f4(x):
    return x[0].add((var(1.0).sub(x[0])).div(epi_tv1m).mul(delta_t))
def f4_domain(x):
    return x[0].add(((x[0].sub_r(var(1.0))).mul(var(1.0).div(epi_tv1m))).mul(delta_t))
def f5(x):
    return x[0].add(((var(1.0).div(var(1.0).add(torch.exp(var(-2.0).mul(epi_ks).mul(x[1].sub(epi_us)))))).sub(x[0])).div(epi_ts1).mul(delta_t))
def f5_domain(x):
    return x[0].add(((x[0].sub_r(((((x[1].sub_l(epi_us).mul(var(-2.0).mul(epi_ks)))).exp()).add(var(1.0))).div(var(1.0)))).mul(var(1.0).div(epi_ts1))).mul(delta_t))

# in stage 2
def f10(x):
    return x[0].add(stim.sub(jfi2).sub(jso2(x[0]).add(jsi2)).mul(delta_t))
def f10_domain(x):
    return x[0].add(jso2_domain(x[0]).add(jsi2).sub_r(stim.sub(jfi2)).mul(delta_t))
def f11(x):
    return x[0].add((var(0.94).sub(x[0])).div(epi_tw1m.add((epi_tw2m.sub(epi_tw1m)).mul(var(1.0).div(var(1.0).add(torch.exp(var(-2.0).mul(epi_kwm).mul(x[1].sub(epi_uwm)))))))).mul(delta_t))
def f11_domain(x):
    return x[0].add(((((((((x[1].sub_l(epi_uwm)).mul(var(-2.0).mul(epi_kwm))).exp()).add(var(1.0))).div(var(1.0))).mul(epi_tw2m.sub(epi_tw1m))).add(epi_tw1m)).div(x[0].sub_r(var(0.94)))).mul(delta_t))
def f12(x):
    return x[0].add(var(-1.0).mul(x[0]).div(epi_tv2m).mul(delta_t))
def f12_domain(x):
    return x[0].add((x[0].mul(var(-1.0).mul(epi_tv2m))).mul(delta_t))
def f13(x):
    return x[0].add(((var(1.0).div(var(1.0).add(torch.exp(var(-2.0).mul(epi_ks).mul(x[1].sub(epi_us)))))).sub(x[0])).div(epi_ts1).mul(delta_t))
def f13_domain(x):
    return x[0].add(((x[0].sub_r(((((x[1].sub_l(epi_us).mul(var(-2.0).mul(epi_ks)))).exp()).add(var(1.0))).div(var(1.0)))).mul(var(1.0).div(epi_ts1))).mul(delta_t))

# in stage 3
def f18(x):
    return x[0].add(stim.sub(jfi3).sub(jso3(x[0]).add(jsi3(x[1], x[2]))).mul(delta_t))
def f18_domain(x):
    return x[0].add(jso3_domain(x[0]).add(jsi3_domain(x[1], x[2])).sub_r(stim.sub(jfi3)).mul(delta_t))
def f19(x):
    return x[0].add(var(-1.0).mul(x[0]).div(epi_twp).mul(delta_t))
def f19_domain(x):
    return x[0].add((x[0].mul(var(-1.0).div(epi_twp))).mul(delta_t))
def f20(x):
    return x[0].add(var(-1.0).mul(x[0]).div(epi_tv2m).mul(delta_t))
def f20_domain(x):
    return x[0].add((x[0].mul(var(-1.0).mul(epi_tv2m))).mul(delta_t))
def f21(x):
    return x[0].add(((var(1.0).div(var(1.0).add(torch.exp(var(-2.0).mul(epi_ks).mul(x[1].sub(epi_us)))))).sub(x[0])).div(epi_ts2).mul(delta_t))
def f21_domain(x):
    return x[0].add(((x[0].sub_r(((((x[1].sub_l(epi_us).mul(var(-2.0).mul(epi_ks)))).exp()).add(var(1.0))).div(var(1.0)))).mul(var(1.0).div(epi_ts2))).mul(delta_t))

# in stage 4
def f25(x):
    return x[0].add(stim.sub(jfi4(x[0], x[1])).sub(jso4(x[0]).add(jsi4(x[2], x[3]))).mul(delta_t))
def f25_domain(x):
    return x[0].add(jso4_domain(x[0]).add(jsi4_domain(x[2], x[3])).sub_r(jfi4_domain(x[0], x[1]).sub_r(stim)).mul(delta_t))
def f26(x):
    return x[0].add(var(-1.0).mul(x[0].div(epi_twp)).mul(delta_t))
def f26_domain(x):
    return x[0].add((x[0].mul(var(-1.0).div(epi_twp))).mul(delta_t))
def f27(x):
    return x[0].add(var(-1.0).mul(x[0].div(epi_tvp)).mul(delta_t))
def f27_domain(x):
    return x[0].add((x[0].mul(var(-1.0).mul(epi_tvp))).mul(delta_t))
def f28(x):
    return x[0].add(((var(1.0).div(var(1.0).add(torch.exp(var(-2.0).mul(epi_ks).mul(x[1].sub(epi_us)))))).sub(x[0])).div(epi_ts2).mul(delta_t))
def f28_domain(x):
    return x[0].add(((x[0].sub_r(((((x[1].sub_l(epi_us).mul(var(-2.0).mul(epi_ks)))).exp()).add(var(1.0))).div(var(1.0)))).mul(var(1.0).div(epi_ts2))).mul(delta_t))


def construct_syntax_tree(Theta):

    # set up t, x, res
    l35 = Assign(['res', 'u'], f_equal_domain, None)
    l34 = Assign(['x_min', 'w'], f_min_domain, l35)
    l33 = Assign(['x_max', 'w'], f_max_domain, l34)
    l32 = Assign(['t'], f_add_one_domain, l33)

    l31 = Assign(['stage'], f_stage_4_domain, None)
    l30 = Assign(['stage'], f_stage_3_domain, None)
    l29 = Ifelse('u', Theta, fself, l30, l31, None)
    l28 = Assign(['s', 'u'], f28_domain, l29)
    l27 = Assign(['v'], f27_domain, l28)
    l26 = Assign(['w'], f26_domain, l27)
    l25 = Assign(['u', 'v', 'w', 's'], f25_domain, l26)

    l24 = Assign(['stage'], f_stage_4_domain, None)
    l23 = Assign(['stage'], f_stage_3_domain, None)
    l22_1 = Assign(['stage'], f_stage_2_domain, None)
    # l22_0 = Ifelse('u', var(0.013), fself, l22_1, l23, None)
    l22 = Ifelse('u', Theta, fself, l23, l24, None)
    l21 = Assign(['s', 'u'], f21_domain, l22)
    l20 = Assign(['v'], f20_domain, l21)
    l19 = Assign(['w'], f19_domain, l20)
    l18 = Assign(['u', 'w', 's'], f18_domain, l19)
    l17 = Ifelse('stage', var(3.0), fself, l18, l25, None)

    l16 = Assign(['stage'], f_stage_3_domain, None)
    l15 = Assign(['stage'], f_stage_2_domain, None)
    l14_1 = Assign(['stage'], f_stage_1_domain, None)
    # l14_0 = Ifelse('u', var(0.005), fself, l14_1, l15, None)
    l14 = Ifelse('u', var(0.013), fself, l15, l16, None)
    l13 = Assign(['s', 'u'], f13_domain, l14)
    l12 = Assign(['v'], f12_domain, l13)
    l11 = Assign(['w', 'u'], f11_domain, l12)
    l10 = Assign(['u'], f10_domain, l11)
    l9 = Ifelse('stage', var(2.0), fself, l10, l17, None) # in model2

    l8 = Assign(['stage'], f_stage_2_domain, None)
    l7 = Assign(['stage'], f_stage_1_domain, None)
    l6 = Ifelse('u', var(0.006), fself, l7, l8, None)
    l5 = Assign(['s', 'u'], f5_domain, l6)
    l4 = Assign(['v'], f4_domain, l5)
    l3 = Assign(['w', 'u'], f3_domain, l4)
    l2 = Assign(['u'], f2_domain, l3)
    l1 = Ifelse('stage', var(1.0), fself, l2, l9, l32) # in model1

    l0 = WhileSample('t', var(8), l1, None)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    # set up t, x, res
    l35 = AssignPoint(['res', 'u'], f_equal, None)
    l34 = AssignPoint(['x_min', 'w'], f_min, l35)
    l33 = AssignPoint(['x_max', 'w'], f_max, l34)
    l32 = AssignPoint(['t'], f_add_one, l33)

    l31 = AssignPoint(['stage'], f_stage_4, None)
    l30 = AssignPoint(['stage'], f_stage_3, None)
    l29 = IfelsePoint('u', Theta, fself, l30, l31, None)
    l28 = AssignPoint(['s', 'u'], f28, l29)
    l27 = AssignPoint(['v'], f27, l28)
    l26 = AssignPoint(['w'], f26, l27)
    l25 = AssignPoint(['u', 'v', 'w', 's'], f25, l26)

    l24 = AssignPoint(['stage'], f_stage_4, None)
    l23 = AssignPoint(['stage'], f_stage_3, None)
    l22_1 = AssignPoint(['stage'], f_stage_2, None)
    # l22_0 = IfelsePoint('u', var(0.013), fself, l22_1, l23, None)
    l22 = IfelsePoint('u', Theta, fself, l23, l24, None) # the condition in the third stage
    l21 = AssignPoint(['s', 'u'], f21, l22)
    l20 = AssignPoint(['v'], f20, l21)
    l19 = AssignPoint(['w'], f19, l20)
    l18 = AssignPoint(['u', 'w', 's'], f18, l19)
    l17 = IfelsePoint('stage', var(3.0), fself, l18, l25, None)

    l16 = AssignPoint(['stage'], f_stage_3, None)
    l15 = AssignPoint(['stage'], f_stage_2, None)
    l14_1 = AssignPoint(['stage'], f_stage_1, None)
    # l14_0 = IfelsePoint('u', var(0.005), fself, l14_1, l15, None)
    l14 = IfelsePoint('u', var(0.013), fself, l15, l16, None)
    l13 = AssignPoint(['s', 'u'], f13, l14)
    l12 = AssignPoint(['v'], f12, l13)
    l11 = AssignPoint(['w', 'u'], f11, l12)
    l10 = AssignPoint(['u'], f10, l11)
    l9 = IfelsePoint('stage', var(2.0), fself, l10, l17, None) # in model2

    l8 = AssignPoint(['stage'], f_stage_2, None)
    l7 = AssignPoint(['stage'], f_stage_1, None)
    l6 = IfelsePoint('u', var(0.006), fself, l7, l8, None)
    l5 = AssignPoint(['s', 'u'], f5, l6)
    l4 = AssignPoint(['v'], f4, l5)
    l3 = AssignPoint(['w', 'u'], f3, l4)
    l2 = AssignPoint(['u'], f2, l3)
    l1 = IfelsePoint('stage', var(1.0), fself, l2, l9, l32) # in model1

    l0 = WhilePoint('stage', var(8), l1, None)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    # set up t, x, res
    l35 = AssignPointSmooth(['res', 'u'], f_equal, None)
    l34 = AssignPointSmooth(['x_min', 'w'], f_min, l35)
    l33 = AssignPointSmooth(['x_max', 'w'], f_max, l34)
    l32 = AssignPointSmooth(['t'], f_add_one, l33)

    l31 = AssignPointSmooth(['stage'], f_stage_4, None)
    l30 = AssignPointSmooth(['stage'], f_stage_3, None)
    l29 = IfelsePointSmooth('u', Theta, fself, l30, l31, None)
    l28 = AssignPointSmooth(['s', 'u'], f28, l29)
    l27 = AssignPointSmooth(['v'], f27, l28)
    l26 = AssignPointSmooth(['w'], f26, l27)
    l25 = AssignPointSmooth(['u', 'v', 'w', 's'], f25, l26)

    l24 = AssignPointSmooth(['stage'], f_stage_4, None)
    l23 = AssignPointSmooth(['stage'], f_stage_3, None)
    l22_1 = AssignPointSmooth(['stage'], f_stage_2, None)
    # l22_0 = IfelsePointSmooth('u', var(0.013), fself, l22_1, l23, None)
    l22 = IfelsePointSmooth('u', Theta, fself, l23, l24, None) # the condition in the third stage
    l21 = AssignPointSmooth(['s', 'u'], f21, l22)
    l20 = AssignPointSmooth(['v'], f20, l21)
    l19 = AssignPointSmooth(['w'], f19, l20)
    l18 = AssignPointSmooth(['u', 'w', 's'], f18, l19)
    l17 = IfelsePointSmooth('stage', var(3.0), fself, l18, l25, None)

    l16 = AssignPointSmooth(['stage'], f_stage_3, None)
    l15 = AssignPointSmooth(['stage'], f_stage_2, None)
    l14_1 = AssignPointSmooth(['stage'], f_stage_1, None)
    # l14_0 = IfelsePointSmooth('u', var(0.005), fself, l14_1, l15, None)
    l14 = IfelsePointSmooth('u', var(0.013), fself, l15, l16, None)
    l13 = AssignPointSmooth(['s', 'u'], f13, l14)
    l12 = AssignPointSmooth(['v'], f12, l13)
    l11 = AssignPointSmooth(['w', 'u'], f11, l12)
    l10 = AssignPointSmooth(['u'], f10, l11)
    l9 = IfelsePointSmooth('stage', var(2.0), fself, l10, l17, None) # in model2

    l8 = AssignPointSmooth(['stage'], f_stage_2, None)
    l7 = AssignPointSmooth(['stage'], f_stage_1, None)
    l6 = IfelsePointSmooth('u', var(0.006), fself, l7, l8, None)
    l5 = AssignPointSmooth(['s', 'u'], f5, l6)
    l4 = AssignPointSmooth(['v'], f4, l5)
    l3 = AssignPointSmooth(['w', 'u'], f3, l4)
    l2 = AssignPointSmooth(['u'], f2, l3)
    l1 = IfelsePointSmooth('stage', var(1.0), fself, l2, l9, l32) # in model1

    l0 = WhilePointSmooth('stage', var(8), l1, None)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict



