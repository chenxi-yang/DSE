
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
            symbol_table['lin'] = domain.Interval(0.0, 0.0)
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

            # symbol_table_list = split_symbol_table(symbol_table, ['x'], partition=10)

            return symbol_table_list
    
    if DOMAIN == "zonotope":
        def initialization(x_l, x_r):
            symbol_table_list = list()
            symbol_table = dict()
            symbol_table['i'] = domain.Interval(0, 0).getZonotope()
            symbol_table['x'] = domain.Interval(x_l[0], x_r[0]).getZonotope()
            symbol_table['lin'] = domain.Interval(x_l[0], x_r[0])
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
        # symbol_table_list = list()
        symbol_table = dict()
        symbol_table['i'] = domain.Interval(var(0.0), var(0.0))
        symbol_table['x'] = domain.Interval(var(x[0]),  var(x[0]))
        symbol_table['lin'] = domain.Interval(var(0.0), var(0.0))
        symbol_table['isOn'] = domain.Interval(var(0.0), var(0.0))

        symbol_table['res'] = domain.Interval(var(0.0), var(0.0))
        symbol_table['x_min'] = domain.Interval(P_INFINITY, P_INFINITY)
        symbol_table['x_max'] = domain.Interval(N_INFINITY, N_INFINITY)
        symbol_table['probability'] = var(1.0)
        symbol_table['explore_probability'] = var(1.0)

        symbol_table['target_res'] = var(y)
        # symbol_table_list.append(symbol_table)
    
    return symbol_table
    # return symbol_table_list


def initialization_point(x):
    symbol_table = dict()
    symbol_table['i'] = var(0.0)
    symbol_table['x'] = var(x[0])
    symbol_table['lin'] = var(0.0)
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
def fself_0(x):
    return x[0]
# def fself_add(x):
#     return x.add(var(2.0))

# function in assign
def f_max(x):
    # print(f"Debug: max, {x[0]}, {x[1]}")
    return torch.max(x[0], x[1])
def f_max_domain(Theta):
    def res(x):
        y = x[0].max(x[1])
        # if x[1].right.data.item() > x[1].left.data.item():
        #     print(f"DEBUG: f_max_domain, {torch.autograd.grad(y.left, Theta, retain_graph=True, allow_unused=True)}")
        return y
    return res

def f_min(x):
    # print(f"Debug: min, {x[0]}, {x[1]}")
    y = torch.min(x[0], x[1])
    return y

# def f_min_domain(x):
#     return x[0].min(x[1])

def f_min_domain(Theta):
    def res(x):
        y = x[0].min(x[1])
        # if x[1].right.data.item() > x[1].left.data.item():
        #     print(f"DEBUG: f_max_domain, {torch.autograd.grad(y.left, Theta, retain_graph=True, allow_unused=True)}")
        return y
    return res


# def f6(x):
#     return x[0].sub(var(0.1).mul(x[0].sub(var(60))))
# def f6_domain(x):
#     return x[0].sub_l((x[0].sub_l(var(60))).mul(var(0.1)))
def f6(x):
    y = x[0].sub(var(0.1).mul(x[0].sub(x[1])))
    # print(f"DEBUG,  f6, {x[0].data.item()}, {y.data.item()}")
    return y
def f6_domain(x):
    return x[0].sub_l((x[0].sub_l(x[1])).mul(var(0.1)))

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
# def f12(x):
#     return x[0].sub(var(0.1).mul(x[0].sub(var(60)))).add(var(5.0))
# def f12_domain(x):
#     return x[0].sub_l((x[0].sub_l(var(60.0))).mul(var(0.1))).add(var(5.0))

def f12(x):
    y = x[0].sub(var(0.1).mul(x[0].sub(x[1]))).add(var(5.0))
    # print(f"DEBUG,  f12, {x[0].data.item()}, {y.data.item()}")
    return y
def f12_domain(x):
    return x[0].sub_l((x[0].sub_l(x[1])).mul(var(0.1))).add(var(5.0))

# def f12_theta(theta):
#     def res(x):
#         return x[0].sub(var(0.1).mul(x[0].sub(theta))).add(var(5.0))
#     return res
# def f12_theta_domain(theta):
#     def res(x):
#         return x[0].sub_l((x[0].sub_l(theta)).mul(var(0.1))).add(var(5.0))
#     return res

def f19(x):
    return x[1]
def f19_domain(x):
    return x[1]

def sigmoid_domain(x):
    return (x.mul(var(-1.0)).exp().add(var(1.0))).div(var(1.0))

def linear_domain(x, Theta, n=2):
    # hidden_states: n
    # len_input == 2
    # Theta[:n]: weights of x1,
    # Theta[n:2n]: weights of x2
    # Theta[2n:3n]: bias of hidden state
    # Theta[3n:4n]: weights of output
    # debug_file = open(debug_file_dir, 'a')
    #! for DEBUG
    # Theta[1], Theta[2], Theta[3], Theta[4], Theta[5], Theta[6], Theta[7] = var(0.0), var(0.1), var(0.0), var(0.0), var(0.0), var(1.0), var(1.0)
    # print(f"DEBUG: linear domain- input, {x[0].left.data.item()}, {x[0].right.data.item()}, {x[1].left.data.item()}, {x[1].right.data.item()}")
    # if x[0].right.data.item() > x[0].left.data.item():
    #     print(f"DEBUG: linear domain- input, {x[0].left.data.item()}, {x[0].right.data.item()}, {x[1].left.data.item()}, {x[1].right.data.item()}")
    len_input = len(x)
    hidden_state_list = list()
    output = var(0.0)
    # print(f"DEBUG: len(theta), {len(Theta)}")
    for i in range(n):
        u = x[0].mul(Theta[i + 1]).add(x[1].mul(Theta[i+n + 1])).add(Theta[i+2*n + 1])

        # if x[0].right.data.item() > x[0].left.data.item(): 
        #     print(f"DEBUG:, hidden state **BEFORE** sigmoid {u.left.data.item()}, {u.right.data.item()}")
        # add sigmoid to hidden neuron
        u = sigmoid_domain(u)
        # if x[0].right.data.item() > x[0].left.data.item():
        #     print(f"DEBUG:, hidden state **AFTER** sigmoid {u.left.data.item()}, {u.right.data.item()}")
        hidden_state_list.append(u)
    for i in range(n):
        # print(hidden_state_list[i].left.data.item(), Theta[i+3*n].data.item(), sigmoid_domain(hidden_state_list[i].mul(Theta[i+3*n])).left.data.item())
        # output = (hidden_state_list[i].mul(Theta[i+3*n+ 1])).add(output)
        # output = sigmoid_domain(hidden_state_list[i]).mul(Theta[i+3*n+ 1]).add(output)
        output = hidden_state_list[i].mul(Theta[i+3*n+ 1]).add(output)
    # print(f"DEBUG: Output before adding bias, {output.left.data.item()}, {output.right.data.item()}")

    output = output.add(Theta[4*n+1])
    
    # print(f"DEBUG: theta, {[i.data.item() for i in Theta]}")
    # if x[0].right.data.item() > x[0].left.data.item(): 
    #     print(f"DEBUG: linear_domain, {torch.autograd.grad(output.left, Theta, retain_graph=True, allow_unused=True)[0][0]}")
    # print(f"DEBUG: hidden {u.left.data.item() for u in hidden_state_list}, \n {u.right.data.item() for u in hidden_state_list}")
    # if x[0].right.data.item() > x[0].left.data.item(): 
    #     print(f"DEBUG: Output, {output.left.data.item()}, {output.right.data.item()}")

    # exit(0)
    return output

def f_degrade_nn_domain(Theta):
    def nn(x):
        # curT = domain.Interval(var(60.0), var(60.0))
        output = linear_domain(x, Theta, n=2)
        # output = x[0].mul(Theta[1]).add(x[1].mul(Theta[3]))
        # output = x[0].sub_l((x[0].sub_l(x[1])).mul(var(0.1)))
        return output
    return nn

def sigmoid(x):
    return var(1.0).div((var(1.0).add(torch.exp(var(-1.0).mul(x)))))

def linear(input, Theta, n=2):
    # hidden_states: n
    # len_input == 2
    # Theta[:n]: weights of x1,
    # Theta[n:2n]: weights of x2
    # Theta[2n:3n]: bias of hidden state
    # Theta[3n:4n]: weights of output
    #! for DEBUG
    # Theta[1], Theta[2], Theta[3], Theta[4], Theta[5], Theta[6], Theta[7] = var(0.0), var(0.1), var(0.0), var(0.0), var(0.0), var(1.0), var(1.0)
    len_input = len(input)
    hidden_state_list = list()
    for i in range(n):
        hidden_state_list.append(var(0.0))
    # hidden_state_list = [var(0.0)] * n
    output = var(0.0)
    # print(f"DEBUG: len(theta), {len(Theta)}")
    for i in range(n):
        hidden_state_list[i] = input[0].mul(Theta[i + 1]).add(input[1].mul(Theta[i+n + 1])).add(Theta[i+2*n + 1])
    for i in range(n):
        output = sigmoid(hidden_state_list[i]).mul(Theta[i+3*n+1]).add(output)
        # output = (hidden_state_list[i].mul(Theta[i+3*n + 1])).add(output)
    output = output.add(Theta[4*n + 1])
    
    # print(f"DEBUG: linear, {torch.autograd.grad(output, Theta, retain_graph=True, allow_unused=True)}")
        
    return output

def f_degrade_nn(Theta):
    def nn(x):
        # curT = var(60.0)
        output = linear(x, Theta, n=2)
        # output = x[0].sub(var(0.1).mul(x[0].sub(x[1])))
        # output = x[0].mul(Theta[1]).add(x[1].mul(Theta[3]))
        return output
    return nn

def f_equal(x):
    return x[1]

def f_equal_domain(x):
    return x[1]

def construct_syntax_tree(Theta):
    
    l18_2 = Assign(['x_max', 'x'], f_max_domain(Theta), None)
    l18_1 = Assign(['x_min', 'x'], f_min_domain(Theta), l18_2)
    l18 = Assign('i', f18_domain, l18_1)

    l8 = Assign(['isOn'], f8_domain, None)
    l10 = Assign(['isOn'], f10_domain, None)
    l7 = Ifelse('x', Theta, fself_0, l8, l10, None)

    # l6 = Assign(['x'], f6_domain, l7)
    l6 = Assign(['x', 'lin'], f_degrade_nn_domain(Theta), l7)

    l14 = Assign(['isOn'], f8_domain, None)
    l16 = Assign(['isOn'], f10_domain, None)
    l13 = Ifelse('x', var(80.0), fself, l14, l16, None)

    l12 = Assign(['x', 'lin'], f12_domain, l13)
    l5 = Ifelse('isOn', var(0.5), fself, l6, l12, l18)

    l19 = Assign(['res', 'x'], f19_domain, None)
    l4 = WhileSample('i', var(40.0), l5, l19)
    l0 = Assign(['lin', 'x'], f_equal_domain, l4)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point_generate(Theta):

    l18_2 = AssignPoint(['x_max', 'x'], f_max, None)
    l18_1 = AssignPoint(['x_min', 'x'], f_min, l18_2)
    l18 = AssignPoint('i', f18, l18_1)

    l8 = AssignPoint(['isOn'], f8, None)
    l10 = AssignPoint(['isOn'], f10, None)
    l7 = IfelsePoint('x', Theta, fself_0, l8, l10, None)

    l6 = AssignPoint(['x', 'lin'], f6, l7)

    l14 = AssignPoint(['isOn'], f8, None)
    l16 = AssignPoint(['isOn'], f10, None)
    l13 = IfelsePoint('x', var(80.0), fself, l14, l16, None)

    l12 = AssignPoint(['x', 'lin'], f12, l13)
    l5 = IfelsePoint('isOn', var(0.5), fself, l6, l12, l18)

    l19 = AssignPoint(['res', 'x'], f19, None)
    l4 = WhilePoint('i', var(40), l5, l19)
    l0 = AssignPoint(['lin', 'x'], f_equal, l4)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_point(Theta):

    l18_2 = AssignPoint(['x_max', 'x'], f_max, None)
    l18_1 = AssignPoint(['x_min', 'x'], f_min, l18_2)
    l18 = AssignPoint('i', f18, l18_1)

    l8 = AssignPoint(['isOn'], f8, None)
    l10 = AssignPoint(['isOn'], f10, None)
    l7 = IfelsePoint('x', Theta, fself_0, l8, l10, None)

    l6 = AssignPoint(['x', 'lin'], f_degrade_nn(Theta), l7)

    l14 = AssignPoint(['isOn'], f8, None)
    l16 = AssignPoint(['isOn'], f10, None)
    l13 = IfelsePoint('x', var(80.0), fself, l14, l16, None)

    l12 = AssignPoint(['x', 'lin'], f12, l13)
    l5 = IfelsePoint('isOn', var(0.5), fself, l6, l12, l18)

    l19 = AssignPoint(['res', 'x'], f19, None)
    l4 = WhilePoint('i', var(40), l5, l19)
    l0 = AssignPoint(['lin', 'x'], f_equal, l4)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict


def construct_syntax_tree_smooth_point(Theta):

    l18_2 = AssignPointSmooth(['x_max', 'x'], f_max, None)
    l18_1 = AssignPointSmooth(['x_min', 'x'], f_min, l18_2)
    l18 = AssignPointSmooth('i', f18, l18_1)

    l8 = AssignPointSmooth(['isOn'], f8, None)
    l10 = AssignPointSmooth(['isOn'], f10, None)
    l7 = IfelsePointSmooth('x', Theta, fself_0, l8, l10, None)

    # l6 = AssignPointSmooth(['x', 'lin'], f6, l7)
    l6 = AssignPointSmooth(['x', 'lin'], f_degrade_nn(Theta), l7)

    l14 = AssignPointSmooth(['isOn'], f8, None)
    l16 = AssignPointSmooth(['isOn'], f10, None)
    l13 = IfelsePointSmooth('x', var(80.0), fself, l14, l16, None)

    l12 = AssignPointSmooth(['x', 'lin'], f12, l13)
    l5 = IfelsePointSmooth('isOn', var(0.5), fself, l6, l12, l18)

    l19 = AssignPoint(['res', 'x'], f19, None)
    l4 = WhilePointSmooth('i', var(40.0), l5, l19)
    l0 = AssignPoint(['lin', 'x'], f_equal, l4)

    tree_dict = dict()
    tree_dict['entry'] = l0
    tree_dict['para'] = Theta

    return tree_dict
