import numpy as np
from termcolor import colored

from constants import *
# from optimization import *

if benchmark_name == "thermostat":
    from thermostat_nn_sound import (
        ThermostatNN,
        load_model,
        save_model,
        initialization_abstract_state,
    )
if benchmark_name == "mountain_car":
    from mountain_car_sound import (
        MountainCar,
        load_model,
        save_model,
        initialization_abstract_state,
    )

import domain


def check_safety(res, target):
    intersection_interval = get_intersection(res, target)
    if intersection_interval.isEmpty():
        # print('isempty')
        score = torch.max(target.left.sub(res.left), res.right.sub(target.right)).div(res.getLength())
    else:
        # print('not empty')
        score = var(1.0).sub(intersection_interval.getLength().div(res.getLength()))
    
    return score


def eval(X, Y, m, target, category):
    quan_dist = var(0.0)
    safe_dist = var(0.0)
    # Theta = var_list(theta)

    # root = construct_syntax_tree(Theta)
    # root_point = construct_syntax_tree_point(Theta)
    # root_smooth_point = construct_syntax_tree_smooth_point(Theta)
    y_pred = list()
    y_min = P_INFINITY.data.item()
    y_max = N_INFINITY.data.item()
    safe_min = P_INFINITY.data.item()
    safe_max = N_INFINITY.data.item()

    # quantative distance
    data_loss = var(0.0)
    for idx, x in enumerate(X):
        point, label = x, Y[idx]
        point_data = initialization_point_nn(point)
        y_point_list = m(point_data, 'concrete')
        data_loss += distance_f_point(y_point_list[0]['x'].c[2], var(label))
        # symbol_table_point = root_point['entry'].execute(symbol_table_point)

        y_pred.append(y_point_list[0]['x'].c[2].data.item())

        safe_property_min = y_point_list[0]['safe_range'].left.data.item()
        safe_property_max = y_point_list[0]['safe_range'].right.data.item()
        safe_min = min(safe_property_min, safe_min)
        safe_max = max(safe_property_max, safe_max)

    quan_dist = data_loss.div(len(X))
    # symbol_table_rep = initialization(x_l, x_r)
    # symbol_table_rep = root.execute(symbol_table_rep)
    # print('real y interval', y_min, y_max)
    print('real safe interval', safe_min, safe_max)
    safe_res = domain.Interval(var(safe_min), var(safe_max))
    safe_dist = check_safety(safe_res, target)

    print(category + ':')
    print('Quantative Objective: {0:.5f}, Safe Objective: {1:.5f}'.format(quan_dist.data.item(), safe_dist.data.item()))
    if safe_dist.data.item() > 0.0: # TODO: set to epsilon?
        print(colored('Not Safe!', 'red'))
    else:
        print(colored('Safe!', 'green'))
    
    log_file = open(file_dir, 'a')
    # Quantitative loss & safe
    log_file.write(f"Real Safe Interval: [{safe_min}, {safe_max}]\n")
    log_file.write('Test:' + str(quan_dist.data.item()) + ',' + str(safe_dist.data.item()) + '\n')
    log_file.close()

    return


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)
    return res_interval


def get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx):
    trajectory_loss = var_list([0.0])
    for state in symbol_table['trajectory']:
        X = state[target_idx]
        print(f"X:{X.left.data.item(), X.right.data.item()}")
        safe_interval = target["condition"]
        unsafe_probability_condition = target["phi"]
        intersection_interval = get_intersection(X, safe_interval)
        if intersection_interval.isEmpty():
            unsafe_value = var_list([1.0])
        else:
            safe_probability = intersection_interval.getLength().div(X.getLength())
            # TODO: remove this part
            if safe_probability.data.item() > 1 - unsafe_probability_condition.data.item():
                unsafe_value = var_list([0.0])
            else:
                unsafe_value = 1 - safe_probability
        trajectory_loss = torch.max(trajectory_loss, unsafe_value)
    return trajectory_loss


def extract_unsafe(abstract_state, target_component, target_idx):
    aggregation_p = var_list([0.0])
    abstract_state_unsafe_value = var_list([0.0])
    for symbol_table in abstract_state:
        trajectory_unsafe_value = get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx=target_idx)
        # print(f"component p: {symbol_table['probability'].data.item()}, trajectory_unsafe_value: {trajectory_unsafe_value}")
        abstract_state_unsafe_value += symbol_table['probability'] * trajectory_unsafe_value
        aggregation_p += symbol_table['probability']
        print(f"temporary aggragation p: {aggregation_p}")
    return aggregation_p, abstract_state_unsafe_value


def verify(abstract_state_list, target):
    for idx, target_component in enumerate(target):
        target_name = target_component["name"]
        all_unsafe_probability = var_list([0.0])
        print(f"# of abstract state: {len(abstract_state_list)}")
        for abstract_state in abstract_state_list:
            aggregation_p, unsafe_probability = extract_unsafe(abstract_state, target_component, target_idx=idx)
            print(f"aggregation_p: {aggregation_p.data.item()}, unsafe_probability: {unsafe_probability.data.item()}")
            #! make the aggregation_p make more sense
            aggregation_p = torch.min(var(1.0), aggregation_p)
            all_unsafe_probability += aggregation_p * unsafe_probability
        
        if not debug:
            log_file_evaluation = open(file_dir_evaluation, 'a')
        if all_unsafe_probability.data.item() <= target_component['phi'].data.item():
            print(colored(f"#{target_name}: Verified Safe!", "green"))
            if not debug:
                log_file_evaluation.write(f"Verification of #{target_name}#: Verified Safe!\n")
        else:
            print(colored(f"#{target_name}: Not Verified Safe!", "red"))
            if not debug:
                log_file_evaluation.write(f"Verification of #{target_name}#: Not Verified Safe!\n")
        
        print(f"learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target['phi'].data.item()}")
        if not debug:
            log_file_evaluation.write(f"Details#learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target['phi'].data.item()}\n")

def show_component_p(component_list):
    component_p_list = list()
    for component in component_list:
        component_p_list.append(component['p'])
        # print(f"component p: {component['p']}, center: {component['center']}, width: {component['width']}")
    # print(f"component p list: {component_p_list}")
    print(f"sum of component p: {sum(component_p_list)}")
    return 


def verification(model_path, model_name, component_list, target):
    if benchmark_name == "thermostat":
        m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    if benchmark_name == "mountain_car":
        m = MountainCar(l=l, nn_mode=nn_mode, module=module)
    _, m = load_model(m, MODEL_PATH, name=model_name)
    if m is None:
        print(f"No model to Verify!!")
        exit(0)
    m.cuda()
    m.eval()
    # print(m.nn.linear1.weight)

    abstract_state_list = initialization_abstract_state(component_list)
    print(f"Ini # of abstract state: {len(abstract_state_list)}")
    # show_component_p(component_list)
    abstract_state_list = m(abstract_state_list)
    verify(abstract_state_list, target)
