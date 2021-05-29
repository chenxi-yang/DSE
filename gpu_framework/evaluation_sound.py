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
if benchmark_name == "unsound_1":
    from unsound_1_sound import (
        Unsound_1,
        load_model,
        save_model,
        initialization_abstract_state,
    )
if benchmark_name == "sampling_1":
    from sampling_1_sound import (
        Sampling_1,
        load_model,
        save_model,
        initialization_abstract_state,
    )

if benchmark_name == "sampling_2":
    from sampling_2_sound import (
        Sampling_2,
        load_model,
        save_model,
        initialization_abstract_state,
    )

if benchmark_name == "path_explosion":
    from path_explosion_sound import (
        PathExplosion,
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


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)
    return res_interval


def get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx):
    trajectory_loss = var_list([0.0])
    # print(f"trajectory len: {len(symbol_table['trajectory'])}")
    tmp_symbol_table_tra_loss = list()
    safe_interval = target_component["condition"]
    unsafe_probability_condition = target_component["phi"]
    method = target_component["method"]
    if method == "last":
        trajectory = [symbol_table['trajectory'][-1]]
    elif method == "all":
        trajectory = symbol_table['trajectory'][:]

    for state in trajectory:
        X = state[target_idx]
        if debug:
            print(f"X:{X.left.data.item(), X.right.data.item()}")
        intersection_interval = get_intersection(X, safe_interval)
        if intersection_interval.isEmpty():
            unsafe_value = var_list([1.0])
        else:
            safe_probability = (intersection_interval.getLength() + eps).div(X.getLength() + eps)
            # TODO: remove this part
            # if safe_probability.data.item() > 1 - unsafe_probability_condition.data.item():
            #     unsafe_value = var_list([0.0])
            # else:
            #     unsafe_value = 1 - safe_probability
            if real_unsafe_value:
                unsafe_value = 1 - safe_probability
            else:
                if safe_probability.data.item() > 1 - unsafe_probability_condition.data.item():
                    unsafe_value = var_list([0.0])
                else:
                    unsafe_value = 1 - safe_probability
        
        # when verify
        # in verification, for each component, if a part of it is unsafe, it's unsafe
        if unsafe_value > 0.0:
            unsafe_value = 1.0
        else:
            unsafe_value = 0.0

        if verify_outside_trajectory_loss:
            if debug:
                print("loss of one symbol table", unsafe_value, symbol_table["probability"])
            tmp_symbol_table_tra_loss.append(unsafe_value * symbol_table["probability"])
        else:
            trajectory_loss = torch.max(trajectory_loss, unsafe_value)
            # print(f"trajectory_loss: {trajectory_loss.data.item()}")
    if verify_outside_trajectory_loss:
        return tmp_symbol_table_tra_loss
    else:
        return trajectory_loss


def extract_unsafe(abstract_state, target_component, target_idx):
    aggregation_p = var_list([0.0])
    abstract_state_unsafe_value = var_list([0.0])
    if verify_outside_trajectory_loss:
        symbol_table_wise_loss_list = list()
        # print(f"Abstract_state: {len(abstract_state)}")
        for symbol_table in abstract_state:
            # print('Symbol_Table')
            tmp_symbol_table_tra_loss = get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx=target_idx)
            # print(f"tmp_symbol_table_tra_loss: {[i.cpu().detach() for i in tmp_symbol_table_tra_loss]}, p: {symbol_table['probability'].data.item()}")
            symbol_table_wise_loss_list.append(tmp_symbol_table_tra_loss)
            aggregation_p += symbol_table['probability']
        abstract_state_wise_trajectory_loss = zip(*symbol_table_wise_loss_list)
        for l in abstract_state_wise_trajectory_loss:
            # print(l) 
            abstract_state_unsafe_value = torch.max(abstract_state_unsafe_value, torch.sum(torch.stack(l)))
    else:
        for symbol_table in abstract_state:
            trajectory_unsafe_value = get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx=target_idx)
            # print(f"component p: {symbol_table['probability'].data.item()}, trajectory_unsafe_value: {trajectory_unsafe_value}")
            abstract_state_unsafe_value += symbol_table['probability'] * trajectory_unsafe_value
            aggregation_p += symbol_table['probability']
        # print(f"temporary aggragation p: {aggregation_p}")
    return aggregation_p, abstract_state_unsafe_value


def verify(abstract_state_list, target):
    for idx, target_component in enumerate(target):
        target_name = target_component["name"]
        all_unsafe_probability = var_list([0.0])
        # print(f"# of abstract state: {len(abstract_state_list)}")
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
        
        print(f"learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target_component['phi'].data.item()}")
        if not debug:
            log_file_evaluation.write(f"Details#learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target_component['phi'].data.item()}\n")
    if debug:
        exit(0)


def in_interval(x, y):
    # check if x in y
    if x.left >= y.left and x.right <= y.right:
        return True
    else:
        return False


def extract_symbol_table_trajectory_worst_case(symbol_table, target_component, target_idx):
    safe_interval = target_component["condition"]
    method = target_component["method"]
    if method == "last":
        trajectory = [symbol_table['trajectory'][-1]]
    elif method == "all":
        trajectory = symbol_table['trajectory'][:]
    
    symbol_table_worst_case_safe = True
    for state in trajectory:
        X = state[target_idx]
        # print(f"X: {X.left}, {X.right}")
        if not in_interval(X, safe_interval):
            return False

    return symbol_table_worst_case_safe


def extract_unsafe_worst_case(abstract_state, target_component, target_idx):
    abstract_state_worst_case_safe = True
    for symbol_table in abstract_state:
        abstract_state_worst_case_safe = extract_symbol_table_trajectory_worst_case(symbol_table, target_component, target_idx)
        if not abstract_state_worst_case_safe:
            return abstract_state_worst_case_safe
    return abstract_state_worst_case_safe


def verify_worst_case(abstract_state_list, target):
    for idx, target_component in enumerate(target):
        target_name = target_component["name"]
        all_unsafe_probability = var_list([0.0])
        # print(f"# of abstract state: {len(abstract_state_list)}")
        worst_case_safe = True
        for abstract_state in abstract_state_list:
            worst_case_safe = extract_unsafe_worst_case(abstract_state, target_component, target_idx=idx)
            if not worst_case_safe:
                break
        
        if not debug:
            log_file_evaluation = open(file_dir_evaluation, 'a')
        if worst_case_safe:
            print(colored(f"#{target_name}: Verified Safe!", "green"))
            if not debug:
                log_file_evaluation.write(f"Verification of #{target_name}#: Verified Safe!\n")
                log_file_evaluation.flush()
        else:
            print(colored(f"#{target_name}: Not Verified Safe!", "red"))
            if not debug:
                log_file_evaluation.write(f"Verification of #{target_name}#: Not Verified Safe!\n")
                log_file_evaluation.flush()
        
        # print(f"learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target_component['phi'].data.item()}")
        # if not debug:
        #     log_file_evaluation.write(f"Details#learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target_component['phi'].data.item()}\n")
    

def show_component_p(component_list):
    component_p_list = list()
    for component in component_list:
        component_p_list.append(component['p'])
        # print(f"component p: {component['p']}, center: {component['center']}, width: {component['width']}")
    # print(f"component p list: {component_p_list}")
    print(f"sum of component p: {sum(component_p_list)}")
    return 


# def analysis_trajectories(abstract_state_list):
#     return 
def store_trajectory(abstract_state_list, trajectory_path):
    trajectory_log_file = open(trajectory_path, 'w')
    trajectory_log_file.write(f"{analysis_name_list}\n")
    for abstract_state_idx, abstract_state in enumerate(abstract_state_list):
        trajectory_log_file.write(f"abstract_state {abstract_state_idx}\n")
        for symbol_table_idx, symbol_table in enumerate(abstract_state):
            trajectory = symbol_table['trajectory']
            trajectory_log_file.write(f"symbol_table {symbol_table_idx}\n")
            for state in trajectory:
                for x in state:
                    trajectory_log_file.write(f"{float(x.left)}, {float(x.right)};")
                trajectory_log_file.write(f"\n")
    trajectory_log_file.close()
    return 


def verification(model_path, model_name, component_list, target, trajectory_path):
    if benchmark_name == "thermostat":
        m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    if benchmark_name == "mountain_car":
        m = MountainCar(l=l, nn_mode=nn_mode, module=module)
    if benchmark_name == "unsound_1":
        m = Unsound_1(l=l, nn_mode=nn_mode)
    if benchmark_name == "sampling_1":
        m = Sampling_1(l=l, nn_mode=nn_mode)
    if benchmark_name == "sampling_2":
        m = Sampling_2(l=l, nn_mode=nn_mode)
    if benchmark_name == "path_explosion":
        m = PathExplosion(l=l, nn_mode=nn_mode)
    
    _, m = load_model(m, MODEL_PATH, name=model_name)
    if m is None:
        print(f"No model to Verify!!")
        # exit(0)
        return
    # m.cuda()
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False
    # print(m.nn.linear1.weight)
    
    abstract_state_list = initialization_abstract_state(component_list)
    print(f"Ini # of abstract state: {len(abstract_state_list)}")
    show_component_p(component_list)
    # print(abstract_state_list[0][0]["x"].c)
    abstract_state_list = m(abstract_state_list)
    store_trajectory(abstract_state_list, trajectory_path)
    if verify_use_probability:
        verify(abstract_state_list, target)
    else:
        verify_worst_case(abstract_state_list, target)
