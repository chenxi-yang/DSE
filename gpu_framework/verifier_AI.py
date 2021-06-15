import numpy as np
from termcolor import colored

from constants import *

import domain


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
def store_trajectory(abstract_state_list, trajectory_path, category=None):
    if category is not None:
        trajectory_path = trajectory_path + f"_{category}"
    trajectory_path += ".txt"
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


def verifier_AI(model_path, model_name, component_list, target, trajectory_path):
    m = Program(l=l, nn_mode=nn_mode)
    
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
    
    if extract_one_trajectory:
        abstract_state_list = initialization_abstract_state_point(component_list)
        category = 'point'
    else:
        abstract_state_list = initialization_abstract_state(component_list)
        category = None
    print(f"Ini # of abstract state: {len(abstract_state_list)}")
    show_component_p(component_list)
    # print(abstract_state_list[0][0]["x"].c)
    abstract_state_list = m(abstract_state_list)
    store_trajectory(abstract_state_list, trajectory_path, category=category)
    
    verify_worst_case(abstract_state_list, target)
