import numpy as np
from termcolor import colored

from constants import *
import constants
import importlib

from utils import (
    load_model,
    create_abstract_states_from_components,
)

import import_hub as hub
importlib.reload(hub)
from import_hub import *


def in_interval(x, y):
    # check if x in y
    if x.left >= y.left and x.right <= y.right:
        return True
    else:
        return False


def trajectory_worst_case(trajectory, target_component, target_idx):
    if target_component["map_mode"] is False:
        safe_interval = target_component["condition"]
    method = target_component['method']
    if method == 'last':
        trajectory = [trajectory[-1]]
    elif method == 'all':
        trajectory = trajectory
    
    trajectory_worst_unsafe = False
    for state_idx, state in enumerate(trajectory):
        X = state[target_idx]
        if target_component["map_mode"] is True:
            safe_interval = target_component["map_condition"][state_idx] # the constraint over the k-th step
        if not in_interval(X, safe_interval):
            return True
    return trajectory_worst_unsafe


def verify_worst_case(output_states, target):
    for idx, target_component in enumerate(target):
        target_name = target_component["name"]

        worst_case_unsafe_num = 0.0
        for trajectory in output_states['trajectories']:
            worst_case_unsafe = trajectory_worst_case(trajectory, target_component, target_idx=idx)
            if worst_case_unsafe:
                worst_case_unsafe_num += 1

        worst_case_unsafe_num = worst_case_unsafe_num * 1.0 / len(output_states['trajectories'])
        print(f"verify AI: #{target_name}, worst case unsafe num: {worst_case_unsafe_num}")
        if not constants.debug:
            log_file_evaluation = open(constants.file_dir_evaluation, 'a')
            log_file_evaluation.write(f"verify AI: #{target_name}, worst case unsafe num: {worst_case_unsafe_num}\n")
            log_file_evaluation.flush()
        

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
def store_trajectory(output_states, trajectory_path, category=None):
    trajectory_path = trajectory_path + f"_AI"
    trajectory_path += ".txt"
    trajectory_log_file = open(trajectory_path, 'w')
    trajectory_log_file.write(f"{constants.name_list}\n")
    for trajectory_idx, trajectory in enumerate(output_states['trajectories']):
        trajectory_log_file.write(f"trajectory_idx {trajectory_idx}\n")
        for state in trajectory:
            for x in state:
                trajectory_log_file.write(f"{float(x.left)}, {float(x.right)};")
            trajectory_log_file.write(f"\n")
    trajectory_log_file.close()
    return 


def verifier_AI(model_path, model_name, components, target, trajectory_path):
    m = Program(l=l, nn_mode=nn_mode)
    
    _, m = load_model(m, MODEL_PATH, name=model_name)
    if m is None:
        print(f"No model to Verify!!")
        return
    # m.cuda()
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False
    
    abstract_states = create_abstract_states_from_components(components)
    ini_states = initialize_components(abstract_states)

    output_states = m(ini_states)
    # TODO: to update the trajectory
    store_trajectory(output_states, trajectory_path, category=None)
    
    verify_worst_case(output_states, target)
