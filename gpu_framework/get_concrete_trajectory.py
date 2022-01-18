import numpy as np
from termcolor import colored

from constants import *
import constants
import importlib

from utils import (
    load_model,
)

import import_hub as hub
importlib.reload(hub)
from import_hub import *


def store_trajectory(output_states, trajectory_path, category=None):
    trajectory_path = trajectory_path + f"_{category}"
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


<<<<<<< HEAD
<<<<<<< HEAD
def is_safe_trajectory(trajectory, safe_range_list):
    for state in trajectory:
        for x in state:
            if x >= safe_range_list[0] and x<= safe_range_list[1]:
                pass
            else:
                return False
    return True


def count_concrete_trajectories_safety(output_states, safe_range_list):
    total_trajectories = 0
    safe_trajectories = 0
    for trajectory_idx, trajectory in enumerate(output_states['trajectories']):
        if is_safe_trajectory(trajectory):
            safe_trajectories += 1
        total_trajectories += 1
    print(f"Concrete Safe Trajectory Percentage: {safe_trajectories/total_trajectories}")


=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
def extract_trajectory(
        model_path,
        model_name,
        ini_states,
        trajectory_path,
<<<<<<< HEAD
<<<<<<< HEAD
        safe_range_list,
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
    ):
    m = Program(l=l, nn_mode=nn_mode)

    _, m = load_model(m, model_path, name=model_name)
    if m is None:
        print(f"model path: {model_path}/{model_name} no model to extract concrete trajectory")
        return 
        # raise ValueError(f"No model to extract concrete trajectory!")
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False
    
    output_states = m(ini_states)
<<<<<<< HEAD
<<<<<<< HEAD
    # store_trajectory(output_states, trajectory_path, category="single")
    count_concrete_trajectories_safety(output_states, safe_range_list)
=======
    store_trajectory(output_states, trajectory_path, category="single")
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
    store_trajectory(output_states, trajectory_path, category="single")
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    print(f"#### Extract Trajectory ####")
    for safe_range_bound in safe_range_bound_list:
        print(f"Safe Range Bound: {safe_range_bound}")

        for i in range(5):
            constants.status = 'verify_AI'
            import import_hub as hub
            importlib.reload(hub)
            from import_hub import *

<<<<<<< HEAD
<<<<<<< HEAD
            ini_states = initialization_components_point(x_l, x_r)
=======
            ini_states = initialization_components_point()
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
            ini_states = initialization_components_point()
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f

            extract_trajectory(
                model_path=MODEL_PATH,
                model_name=f"{model_name_prefix}_{safe_range_bound}_{i}_{0}",
                ini_states=ini_states,
<<<<<<< HEAD
<<<<<<< HEAD
                trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}",
                safe_range_list=safe_range_list[0],
=======
                trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}"
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
=======
                trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}"
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
            )
