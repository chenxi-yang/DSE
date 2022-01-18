'''
For the entire trajectory
total epochs: K, every K/3 model, extract the trajectory
1. extract 100 randomly sampled concrete trajectories
2. extract the trajectories
'''
import time
import constants
import importlib
import torch

from utils import (
    load_model,
    create_abstract_states_from_components,
    extract_abstract_representation,
)

from data_loader import (
    load_data,
)


def store_trajectory(output_states, trajectory_path):
    trajectory_log_file = open(trajectory_path, 'w')
    trajectory_log_file.write(f"###\n")
    # print(f"trajectory starts")
    for trajectory_idx, trajectory_l in enumerate(output_states['trajectories_l']):
        # print(f"{trajectory_l}")
        # continue
        trajectory_r = output_states['trajectories_r'][trajectory_idx]
        trajectory_log_file.write(f"trajectory_idx {trajectory_idx}\n")
        for state_idx, state_l in enumerate(trajectory_l):
            state_r = trajectory_r[state_idx]
            for idx, x_l in enumerate(state_l):
                x_r = state_r[idx]
                trajectory_log_file.write(f"{float(x_l)}, {float(x_r)};")
            trajectory_log_file.write(f"\n")
    trajectory_log_file.close()
    return 


<<<<<<< HEAD
def is_safe_trajectory(trajectory, safe_range_list):
    for state in trajectory:
        for x in state:
            if x >= safe_range_list[0] and x<= safe_range_list[1]:
                pass
            else:
                return False
    return True


def calculate_safety(output_states, safe_range_list):
    total_trajectories = 0
    safe_trajectories = 0
    for trajectory_idx, trajectory_l in enumerate(output_states['trajectories_l']):
        trajectory_r = output_states['trajectories_r'][trajectory_idx]
        safe_flag = 1
        for state_idx, state_l in enumerate(trajectory_l):
            state_r = trajectory_r[state_idx]
            target_l, target_r = state_l[0], state_r[1]
            if target_l < safe_range_list[0] or target_r > safe_range_list[1]:
                safe_flag = 0
                break
        if safe_flag == 1:
            safe_trajectories += 1
        total_trajectories += 1
    print(f"Concrete Safe Trajectory Percentage: {safe_trajectories/total_trajectories}")


=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
def extract_trajectory(
        method_dict,
        ini_states,
        category,
        epoch,
<<<<<<< HEAD
        safe_range_list=None,
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
    ):
    model_path, model_name = method_dict['model_path'], method_dict['model_name']
    m = Program(l=64, nn_mode='complex')

    _, m = load_model(m, model_path, name=model_name, epoch=epoch)
    if m is None:
        print(f"model path: {model_path}/{model_name} no model to extract trajectory")
        return 
        # raise ValueError(f"No model to extract concrete trajectory!")
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False
    
    output_states = m(ini_states)
    store_trajectory(
        output_states, 
        trajectory_path=f"plot/plot_trajectories/{method_dict['model_name']}_{method_dict['method']}_{epoch}_{category}.txt",
    )
<<<<<<< HEAD
    calculate_safety(output_states, safe_range_list)
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f


if __name__ == "__main__":

    assert(constants.plot == True)

    # use the first model
    configs = dict()
    configs['Thermostat'] = dict()
    configs['Thermostat-New'] = dict() # 1
    configs['Racetrack'] = dict()
    configs['Racetrack-Easy-Multi'] = dict()
    configs['Racetrack-Relaxed-Multi'] = dict() # 2
    configs['Racetrack-Moderate'] = dict()
    configs['Racetrack-Moderate2'] = dict()
    configs['Racetrack-Moderate3'] = dict()
    configs['Racetrack-Moderate3-1'] = dict()
    configs['Racetrack-Hard'] = dict()
    configs['AircraftCollision'] = dict() 
    configs['AC-New'] = dict() # 3
    configs['AC-New-1'] = dict()
<<<<<<< HEAD
    configs['cartpole_v3'] = dict()
    configs['cartpole_v2'] = dict()
=======
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f

    # Thermostat
    # configs['Thermostat-New']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_10_83.0_4_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Thermostat-New',
    #     'epoch_list': [1499],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_10_83.0_4_0",
    #     'method': 'DSE',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     'epoch_list': [499, 1499],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
<<<<<<< HEAD
    # configs['Thermostat-New']['DiffAI'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"thermostat_new_complex_64_2_100_10_83.0_1_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     'epoch_list': [1499],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
=======
    configs['Thermostat-New']['DiffAI'] = {
        'model_path': f"gpu_DiffAI/models",
        'model_name': f"thermostat_new_complex_64_2_100_10_83.0_1_0",
        'method': 'DiffAI+',
        'benchmark': 'Thermostat-New',
        # 'epoch_list': [0, 1499],
        'epoch_list': [1499],
        # 'epoch_list': [4999],
        'benchmark_name': 'thermostat_new',
        'dataset_path': "dataset/thermostat_new_83.0.txt",
        'x_l': [60.0],
        'x_r': [64.0],
    }
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f

    # AC-New-1
    # configs['AC-New-1']['Ablation(10)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new-1',
    #     'epoch_list': [999],
    #     'benchmark_name': 'aircraft_collision_new_1',
    #     'dataset_path': "dataset/aircraft_collision_new_1_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New-1']['DiffAI(10)'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"aircraft_collision_new_1_complex_64_2_100_10_100000.0_1_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'AC-new-1',
    #     'epoch_list': [999],
    #     'benchmark_name': 'aircraft_collision_new_1',
    #     'dataset_path': "dataset/aircraft_collision_new_1_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New-1']['DSE(10)'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0",
    #     'method': 'DSE',
    #     'benchmark': 'AC-new-1',
    #     'epoch_list': [499],
    #     'benchmark_name': 'aircraft_collision_new_1',
    #     'dataset_path': "dataset/aircraft_collision_new_1_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New-1']['DSE(10)'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0",
    #     'method': 'DSE',
    #     'benchmark': 'AC-new-1',
    #     'epoch_list': [999],
    #     'benchmark_name': 'aircraft_collision_new_1',
    #     'dataset_path': "dataset/aircraft_collision_new_1_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }

    # Racetrack-Relaxed-Multi
    # configs['Racetrack-Relaxed-Multi']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"racetrack_relaxed_multi_complex_64_2_1_10_0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'epoch_list': [1499],
    #     'benchmark_name': 'racetrack_relaxed_multi',
    #     'dataset_path': "dataset/racetrack_relaxed_multi_0.txt",
    #     'x_l': [5.0],
    #     'x_r': [6.0],
    # }
    # configs['Racetrack-Relaxed-Multi']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"racetrack_relaxed_multi_complex_64_2_2_10_0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Racetrack-Easy-Multi',
    #     'epoch_list': [1999, 5999],
    #     'benchmark_name': 'racetrack_relaxed_multi',
    #     'dataset_path': "dataset/racetrack_relaxed_multi_0.txt",
    #     'x_l': [5.0],
    #     'x_r': [6.0],
    # }
    # configs['Racetrack-Relaxed-Multi']['DiffAI'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"racetrack_relaxed_multi_complex_64_2_10_10_0_0_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'epoch_list': [5999],
    #     'benchmark_name': 'racetrack_relaxed_multi',
    #     'dataset_path': "dataset/racetrack_relaxed_multi_0.txt",
    #     'x_l': [5.0],
    #     'x_r': [6.0],
    # }
<<<<<<< HEAD
    # configs['Racetrack-Relaxed-Multi']['DiffAI'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"racetrack_relaxed_multi_complex_64_2_10_10_0_0_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'epoch_list': [5999],
    #     'benchmark_name': 'racetrack_relaxed_multi',
    #     'dataset_path': "dataset/racetrack_relaxed_multi_0.txt",
    #     'x_l': [5.0],
    #     'x_r': [6.0],
    # }
    configs['cartpole_v3']['DiffAI'] = {
        'model_path': f"gpu_DiffAI/models",
        'model_name': f"cartpole_v3_complex_64_2_1_50_0.21_0_0",
        'method': 'DiffAI+',
        'benchmark': 'cartpole_v3',
        'epoch_list': [999],
        'benchmark_name': 'cartpole_v3',
        'dataset_path': "dataset/cartpole_v3_0.21.txt",
        'x_l': [-0.05, -0.05, -0.05, -0.05],
        'x_r': [0.05, 0.05, 0.05, 0.05],
        'safe_range_list': [-0.210, 0.210],
    }
    configs['cartpole_v3']['DSE'] = {
        'model_path': f"gpu_DSE/models",
        'model_name': f"cartpole_v3_complex_64_2_1_50_0.21_0_0",
        'method': 'DSE',
        'benchmark': 'cartpole_v3',
        'epoch_list': [999],
        'benchmark_name': 'cartpole_v3',
        'dataset_path': "dataset/cartpole_v3_0.21.txt",
        'x_l': [-0.05, -0.05, -0.05, -0.05],
        'x_r': [0.05, 0.05, 0.05, 0.05],
        'safe_range_list': [-0.210, 0.210],
    }
    configs['cartpole_v3']['Ablation'] = {
        'model_path': f"gpu_only_data/models",
        'model_name': f"cartpole_v3_complex_64_2_1_50_0.21_0_0",
        'method': 'Ablation',
        'benchmark': 'cartpole_v3',
        'epoch_list': [999],
        'benchmark_name': 'cartpole_v3',
        'dataset_path': "dataset/cartpole_v3_0.21.txt",
        'x_l': [-0.05, -0.05, -0.05, -0.05],
        'x_r': [0.05, 0.05, 0.05, 0.05],
        'safe_range_list': [-0.210, 0.210],
    }
    configs['cartpole_v2']['DiffAI'] = {
        'model_path': f"gpu_DiffAI/models",
        'model_name': f"cartpole_v2_complex_64_2_1_50_0.1_0_0",
        'method': 'DiffAI+',
        'benchmark': 'cartpole_v2',
        'epoch_list': [999],
        'benchmark_name': 'cartpole_v2',
        'dataset_path': "dataset/cartpole_v2_0.1.txt",
        'x_l': [-0.05, -0.05, -0.05, -0.05],
        'x_r': [0.05, 0.05, 0.05, 0.05],
        'safe_range_list': [-0.1, 0.1],
    }
    configs['cartpole_v2']['DSE'] = {
        'model_path': f"gpu_DSE/models",
        'model_name': f"cartpole_v2_complex_64_2_1_50_0.1_0_0",
        'method': 'DSE',
        'benchmark': 'cartpole_v2',
        'epoch_list': [999],
        'benchmark_name': 'cartpole_v2',
        'dataset_path': "dataset/cartpole_v2_0.1.txt",
        'x_l': [-0.05, -0.05, -0.05, -0.05],
        'x_r': [0.05, 0.05, 0.05, 0.05],
        'safe_range_list': [-0.1, 0.1],
    }
    configs['cartpole_v2']['Ablation'] = {
        'model_path': f"gpu_only_data/models",
        'model_name': f"cartpole_v2_complex_64_2_1_50_0.1_0_0",
        'method': 'Ablation',
        'benchmark': 'cartpole_v2',
        'epoch_list': [999],
        'benchmark_name': 'cartpole_v2',
        'dataset_path': "dataset/cartpole_v2_0.1.txt",
        'x_l': [-0.05, -0.05, -0.05, -0.05],
        'x_r': [0.05, 0.05, 0.05, 0.05],
        'safe_range_list': [-0.1, 0.1],
    }
=======
    

>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f

    torch.autograd.set_detect_anomaly(True)
    print(f"#### Extract Trajectory ####")
    for benchmark, benchmark_dict in configs.items():
        for method, method_dict in benchmark_dict.items():
            for epoch in method_dict['epoch_list']:
                print(f"-- GENERATE TRAJECTORY -- {benchmark} & {method} & {epoch}")
                start_time = time.time()
                constants.status = 'verify_AI'
                constants.benchmark_name = method_dict['benchmark_name']

                Trajectory_train, Trajectory_test = load_data(
                    train_size=100, 
                    test_size=200, # 20000
                    dataset_path=method_dict['dataset_path'],
                )
                components = extract_abstract_representation(
                    Trajectory_test, 
                    x_l=method_dict['x_l'], 
                    x_r=method_dict['x_r'], 
                    num_components=100)

                import import_hub as hub
                importlib.reload(hub)
                from import_hub import *

                ini_states = initialization_components_point(x_l=method_dict['x_l'], x_r=method_dict['x_r'])
                extract_trajectory(
                    method_dict=method_dict,
                    ini_states=ini_states,
                    category='concrete',
                    epoch=epoch,
<<<<<<< HEAD
                    safe_range_list=method_dict['safe_range_list'],
                )
                # no symbolic trajectories
                # exit(0)
                # abstract_states = create_abstract_states_from_components(components)
                # ini_states = initialize_components(abstract_states)
                # extract_trajectory(
                #     method_dict=method_dict,
                #     ini_states=ini_states,
                #     category='symbolic',
                #     epoch=epoch,
                # )
=======
                )
                # exit(0)
                abstract_states = create_abstract_states_from_components(components)
                ini_states = initialize_components(abstract_states)
                extract_trajectory(
                    method_dict=method_dict,
                    ini_states=ini_states,
                    category='symbolic',
                    epoch=epoch,
                )
>>>>>>> 69e3c7c6074948b0d898e3ae03f538ae3313895f
                print(f"-- FINISH TRAJECTORY -- {benchmark} & {method} -- {time.time() - start_time} s.")

    # configs['Thermostat']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"thermostat_refined_complex_64_2_1_5000_83.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Thermostat',
    #     # 'epoch_list': [0, 1499],
    #     'epoch_list': [1499],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'thermostat_refined',
    #     'dataset_path': "dataset/thermostat_refined_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['Ablation(10)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_10_83.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     # 'epoch_list': [199, 989, 999],
    #     # 'epoch_list': [989],
    #     # 'epoch_list': [199, 989, 1499],
    #     'epoch_list': [199, 989, 1499],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['Ablation(50)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_50_83.0_1_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     # 'epoch_list': [199, 989, 999],
    #     # 'epoch_list': [989],
    #     # 'epoch_list': [199, 989, 1499],
    #     'epoch_list': [199, 989, 1499],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['Ablation(100)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_100_83.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     # 'epoch_list': [199, 989, 999],
    #     # 'epoch_list': [989],
    #     # 'epoch_list': [199, 989, 1499],
    #     'epoch_list': [999],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['Ablation(1000)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_1000_83.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     # 'epoch_list': [199, 989, 999],
    #     # 'epoch_list': [989],
    #     # 'epoch_list': [199, 989, 1499],
    #     'epoch_list': [999],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['Ablation(2500)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_2500_83.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     # 'epoch_list': [199, 989, 999],
    #     # 'epoch_list': [989],
    #     # 'epoch_list': [199, 989, 1499],
    #     'epoch_list': [324],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"thermostat_new_complex_64_2_1_100_83.0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     'epoch_list': [999, 4999],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat-New']['DiffAI'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"thermostat_new_complex_64_2_100_100_83.0_0_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'Thermostat-New',
    #     # 'epoch_list': [0, 1499],
    #     'epoch_list': [4999],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'thermostat_new',
    #     'dataset_path': "dataset/thermostat_new_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat']['DiffAI'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"thermostat_refined_complex_64_2_100_100_83.0_0_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'Thermostat',
    #     # 'epoch_list': [0, 1499],
    #     'epoch_list': [749],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'thermostat_refined',
    #     'dataset_path': "dataset/thermostat_refined_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }
    # configs['Thermostat']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"thermostat_refined_complex_64_2_1_100_83.0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Thermostat',
    #     # 'epoch_list': [0, 1499],
    #     'epoch_list': [749],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'thermostat_refined',
    #     'dataset_path': "dataset/thermostat_refined_83.0.txt",
    #     'x_l': [60.0],
    #     'x_r': [64.0],
    # }

    # AC-new
    # configs['AC-New']['Ablation(10)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_10_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [1499],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New']['Ablation(50)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_50_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [999, 1499],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New']['Ablation(100)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_100_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [999],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New']['Ablation(500)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_500_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [999],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New']['Ablation(2500)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [9, 99, 199, 237],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New']['Ablation(2500)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [999],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New']['Ablation(5000)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_5000_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [3555],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AC-New']['Ablation(10000)'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC-new',
    #     'epoch_list': [1999, 2499, 2725],
    #     'benchmark_name': 'aircraft_collision_new',
    #     'dataset_path': "dataset/aircraft_collision_new_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }

    # configs['Racetrack']['DiffAI'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'Racetrack',
    #     # 'epoch_list': [0, 2999],
    #     'epoch_list': [749],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'racetrack_easy_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_easy_classifier_ITE_0.txt",
    #     'x_l': [4.0],
    #     'x_r': [6.0],
    # }
    # configs['Racetrack']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Racetrack',
    #     # 'epoch_list': [0, 2999],
    #     'epoch_list': [749],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'racetrack_easy_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_easy_classifier_ITE_0.txt",
    #     'x_l': [4.0],
    #     'x_r': [6.0],
    # }
    # configs['Racetrack-Moderate']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Racetrack-Moderate',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     'epoch_list': [4799],
    #     'benchmark_name': 'racetrack_moderate_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_moderate_classifier_ITE_0.txt",
    #     'x_l': [7.0],
    #     'x_r': [10.0],
    # }
    # configs['Racetrack-Moderate2']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Racetrack-Moderate2',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     'epoch_list': [0, 1999, 3999, 5999, 7999],
    #     'benchmark_name': 'racetrack_moderate_2_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_moderate_2_classifier_ITE_0.txt",
    #     'x_l': [7.0],
    #     'x_r': [10.0],
    # }
    # configs['Racetrack-Moderate2']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"racetrack_moderate_2_classifier_ITE_complex_64_2_1_5000_0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Racetrack-Moderate2',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     # 'epoch_list': [0, 1999, 3999, 5999, 7999],
    #     'epoch_list': [3999],
    #     'benchmark_name': 'racetrack_moderate_2_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_moderate_2_classifier_ITE_0.txt",
    #     'x_l': [7.0],
    #     'x_r': [10.0],
    # }
    # configs['Racetrack-Moderate3']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"racetrack_moderate_3_classifier_ITE_complex_64_2_1_5000_0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Racetrack-Moderate3',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     # 'epoch_list': [0, 1999, 3999, 5999, 7999],
    #     # 'epoch_list': [499, 999],
    #     'epoch_list': [999],
    #     'benchmark_name': 'racetrack_moderate_3_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_moderate_3_classifier_ITE_0.txt",
    #     'x_l': [7.0],
    #     'x_r': [9.0],
    # }
    # configs['Racetrack-Moderate3-1']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"racetrack_moderate_3_classifier_ITE_complex_64_2_2_5000_0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Racetrack-Moderate3',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     # 'epoch_list': [0, 1999, 3999, 5999, 7999],
    #     # 'epoch_list': [499, 999],
    #     'epoch_list': [1999],
    #     'benchmark_name': 'racetrack_moderate_3_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_moderate_3_classifier_ITE_0.txt",
    #     'x_l': [7.0],
    #     'x_r': [9.0],
    # }
    # configs['Racetrack-Easy-Multi']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"racetrack_easy_multi_complex_64_2_1_1000_0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Racetrack-Easy-Multi',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     # 'epoch_list': [0, 1999, 3999, 5999, 7999],
    #     # 'epoch_list': [499, 999],
    #     'epoch_list': [1499],
    #     'benchmark_name': 'racetrack_easy_multi',
    #     'dataset_path': "dataset/racetrack_easy_multi_0.txt",
    #     'x_l': [4.0],
    #     'x_r': [6.0],
    # }

    # configs['Racetrack-Moderate3']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Racetrack-Moderate3',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     # 'epoch_list': [0, 1999, 3999, 5999, 7999],
    #     'epoch_list': [7999], # 9999
    #     'benchmark_name': 'racetrack_moderate_3_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_moderate_3_classifier_ITE_0.txt",
    #     'x_l': [7.0],
    #     'x_r': [9.0],
    # }
    # configs['Racetrack-Hard']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"racetrack_hard_classifier_ITE_complex_64_2_1_5000_0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'Racetrack-Hard',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     'epoch_list': [1499, 2499],
    #     'benchmark_name': 'racetrack_hard_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_hard_classifier_ITE_0.txt",
    #     'x_l': [4.0],
    #     'x_r': [6.0],
    # }
    # configs['Racetrack-Hard']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"racetrack_hard_classifier_ITE_complex_64_2_1_100_0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'Racetrack-Hard',
    #     # 'epoch_list': [0, 2999],
    #     # 'epoch_list': [0, 1499, 3499],
    #     # 'epoch_list': [0, 1999, 3999, 5999, 7999],
    #     'epoch_list': [1499, 2999],
    #     'benchmark_name': 'racetrack_hard_classifier_ITE',
    #     'dataset_path': "dataset/racetrack_hard_classifier_ITE_0.txt",
    #     'x_l': [4.0],
    #     'x_r': [6.0],
    # }
    # configs['AircraftCollision']['Ablation'] = {
    #     'model_path': f"gpu_only_data/models",
    #     'model_name': f"aircraft_collision_refined_classifier_ITE_complex_64_2_1_5000_100000.0_0_0",
    #     'method': 'Ablation',
    #     'benchmark': 'AC',
    #     # 'epoch_list': [0, 1999],
    #     'epoch_list': [1499],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'aircraft_collision_refined_classifier_ITE',
    #     'dataset_path': "dataset/aircraft_collision_refined_classifier_ITE_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AircraftCollision']['DiffAI'] = {
    #     'model_path': f"gpu_DiffAI/models",
    #     'model_name': f"aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0",
    #     'method': 'DiffAI+',
    #     'benchmark': 'AC',
    #     # 'epoch_list': [0, 1999],
    #     'epoch_list': [4999],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'aircraft_collision_refined_classifier_ITE',
    #     'dataset_path': "dataset/aircraft_collision_refined_classifier_ITE_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }
    # configs['AircraftCollision']['DSE'] = {
    #     'model_path': f"gpu_DSE/models",
    #     'model_name': f"aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0",
    #     'method': 'DSE',
    #     'benchmark': 'AC',
    #     # 'epoch_list': [0, 1999],
    #     'epoch_list': [999, 4999],
    #     # 'epoch_list': [4999],
    #     'benchmark_name': 'aircraft_collision_refined_classifier_ITE',
    #     'dataset_path': "dataset/aircraft_collision_refined_classifier_ITE_100000.0.txt",
    #     'x_l': [12.0],
    #     'x_r': [16.0],
    # }

