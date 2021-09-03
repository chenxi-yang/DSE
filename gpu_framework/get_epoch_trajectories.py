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
    for trajectory_idx, trajectory_l in enumerate(output_states['trajectories_l']):
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


def extract_trajectory(
        method_dict,
        ini_states,
        category,
        epoch,
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


if __name__ == "__main__":

    assert(constants.plot == True)

    # use the first model
    configs = dict()
    configs['Thermostat'] = dict()
    configs['Racetrack'] = dict()
    configs['AircraftCollision'] = dict()
    configs['Thermostat']['DiffAI'] = {
        'model_path': f"gpu_DiffAI/models",
        'model_name': f"thermostat_refined_complex_64_2_100_100_83.0_0_0",
        'method': 'DiffAI+',
        'benchmark': 'Thermostat',
        # 'epoch_list': [0, 1499],
        'epoch_list': [749],
        # 'epoch_list': [4999],
        'benchmark_name': 'thermostat_refined',
        'dataset_path': "dataset/thermostat_refined_83.0.txt",
        'x_l': [60.0],
        'x_r': [64.0],
    }
    configs['Thermostat']['DSE'] = {
        'model_path': f"gpu_DSE/models",
        'model_name': f"thermostat_refined_complex_64_2_1_100_83.0_0_0",
        'method': 'DSE',
        'benchmark': 'Thermostat',
        # 'epoch_list': [0, 1499],
        'epoch_list': [749],
        # 'epoch_list': [4999],
        'benchmark_name': 'thermostat_refined',
        'dataset_path': "dataset/thermostat_refined_83.0.txt",
        'x_l': [60.0],
        'x_r': [64.0],
    }
    configs['Racetrack']['DiffAI'] = {
        'model_path': f"gpu_DiffAI/models",
        'model_name': f"racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0",
        'method': 'DiffAI+',
        'benchmark': 'Racetrack',
        # 'epoch_list': [0, 2999],
        'epoch_list': [749],
        # 'epoch_list': [4999],
        'benchmark_name': 'racetrack_easy_classifier_ITE',
        'dataset_path': "dataset/racetrack_easy_classifier_ITE_0.txt",
        'x_l': [4.0],
        'x_r': [6.0],
    }
    configs['Racetrack']['DSE'] = {
        'model_path': f"gpu_DSE/models",
        'model_name': f"racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0",
        'method': 'DSE',
        'benchmark': 'Racetrack',
        # 'epoch_list': [0, 2999],
        'epoch_list': [749],
        # 'epoch_list': [4999],
        'benchmark_name': 'racetrack_easy_classifier_ITE',
        'dataset_path': "dataset/racetrack_easy_classifier_ITE_0.txt",
        'x_l': [4.0],
        'x_r': [6.0],
    }
    configs['AircraftCollision']['DiffAI'] = {
        'model_path': f"gpu_DiffAI/models",
        'model_name': f"aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0",
        'method': 'DiffAI+',
        'benchmark': 'AC',
        # 'epoch_list': [0, 1999],
        'epoch_list': [749],
        # 'epoch_list': [4999],
        'benchmark_name': 'aircraft_collision_refined_classifier_ITE',
        'dataset_path': "dataset/aircraft_collision_refined_classifier_ITE_100000.0.txt",
        'x_l': [12.0],
        'x_r': [16.0],
    }
    configs['AircraftCollision']['DSE'] = {
        'model_path': f"gpu_DSE/models",
        'model_name': f"aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0",
        'method': 'DSE',
        'benchmark': 'AC',
        # 'epoch_list': [0, 1999],
        'epoch_list': [749],
        # 'epoch_list': [4999],
        'benchmark_name': 'aircraft_collision_refined_classifier_ITE',
        'dataset_path': "dataset/aircraft_collision_refined_classifier_ITE_100000.0.txt",
        'x_l': [12.0],
        'x_r': [16.0],
    }


    torch.autograd.set_detect_anomaly(True)
    print(f"#### Extract Trajectory ####")
    for benchmark, benchmark_dict in configs.items():
        for method, method_dict in benchmark_dict.items():
            for epoch in method_dict['epoch_list']:
                print(f"-- GENERATE TRAJECTORY -- {benchmark} & {method}")
                start_time = time.time()
                constants.status = 'verify_AI'
                constants.benchmark_name = method_dict['benchmark_name']

                Trajectory_train, Trajectory_test = load_data(
                    train_size=100, 
                    test_size=20000, 
                    dataset_path=method_dict['dataset_path'],
                )
                components = extract_abstract_representation(
                    Trajectory_train, 
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
                )

                abstract_states = create_abstract_states_from_components(components)
                ini_states = initialize_components(abstract_states)
                extract_trajectory(
                    method_dict=method_dict,
                    ini_states=ini_states,
                    category='symbolic',
                    epoch=epoch,
                )
                print(f"-- FINISH TRAJECTORY -- {benchmark} & {method} -- {time.time() - start_time} s.")



