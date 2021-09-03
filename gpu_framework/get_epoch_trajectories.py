'''
For the entire trajectory
total epochs: K, every K/3 model, extract the trajectory
1. extract 100 randomly sampled concrete trajectories
2. extract the trajectories
'''

import constants
import importlib

from utils import (
    load_model,
    create_abstract_states_from_components,
)

from data_loader import (
    load_data,
)


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


def extract_trajectory(
        model_path,
        model_name,
        ini_states,
        trajectory_path,
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
    store_trajectory(output_states, trajectory_path, category="single")


if __name__ == "__main__":

    assert(constants.plot == True)

    # use the first model
    configs = dict()
    configs['Thermostat'] = dict()
    configs['Racetrack'] = dict()
    configs['AircraftCollision'] = dict()
    configs['Thermostat']['DiffAI'] = {
        'model_path': f"../gpu_DiffAI/models",
        'model_name': f"model_thermostat_refined_complex_64_2_100_100_83.0_0_0",
        'method': 'DiffAI+',
        'benchmark': 'Thermostat',
        'epoch_list': [0, 1500],
        'benchmark_name': 'thermostat_refined',
        'dataset_path': "dataset/thermostat_refined_83.0.txt",
        'x_l': [60.0],
        'x_r': [64.0],
    }
    configs['Thermostat']['DSE'] = {
        'model_path': f"../gpu_DSE/models",
        'model_name': f"model_thermostat_refined_complex_64_2_1_100_83.0_0_0",
        'method': 'DSE',
        'benchmark': 'Thermostat',
        'epoch_list': [0, 1500],
        'benchmark_name': 'thermostat_refined',
        'dataset_path': "dataset/thermostat_refined_83.0.txt",
        'x_l': [60.0],
        'x_r': [64.0],
    }
    configs['Racetrack']['DiffAI'] = {
        'model_path': f"../gpu_DiffAI/models",
        'model_name': f"model_racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0",
        'method': 'DiffAI+',
        'benchmark': 'Racetrack',
        'epoch_list': [0, 3000],
        'benchmark_name': 'racetrack_easy_classifier_ITE',
        'dataset_path': "dataset/racetrack_easy_classifier_ITE_0.txt",
        'x_l': [4.0],
        'x_r': [6.0],
    }
    configs['Racetrack']['DSE'] = {
        'model_path': f"../gpu_DSE/models",
        'model_name': f"model_racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0",
        'method': 'DSE',
        'benchmark': 'Racetrack',
        'epoch_list': [0, 3000],
        'benchmark_name': 'racetrack_easy_classifier_ITE',
        'dataset_path': "dataset/racetrack_easy_classifier_ITE_0.txt",
        'x_l': [4.0],
        'x_r': [6.0],
    }
    configs['AircraftCollision']['DiffAI'] = {
        'model_path': f"../gpu_DiffAI/models",
        'model_name': f" model_aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0",
        'method': 'DiffAI+',
        'benchmark': 'AC',
        'epoch_list': [0, 2000],
        'benchmark_name': 'aircraft_collision_refined_classifier_ITE',
        'dataset_path': "dataset/aircraft_collision_refined_classifier_ITE_100000.0.txt",
        'x_l': [12.0],
        'x_r': [16.0],
    }
    configs['AircraftCollision']['DSE'] = {
        'model_path': f"../gpu_DSE/models",
        'model_name': f"model_aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0",
        'method': 'DSE',
        'benchmark': 'AC',
        'epoch_list': [0, 2000],
        'benchmark_name': 'aircraft_collision_refined_classifier_ITE',
        'dataset_path': "dataset/aircraft_collision_refined_classifier_ITE_100000.0.txt",
        'x_l': [12.0],
        'x_r': [16.0],
    }


    torch.autograd.set_detect_anomaly(True)
    print(f"#### Extract Trajectory ####")
    for benchmark, benchmark_dict in configs.items():
        for method, method_dict in benchmark_dict:
            for epoch in method_dict['epoch_list']:
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

                ini_states = initialization_components_point()
                extract_trajectory(
                    method_dict=method_dict,
                    ini_states=ini_states,
                    categorty='concrete',
                )

                abstract_states = create_abstract_states_from_components(components)
                ini_states = initialize_components(abstract_states)
                extract_trajectory(
                    method_dict=method_dict,
                    ini_states=ini_states,
                    categorty='symbolic',
                )



