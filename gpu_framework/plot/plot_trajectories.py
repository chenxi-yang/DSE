from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib
import matplotlib.pyplot as plt


def read_symbolic_trajectories(configs):
    file_name = configs['symbolic_trajectory_path']
    f = open(file_name, 'r')
    f.readline()
    #  read the first property
    symbolic_trajectory_list = list() # each element is a trajectory of (l, r)
    trajectory = list()
    for line in f:
        if 'trajectory_idx' in line:
            if len(trajectory) > 0:
                symbolic_trajectory_list.append(trajectory)
                trajectory = list()
        else:
            # Racetrack, thermostat can choose the first element
            # AC should choose the coordinate, specifically, x
            properties = line.split(';')[0]
            content = properties.split(', ')
            l, r = float(content[0]), float(content[1])
            trajectory.append((l, r))

    return symbolic_trajectory_list


def read_concrete_trajectories(configs):
    file_name = configs['concrete_trajectory_path']
    f = open(file_name, 'r')
    f.readline()

    concrete_trajectory_list = list() # each element is a trajectory of (l, r)
    trajectory = list()
    for line in f:
        if 'trajectory_idx' in line:
            if len(trajectory) > 0:
                concrete_trajectory_list.append(trajectory)
                trajectory = list()
        else:
            # Racetrack, thermostat can choose the first element
            # AC should choose the coordinate, specifically, x
            properties = line.split(';')[0]
            content = properties.split(', ')
            value = float(content[0])
            trajectory.append(value)

    return concrete_trajectory_list


def preprocess_trajectories(trajectory_list):
    # trajectory_list: list of trajectories
    # trajectory: list of (lower bound, upper bound)
    updated_trajectory_list = list()
    for trajectory in trajectory_list:
        tmp_updated_trajectory_list = list()
        for idx, states in enumerate(trajectory):
            l, r = states[0], states[1]
            # recangle
            # x, y, width, height = idx, l, 1.0, r - l
            # tmp_updated_trajectory_list.append((x, y, width, height))
            # rhombus
            x1, y1, x2, y2, x3, y3, x4, y4 = idx+0.5, l, idx+1.0, (l+r)/2.0, idx+0.5, r, idx, (l+r)/2.0
            tmp_updated_trajectory_list.append((x1, y1, x2, y2, x3, y3, x4, y4))
        updated_trajectory_list.append(tmp_updated_trajectory_list)
    
    return updated_trajectory_list


def plot_trajectories(concrete_trajectory_list, symbolic_trajectory_list, configs):
    fig = plt.figure()
    if configs['benchmark'] == 'AC':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Racetrack':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Thermostat':
        ax = fig.add_subplot(111, aspect=0.1)
        plt.xlim([0, 10])
        plt.ylim([40, 90])

    patches = []
    patches_unsafe = []
    patches_starting = []
    for symbolic_trajectory in symbolic_trajectory_list:
        for position in symbolic_trajectory:
            # print(x, y, width, height)
            # rectangle
            # x, y, width, height = position
            # shape = Rectangle(
            #         (x, y),
            #         width,
            #         height,
            #         fill=False,
            #         alpha=0.1,
            #     )
            # rhombus
            x1, y1, x2, y2, x3, y3, x4, y4 = position
            shape = Polygon(
                ((x1, y1), (x2, y2), (x3, y3), (x4, y4)),
                fill=False,
                edgecolor='green',
                alpha=0.35,
            )
            patches.append(shape)
    for position in configs['unsafe_area']:
        x, y, width, height = position
        shape = Rectangle(
            (x, y),
            width, 
            height,
            fill=True,
            alpha=0.6,
            facecolor='gray',
        )
        patches_unsafe.append(shape)
    for position in configs['starting_area']:
        x, y, width, height = position
        shape = Rectangle(
            (x, y),
            width, 
            height,
            fill=True,
            alpha=1.0,
            facecolor='yellow',
        )
        patches_starting.append(shape)
    
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.add_collection(PatchCollection(patches_unsafe, match_original=True))
    ax.add_collection(PatchCollection(patches_starting, match_original=True))

    # plot concrete trajectories
    for concrete_trajectory in concrete_trajectory_list:
        x = [i + 0.5 for i in range(len(concrete_trajectory))]
        plt.plot(x, concrete_trajectory, color='red', linewidth=0.3)
    
    plt.axis('off')
    plt.savefig(f"figures/trajectories/{configs['benchmark']}_{configs['method']}.png",  bbox_inches='tight', pad_inches = 0)

    return


def visualize_trajectories(configs):
    concrete_trajectory_list = read_concrete_trajectories(configs)
    symbolic_trajectory_list = read_symbolic_trajectories(configs)
    symbolic_trajectory_list = preprocess_trajectories(symbolic_trajectory_list)
    plot_trajectories(concrete_trajectory_list, symbolic_trajectory_list, configs)
    return


if __name__ == "__main__":
    unsafe_map_racetrack = [
        (0, 0, 14.5, 4),
        (0, 6, 3.5, 4),
        (3.5, 7, 4, 3),
        (7.5, 8, 4, 2),
        (19.5, 3, 1, 7),
        (0, -10, 20, 10),
        (0, 10, 20.5, 10),
        ]
    starting_area_racetrack = [
        (0, 4, 1, 2),
    ]
    unsafe_map_thermostat = [
        (0, 0, 10, 53.0),
        (0, 83.0, 10, 10),
    ]
    starting_area_thermostat = [
        (0, 60.0, 1, 4.0),
    ]

    configs = dict()
    configs['Thermostat'] = dict()
    configs['Racetrack'] = dict()
    configs['AircraftCollision'] = dict()
    configs['Thermostat']['Initial'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_0_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_0_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat, 
        'starting_area': starting_area_thermostat,
        'benchmark': 'Thermostat',
    }
    configs['Thermostat']['DiffAI(750)'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_749_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_749_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat, 
        'starting_area': starting_area_thermostat,
        'benchmark': 'Thermostat',
    }
    configs['Thermostat']['DSE(750)'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_749_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_749_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat, 
        'starting_area': starting_area_thermostat,
        'benchmark': 'Thermostat',
    }
    configs['Thermostat']['DiffAI(1500)'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_1499_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_1499_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat, 
        'starting_area': starting_area_thermostat,
        'benchmark': 'Thermostat',
    }
    configs['Thermostat']['DSE(1500)'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_1499_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_1499_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat, 
        'starting_area': starting_area_thermostat,
        'benchmark': 'Thermostat',
    }
    configs['Thermostat']['DiffAI(5000)'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_4999_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_4999_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat, 
        'starting_area': starting_area_thermostat,
        'benchmark': 'Thermostat',
    }
    configs['Thermostat']['DSE(5000)'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_4999_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_4999_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat, 
        'starting_area': starting_area_thermostat,
        'benchmark': 'Thermostat',
    }
    # configs['Racetrack']['Initial'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_0_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_0_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack,
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'Racetrack',
    # }
    # configs['Racetrack']['DiffAI(750)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_749_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_749_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack,
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'Racetrack',
    # }
    # configs['Racetrack']['DSE(750)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_749_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_749_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack,
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'Racetrack',
    # }
    # configs['Racetrack']['DiffAI(3000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_2999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_2999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack,
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'Racetrack',
    # }
    # configs['Racetrack']['DSE(3000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack,
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'Racetrack',
    # }
    # configs['Racetrack']['DiffAI(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_100_100_0_0_0_DiffAI+_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack,
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'Racetrack',
    # }
    # configs['Racetrack']['DSE(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack,
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'Racetrack',
    # }
    # configs['AircraftCollision']['Initial'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_0_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_0_symbolic.txt",
    #     'unsafe_area': [
    #             (0, 0, 15, 4),
    #             (0, 6, 4, 4),
    #             (4, 7, 4, 3),
    #             (8, 8, 4, 2),
    #             (19, 3, 1, 7),
    #             (0, -10, 20, 10),
    #             (0, 10, 20, 10),
    #         ],
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DiffAI(750)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_749_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_749_symbolic.txt",
    #     'unsafe_area': [
    #             (0, 0, 15, 4),
    #             (0, 6, 4, 4),
    #             (4, 7, 4, 3),
    #             (8, 8, 4, 2),
    #             (19, 3, 1, 7),
    #             (0, -10, 20, 10),
    #             (0, 10, 20, 10),
    #         ],
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DSE(750)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_749_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_749_symbolic.txt",
    #     'unsafe_area': [
    #             (0, 0, 15, 4),
    #             (0, 6, 4, 4),
    #             (4, 7, 4, 3),
    #             (8, 8, 4, 2),
    #             (19, 3, 1, 7),
    #             (0, -10, 20, 10),
    #             (0, 10, 20, 10),
    #         ],
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DiffAI(2000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_1999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_1999_symbolic.txt",
    #     'unsafe_area': [
    #             (0, 0, 15, 4),
    #             (0, 6, 4, 4),
    #             (4, 7, 4, 3),
    #             (8, 8, 4, 2),
    #             (19, 3, 1, 7),
    #             (0, -10, 20, 10),
    #             (0, 10, 20, 10),
    #         ],
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DSE(2000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_1999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_1999_symbolic.txt",
    #     'unsafe_area': [
    #             (0, 0, 15, 4),
    #             (0, 6, 4, 4),
    #             (4, 7, 4, 3),
    #             (8, 8, 4, 2),
    #             (19, 3, 1, 7),
    #             (0, -10, 20, 10),
    #             (0, 10, 20, 10),
    #         ],
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DiffAI(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_4999_symbolic.txt",
    #     'unsafe_area': [
    #             (0, 0, 15, 4),
    #             (0, 6, 4, 4),
    #             (4, 7, 4, 3),
    #             (8, 8, 4, 2),
    #             (19, 3, 1, 7),
    #             (0, -10, 20, 10),
    #             (0, 10, 20, 10),
    #         ],
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DSE(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_4999_symbolic.txt",
    #     'unsafe_area': [
    #             (0, 0, 15, 4),
    #             (0, 6, 4, 4),
    #             (4, 7, 4, 3),
    #             (8, 8, 4, 2),
    #             (19, 3, 1, 7),
    #             (0, -10, 20, 10),
    #             (0, 10, 20, 10),
    #         ],
    #     'starting_area': [
    #             (0, 4, 1, 2),
    #         ],
    #     'benchmark': 'AC',
    # }

    for benchmark, benchmark_dict in configs.items():
        for method, method_dict in benchmark_dict.items():
            method_dict['method'] = method
            visualize_trajectories(method_dict)



    

    