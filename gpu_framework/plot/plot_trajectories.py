from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib
import matplotlib.pyplot as plt
import math


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
            tmp_state_list = list()
            if configs['benchmark'] == 'Racetrack-Relaxed-Multi':
                properties = line.split(';')[0] 
                content = properties.split(', ')
                l, r = float(content[0]), float(content[1])
                if l < 0 and r <= 0: l, r = abs(r), abs(l)
                elif l < 0 and r > 0: l, r = 0.0,  max(abs(l), abs(r))
                else: pass
                tmp_state_list.append((l, r)) # distance
                properties = line.split(';')[1] 
                content = properties.split(', ')
                l, r = float(content[0]), float(content[1])
                tmp_state_list.append((l, r)) # agent1's x
                properties = line.split(';')[2]
                content = properties.split(', ')
                l, r = float(content[0]), float(content[1])
                tmp_state_list.append((l, r)) # agent2's x
            elif configs['benchmark'] in ['AC', 'AC-New', 'AC-New-1']:
                properties = line.split(';')[0] 
                content = properties.split(', ')
                l, r = float(content[0]), float(content[1])
                if l < 0 and r <= 0: l, r = abs(r), abs(l)
                elif l < 0 and r > 0: l, r = 0.0,  max(abs(l), abs(r))
                else: pass
                l, r = math.sqrt(l), math.sqrt(r)
                tmp_state_list.append((l, r))
            else:
                properties = line.split(';')[0]
                content = properties.split(', ')
                l, r = float(content[0]), float(content[1])
                tmp_state_list.append((l, r)) # distance
            trajectory.append(tmp_state_list)

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
            tmp_state_list = list()
            if configs['benchmark'] == 'Racetrack-Relaxed-Multi':
                properties = line.split(';')[0] 
                content = properties.split(', ')
                value = float(content[0])
                tmp_state_list.append(abs(value)) # distance
                properties = line.split(';')[1] 
                content = properties.split(', ')
                value = float(content[0])
                tmp_state_list.append(value) # agent1's x
                properties = line.split(';')[2]
                content = properties.split(', ')
                value = float(content[0])
                tmp_state_list.append(value) # agent2's x
            elif configs['benchmark'] in ['AC', 'AC-New', 'AC-New-1']:
                properties = line.split(';')[0] 
                content = properties.split(', ')
                value = float(content[0])
                tmp_state_list.append(math.sqrt(abs(value))) # distance
            else:
                properties = line.split(';')[0]
                content = properties.split(', ')
                value = float(content[0])
                tmp_state_list.append(value)
            trajectory.append(tmp_state_list)

    return concrete_trajectory_list


def preprocess_trajectories(trajectory_list, configs):
    # trajectory_list: list of trajectories
    # trajectory: list of (lower bound, upper bound)
    updated_trajectory_list = list()
    for trajectory in trajectory_list:
        tmp_updated_trajectory_list = list()
        for idx, states in enumerate(trajectory):
            states_list = list()
            if configs['benchmark'] not in ['Thermostat', 'AC', 'AC-New', 'AC-New-1']:
                if idx == 0: # not count the initial state
                    continue
            for state in states:
                l, r = state[0], state[1]
                # recangle
                # x, y, width, height = idx, l, 1.0, r - l
                # tmp_updated_trajectory_list.append((x, y, width, height))
                # rhombus
                x1, y1, x2, y2, x3, y3, x4, y4 = idx+0.5, l, idx+1.0, (l+r)/2.0, idx+0.5, r, idx, (l+r)/2.0
                states_list.append((x1, y1, x2, y2, x3, y3, x4, y4))
            tmp_updated_trajectory_list.append(states_list)
        updated_trajectory_list.append(tmp_updated_trajectory_list)
    
    return updated_trajectory_list


def plot_trajectories(concrete_trajectory_list, symbolic_trajectory_list, configs, property_index):
    fig = plt.figure()
    agent_color_list = ['green', 'blue', 'purple']
    if configs['benchmark'] == 'AC':
        ax = fig.add_subplot(111, aspect=0.068)
        plt.xlim([-1, 15])
        plt.ylim([0, 100])
    if configs['benchmark'] == 'AC-New':
        ax = fig.add_subplot(111, aspect=0.068)
        plt.xlim([0, 16])
        plt.ylim([0, 100])
    if configs['benchmark'] == 'AC-New-1':
        ax = fig.add_subplot(111, aspect=0.068)
        plt.xlim([0, 16])
        plt.ylim([0, 100])
    if configs['benchmark'] == 'Racetrack':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Racetrack-Easy-Multi':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Racetrack-Relaxed-Multi':
        ax = fig.add_subplot(111, aspect=1.0)
        if property_index == 0:
            plt.xlim([0, 21])
            plt.ylim([-2, 11])
        if property_index == 1:
            plt.xlim([0, 21])
            plt.ylim([-0.5, 3.5])

    if configs['benchmark'] == 'Thermostat':
        ax = fig.add_subplot(111, aspect=0.1)
        plt.xlim([-1, 11])
        plt.ylim([40, 90])
    if configs['benchmark'] == 'Thermostat-New':
        ax = fig.add_subplot(111, aspect=0.1)
        plt.xlim([0, 21])
        plt.ylim([40, 90])
    if configs['benchmark'] == 'Racetrack-Moderate':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 30])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Racetrack-Moderate2':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Racetrack-Moderate3':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Racetrack-Moderate3-1':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])
    if configs['benchmark'] == 'Racetrack-Hard':
        ax = fig.add_subplot(111, aspect=1.0)
        plt.xlim([0, 20])
        plt.ylim([-5, 15])

    patches = []
    patches_unsafe = []
    patches_starting = []
    for symbolic_trajectory in symbolic_trajectory_list:
        for positions_idx, positions in enumerate(symbolic_trajectory):
            # print(len(positions))
            for position_idx, position in enumerate(positions):
                # print(x, y, width, height)
                # rhombus
                if 'idx_list' in configs:
                    if position_idx in configs['idx_list'][property_index]:
                        pass
                    else:
                        continue
                else:
                    pass  
                x1, y1, x2, y2, x3, y3, x4, y4 = position
                shape = Polygon(
                    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)),
                    fill=False,
                    edgecolor='green',
                    alpha=0.3,
                )
                patches.append(shape)
    for position in configs['unsafe_area'][property_index]:
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
    for position in configs['starting_area'][property_index]:
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
    print(f"len concrete trajectory: {len(concrete_trajectory_list)}")
    for concrete_trajectory in concrete_trajectory_list:
        x = [i + 0.5 for i in range(len(concrete_trajectory))]
        if 'idx_list' in configs:
            for concrete_position_idx in configs['idx_list'][property_index]:
                # print(concrete_position_idx)
                # print(len(concrete_trajectory[0]))
                y = [v[concrete_position_idx] for v in concrete_trajectory]
                plt.plot(x, y, color='red', linewidth=0.25)
        else:
            y = [v[0] for v in concrete_trajectory]
            plt.plot(x, y, color='red', linewidth=0.25)
    
    plt.axis('off')
    plt.savefig(f"figures/trajectories/{configs['benchmark']}_{configs['method']}_{property_index}.png",  bbox_inches='tight', pad_inches = 0)

    return


def visualize_trajectories(configs):
    concrete_trajectory_list = read_concrete_trajectories(configs)
    symbolic_trajectory_list = read_symbolic_trajectories(configs)
    symbolic_trajectory_list = preprocess_trajectories(symbolic_trajectory_list, configs)
    if 'name_list' in configs:
        for property_index in range(len(configs['name_list'])):
            plot_trajectories(concrete_trajectory_list, symbolic_trajectory_list, configs, property_index)
    else:
        plot_trajectories(concrete_trajectory_list, symbolic_trajectory_list, configs, property_index=0)

    return


if __name__ == "__main__":
    unsafe_map_racetrack_relaxed_multi = [
        [
            (0, -10, 0.5, 25), (0.5, 0, 15, 4), (0.5, -10, 20.5, 10), (20.5, 4, 1, 6), 
            (0.5, 7, 4, 1), (0.5, 8, 8, 1), (0.5, 9, 12, 1),
            (0.5, 10, 20.5, 10),
        ],
        [
            (1.0, -10, 21, 10.5),
        ]
       ]
    starting_area_racetrack_relaxed_multi = [[
            (0, 5, 1.0, 1)
        ],
        [
            (0, 0, 1.0, 0.1)
        ]]
    unsafe_map_thermostat_new = [[
        (0, 0, 21, 55.0),
        (0, 83.0, 21, 10),
    ]]
    starting_area_thermostat_new = [[
        (0, 60.0, 1, 4.0),
    ]]
    unsafe_map_ac_new = [[
        (1, 0, 21, math.sqrt(40.0)),
    ]]
    starting_area_ac_new = [[
        (0, 0, 1.0, 0.1),
    ]]

    unsafe_map_ac = [[
        (0, 0, 21, math.sqrt(40.0)),
    ]]
    starting_area_ac= [[
        (-1, 0, 1.0, 0.1),
    ]]
    unsafe_map_racetrack = [[
        (0, 0, 14.5, 4), (0, 6, 3.5, 4), (3.5, 7, 4, 3), (7.5, 8, 4, 2), (19.5, 3, 1, 7),
        (0, -10, 20, 10), (0, 10, 20.5, 10),
    ]]
    starting_area_racetrack = [[
        (0, 4, 1, 2),
    ]]
    unsafe_map_racetrack_easy_multi = [[
        (0, 0, 15.5, 4), (0, 6, 4.5, 4), (3.5, 7, 4, 3), (7.5, 8, 4, 2), (19.5, 3, 1, 7),
        (0, -10, 21, 10), (0, 10, 21.5, 10),
    ]]
    starting_area_racetrack_easy_multi = [[
        (0, 4, 1, 2),
    ]]
    unsafe_map_thermostat = [[
        (-1, 0, 12, 53.0),
        (-1, 83.0, 12, 10),
    ]]
    starting_area_thermostat = [[
        (-1, 60.0, 1, 4.0),
    ]]
    unsafe_map_racetrack_moderate = [[
        (0, 0, 6, 7), (6, 0, 2, 3), (8, 0, 1, 2), (8, 9, 1, 1), (9, 7, 1, 3),
        (9, 0, 1, 1), (10, 0, 2, 1), (10, 5, 2, 5), (12, 7, 1, 3), (13, 9, 1, 1),
        (12, 0, 1, 2), (13, 0, 2, 3), (15, 0, 1, 5), (16, 0, 1, 7), (17, 0, 1, 8),
        (18, 0, 1, 7), (18, 9, 2, 1), (19, 0, 1, 5), (20, 0, 1, 3), (20, 8, 1, 2),
        (21, 0, 6, 2), (21, 7, 1, 3), (22, 6, 2, 4), (24, 7, 3, 3), (27, 9, 3, 1),
        (27, 0, 1, 4), (28, 0, 1, 5), (29, 0, 1, 6),
    ]]
    starting_area_racetrack_moderate = [[
        (0, 7, 1, 3),
    ]]
    unsafe_map_racetrack_moderate_2 = [[
        (0, 0, 5, 7), (5, 0, 1, 6), (6, 0, 2, 3), (8, 0, 1, 2), (8, 9, 1, 1), (9, 7, 1, 3),
        (9, 0, 1, 1), (10, 0, 2, 1), (10, 5, 2, 5), (12, 7, 1, 3), (13, 9, 1, 1),
        (12, 0, 1, 2), (13, 0, 2, 3), (15, 0, 1, 5), (16, 0, 1, 7), (17, 0, 1, 7),
        (18, 0, 2, 7), 
    ]]
    starting_area_racetrack_moderate_2 = [[
        (0, 7, 1, 3),
    ]]
    unsafe_map_racetrack_moderate_3 = [[
        (0, 0, 5, 7), (5, 0, 1, 6), (6, 0, 2, 3), (8, 0, 1, 2), (8, 9, 1, 1), (9, 7, 1, 3),
        (9, 0, 1, 1), (10, 0, 2, 1), (10, 5, 2, 5), (12, 7, 1, 3), (13, 9, 1, 1),
        (12, 0, 1, 2), (13, 0, 2, 3), (15, 0, 1, 5), (16, 0, 1, 7), (17, 0, 1, 7),
        (18, 0, 2, 7), 
    ]]
    starting_area_racetrack_moderate_3 = [[
        (0, 7, 1, 2),
    ]]
    unsafe_map_racetrack_moderate_3 = [[
        (0, 0, 5, 7), (5, 0, 1, 6), (6, 0, 2, 3), (8, 0, 1, 2), (8, 9, 1, 1), (9, 7, 1, 3),
        (9, 0, 1, 1), (10, 0, 2, 1), (10, 5, 2, 5), (12, 7, 1, 3), (13, 9, 1, 1),
        (12, 0, 1, 2), (13, 0, 2, 3), (15, 0, 1, 5), (16, 0, 1, 7), (17, 0, 1, 7),
        (18, 0, 2, 7), 
    ]]
    starting_area_racetrack_moderate_3 = [[
        (0, 7, 1, 2),
    ]]
    unsafe_map_racetrack_hard = [[
        (0, 0, 1, 4), (0, 6, 1, 4), (1, 0, 1, 3), (1, 7, 1, 3), (2, 0, 1, 2), (2, 8, 1, 2),
        (3, 0, 1, 1), (3, 9, 1, 1), (5, 3, 11, 1), (2, 4, 15, 1), (3, 5, 13, 1),
        (5, 6, 10, 1), (15, 0, 5, 1), (18, 1, 2, 1), (19, 2, 1, 1), (15, 9, 5, 1),
        (16, 8, 4, 1), (17, 7, 3, 1), (19, 6, 1, 1),
    ]]
    starting_area_racetrack_hard = [[
        (0, 4, 1, 2)
    ]]

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
    configs['AC-New-1'] = dict() # 3
    # configs['Thermostat-New']['Ablation-10(199)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_199_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_199_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-10(989)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_989_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_989_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-10(1499)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-50(199)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_1_0_Ablation_199_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_1_0_Ablation_199_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-50(989)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_1_0_Ablation_989_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_1_0_Ablation_989_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-50(1499)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_1_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_1_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-100(999)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_100_83.0_0_0_Ablation_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_100_83.0_0_0_Ablation_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-1000(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_1000_83.0_0_0_Ablation_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_1000_83.0_0_0_Ablation_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-2500(324)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_2500_83.0_0_0_Ablation_324_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_2500_83.0_0_0_Ablation_324_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-50(989)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_0_0_Ablation_989_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_0_0_Ablation_989_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-50(1499)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_50_83.0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }

    # AC-new
    # configs['AC-New']['Ablation-10(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10_100000.0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10_100000.0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-50(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_50_100000.0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_50_100000.0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-100(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_100_100000.0_0_0_Ablation_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_100_100000.0_0_0_Ablation_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-500(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_500_100000.0_0_0_Ablation_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_500_100000.0_0_0_Ablation_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-5000(3555)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_5000_100000.0_0_0_Ablation_3555_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_5000_100000.0_0_0_Ablation_3555_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-10000(2000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_1999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_1999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-10000(2500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_2499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_2499_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-10000(2725)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_2725_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_2725_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-1000(369)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_1000_100000.0_0_0_Ablation_369_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_1000_100000.0_0_0_Ablation_369_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-1000(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_1000_100000.0_0_0_Ablation_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_1000_100000.0_0_0_Ablation_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-2500(10)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0_Ablation_9_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0_Ablation_9_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-2500(100)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0_Ablation_99_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0_Ablation_99_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-2500(200)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0_Ablation_199_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_2500_100000.0_0_0_Ablation_199_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-7500(5)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_7500_100000.0_0_0_Ablation_5_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_7500_100000.0_0_0_Ablation_5_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['AC-New']['Ablation-10000(2725)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_2725_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_complex_64_2_1_10000_100000.0_0_0_Ablation_2725_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New',
    # }
    # configs['Thermostat']['DiffAI(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_100_100_83.0_0_0_DiffAI+_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_100_100_83.0_0_0_DiffAI+_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat']['DSE(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_100_83.0_0_0_DSE_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_100_83.0_0_0_DSE_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat']['DSE(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_100_83.0_0_0_DSE_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_100_83.0_0_0_DSE_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat']['Ablation'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_5000_83.0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_5000_83.0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
    # configs['Thermostat']['Initial'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_0_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_0_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
    # configs['Thermostat']['DiffAI(750)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_749_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_749_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
    # configs['Thermostat']['DSE(750)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_749_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_749_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
    # configs['Thermostat']['DiffAI(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
    # configs['Thermostat']['DSE(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
    # configs['Thermostat']['DiffAI(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_100_100_83.0_0_0_DiffAI+_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
    # configs['Thermostat']['DSE(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_refined_complex_64_2_1_100_83.0_0_0_DSE_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat, 
    #     'starting_area': starting_area_thermostat,
    #     'benchmark': 'Thermostat',
    # }
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
    # configs['Racetrack-Easy-Multi']['Ablation(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_easy_multi_complex_64_2_1_1000_0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_easy_multi_complex_64_2_1_1000_0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_easy_multi,
    #     'starting_area': starting_area_racetrack_easy_multi,
    #     'benchmark': 'Racetrack-Easy-Multi',
    # }
    # configs['Racetrack-Relaxed-Multi']['Ablation(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_5000_0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_5000_0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DSE(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DSE(13000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_12999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_12999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DSE(15000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_14999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_14999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DSE(2500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_14999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_100_0_0_0_DSE_14999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DiffAI(2500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_100_100_0_0_0_DiffAI+_2499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_100_100_0_0_0_DiffAI+_2499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DiffAI(15000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_20_100_0_0_0_DiffAI+_14999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_20_100_0_0_0_DiffAI+_14999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Moderate']['Initial'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_0_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_0_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate,
    #     'starting_area': starting_area_racetrack_moderate,
    #     'benchmark': 'Racetrack-Moderate',
    # }
    # configs['Racetrack-Moderate']['DSE(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate,
    #     'starting_area': starting_area_racetrack_moderate,
    #     'benchmark': 'Racetrack-Moderate',
    # }
    # configs['Racetrack-Moderate']['DSE(3500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_3499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_3499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate,
    #     'starting_area': starting_area_racetrack_moderate,
    #     'benchmark': 'Racetrack-Moderate',
    # }
    # configs['Racetrack-Moderate']['DSE(4000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_3999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_3999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate,
    #     'starting_area': starting_area_racetrack_moderate,
    #     'benchmark': 'Racetrack-Moderate',
    # }
    # configs['Racetrack-Moderate']['DSE(4800)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_4799_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_4799_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate,
    #     'starting_area': starting_area_racetrack_moderate,
    #     'benchmark': 'Racetrack-Moderate',
    # }
    # configs['Racetrack-Moderate2']['Initial'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_0_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_0_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_2,
    #     'starting_area': starting_area_racetrack_moderate_2,
    #     'benchmark': 'Racetrack-Moderate2',
    # }
    # configs['Racetrack-Moderate2']['DSE(2000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_1999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_1999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_2,
    #     'starting_area': starting_area_racetrack_moderate_2,
    #     'benchmark': 'Racetrack-Moderate2',
    # }
    # configs['Racetrack-Moderate2']['DSE(4000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_3999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_3999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_2,
    #     'starting_area': starting_area_racetrack_moderate_2,
    #     'benchmark': 'Racetrack-Moderate2',
    # }
    # configs['Racetrack-Moderate2']['DSE(6000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_5999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_5999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_2,
    #     'starting_area': starting_area_racetrack_moderate_2,
    #     'benchmark': 'Racetrack-Moderate2',
    # }
    # configs['Racetrack-Moderate2']['DSE(8000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_7999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_7999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_2,
    #     'starting_area': starting_area_racetrack_moderate_2,
    #     'benchmark': 'Racetrack-Moderate2',
    # }
    # configs['Racetrack-Moderate2']['Ablation(2000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_1999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_1999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_2,
    #     'starting_area': starting_area_racetrack_moderate_2,
    #     'benchmark': 'Racetrack-Moderate2',
    # }
    # configs['Racetrack-Moderate2']['Ablation(4000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_3999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_2_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_3999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_2,
    #     'starting_area': starting_area_racetrack_moderate_2,
    #     'benchmark': 'Racetrack-Moderate2',
    # }

    # Racetrack-Moderate3
    # configs['Racetrack-Moderate3']['Ablation(500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3',
    # }
    # configs['Racetrack-Moderate3']['Ablation(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3',
    # }
    # configs['Racetrack-Moderate3-1']['Ablation(2000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_2_5000_0_0_0_Ablation_1999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_2_5000_0_0_0_Ablation_1999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3-1',
    # }
    # configs['Racetrack-Moderate3']['DSE(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3',
    # }
    # configs['Racetrack-Moderate3']['DSE(2300)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_2299_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_2299_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3',
    # }
    # configs['Racetrack-Moderate3']['DSE(8000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_7999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_7999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3',
    # }
    # configs['Racetrack-Moderate3']['DSE(9000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_8999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_8999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3',
    # }
    # configs['Racetrack-Moderate3']['DSE(10000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_9999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_moderate_3_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_9999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_moderate_3,
    #     'starting_area': starting_area_racetrack_moderate_3,
    #     'benchmark': 'Racetrack-Moderate3',
    # }

    # Racetrack-Hard
    # configs['Racetrack-Hard']['Ablation(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_hard,
    #     'starting_area': starting_area_racetrack_hard,
    #     'benchmark': 'Racetrack-Hard',
    # }
    # configs['Racetrack-Hard']['Ablation(2500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_2499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_5000_0_0_0_Ablation_2499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_hard,
    #     'starting_area': starting_area_racetrack_hard,
    #     'benchmark': 'Racetrack-Hard',
    # }
    # configs['Racetrack-Hard']['DSE(1500)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_hard,
    #     'starting_area': starting_area_racetrack_hard,
    #     'benchmark': 'Racetrack-Hard',
    # }
    # configs['Racetrack-Hard']['DSE(3000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_2999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_hard_classifier_ITE_complex_64_2_1_100_0_0_0_DSE_2999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_hard,
    #     'starting_area': starting_area_racetrack_hard,
    #     'benchmark': 'Racetrack-Hard',
    # }
    
    
    # configs['AircraftCollision']['Ablation'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_5000_100000.0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_5000_100000.0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac,
    #     'starting_area': starting_area_ac,
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DSE(1000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac,
    #     'starting_area': starting_area_ac,
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DiffAI(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_100_100_100000.0_0_0_DiffAI+_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac,
    #     'starting_area': starting_area_ac,
    #     'benchmark': 'AC',
    # }
    # configs['AircraftCollision']['DSE(5000)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_4999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_refined_classifier_ITE_complex_64_2_1_100_100000.0_0_0_DSE_4999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac,
    #     'starting_area': starting_area_ac,
    #     'benchmark': 'AC',
    # }


    # Thermostat
    configs['Thermostat-New']['DiffAI-10(Final)'] = {
        'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_100_10_83.0_0_0_DiffAI+_1499_concrete.txt",
        'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_100_10_83.0_0_0_DiffAI+_1499_symbolic.txt",
        'unsafe_area': unsafe_map_thermostat_new, 
        'starting_area': starting_area_thermostat_new,
        'benchmark': 'Thermostat-New',
    }
    # configs['Thermostat-New']['DSE-10(Middle)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_DSE_499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_DSE_499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['DSE-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_DSE_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_DSE_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }
    # configs['Thermostat-New']['Ablation-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/thermostat_new_complex_64_2_1_10_83.0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_thermostat_new, 
    #     'starting_area': starting_area_thermostat_new,
    #     'benchmark': 'Thermostat-New',
    # }

    # # Racetrack
    # configs['Racetrack-Relaxed-Multi']['Ablation-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_10_0_0_0_Ablation_1499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_1_10_0_0_0_Ablation_1499_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DSE-10(Middle)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_2_10_0_0_0_DSE_1999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_2_10_0_0_0_DSE_1999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DSE-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_2_10_0_0_0_DSE_5999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_2_10_0_0_0_DSE_5999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }
    # configs['Racetrack-Relaxed-Multi']['DiffAI-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_10_10_0_0_0_DiffAI+_5999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/racetrack_relaxed_multi_complex_64_2_10_10_0_0_0_DiffAI+_5999_symbolic.txt",
    #     'unsafe_area': unsafe_map_racetrack_relaxed_multi,
    #     'starting_area': starting_area_racetrack_relaxed_multi,
    #     'benchmark': 'Racetrack-Relaxed-Multi',
    #     'name_list': ['position', 'distance'],
    #     'idx_list': [[1, 2], [0]],
    # }

    # # AC-New-1
    # configs['AC-New-1']['Ablation-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0_Ablation_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0_Ablation_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New-1',
    # }
    # configs['AC-New-1']['DiffAI-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_100_10_100000.0_1_0_DiffAI+_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_100_10_100000.0_1_0_DiffAI+_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New-1',
    # }
    # configs['AC-New-1']['DSE-10(Middle)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0_DSE_499_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0_DSE_499_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New-1',
    # }
    # configs['AC-New-1']['DSE-10(Final)'] = {
    #     'concrete_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0_DSE_999_concrete.txt",
    #     'symbolic_trajectory_path': f"plot_trajectories/aircraft_collision_new_1_complex_64_2_1_10_100000.0_1_0_DSE_999_symbolic.txt",
    #     'unsafe_area': unsafe_map_ac_new, 
    #     'starting_area': starting_area_ac_new,
    #     'benchmark': 'AC-New-1',
    # }

    for benchmark, benchmark_dict in configs.items():
        for method, method_dict in benchmark_dict.items():
            method_dict['method'] = method
            visualize_trajectories(method_dict)



    

    