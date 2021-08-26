from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib
import matplotlib.pyplot as plt


def read_trajectories(configs):
    file_name = configs['trajectory_path']
    f = open(file_name, 'r')
    f.readline()
    #  read the first property
    trajectory_list = list() # each element is a trajectory of (l, r)
    trajectory = list()
    for line in f:
        if 'trajectory_idx' in line:
            if len(trajectory) > 0:
                trajectory_list.append(trajectory)
                trajectory = list()
        else:
            properties = line.split(';')[0]
            content = properties.split(', ')
            l, r = float(content[0]), float(content[1])
            trajectory.append((l, r))

    return trajectory_list


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


def plot_trajectories(trajectories, configs):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1.0)
    plt.xlim([0, 20])
    plt.ylim([-5, 15])

    patches = []
    patches_unsafe = []
    patches_starting = []
    for trajectory in trajectories:
        for position in trajectory:
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
            alpha=0.5,
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
    plt.savefig(f"figures/trajectories/{configs['method']}_{configs['benchmark']}.png")

    return


def visualize_trajectories(configs):
    trajectory_list = read_trajectories(configs)
    trajectory_list = preprocess_trajectories(trajectory_list)
    plot_trajectories(trajectory_list, configs)
    return


def f_dse(unsafe_map):
    components = 100
    configs = {
        'trajectory_path': f"../gpu_DSE/result_test/trajectory/racetrack_easy_classifier_ITE_complex_64_2_1_200_[0]_volume_{components}_1000__0_0_AI.txt",
        'method': 'DSE',
        'benchmark': 'Racetrack',
        'unsafe_area': unsafe_map, 
        'number_property': 1,
        'starting_area': [(0, 4, 1, 2)],
    }
    visualize_trajectories(configs)


def f_only_data(unsafe_map):
    components = 100
    configs = {
        'trajectory_path': f"../gpu_only_data/result_test/trajectory/racetrack_easy_classifier_ITE_complex_64_2_1_200_[0]_volume_{components}_1000__0_0_AI.txt",
        'method': 'Only_Data',
        'benchmark': 'Racetrack',
        'unsafe_area': unsafe_map, 
        'number_property': 1,
        'starting_area': [(0, 4, 1, 2)],
    }
    visualize_trajectories(configs)


if __name__ == "__main__":
    mode = ['only_data', 'DSE']
    unsafe_map = [
        (0, 0, 15, 4),
        (0, 6, 4, 4),
        (4, 7, 4, 3),
        (8, 8, 4, 2),
        (19, 3, 1, 7),
        (0, -10, 20, 10),
        (0, 10, 20, 10),
        ]
    starting_area = [
        (0, 4, 1, 2),
    ]
    if 'DSE' in mode:
        f_dse(unsafe_map)
    if 'only_data' in mode:
        f_only_data(unsafe_map)

    