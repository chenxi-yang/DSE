import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def read_training_loss(mode, benchmark_name, result_prefix):
    pass


def read_point_trajectory(file_name, name_list=['acceleration', 'position', 'velocity']):
    trajectories_dict = dict()
    for name in name_list:
        trajectories_dict[name] = list()

    trajectory_file = open(file_name, 'r')
    trajectory_file.readline()
    
    ini = True
    for line in trajectory_file:
        # print(line)
        if 'trajectory_idx' in line:
            # if 'trajectory_idx 1' in line:
            #     break
            if not ini:
                break
            else:
                ini = False
            tmp_trajectory_dict = {
                    name: list() for name in name_list
                }
        else:
            content = line[:-2].split(';')
            # print(content)
            for idx, state in enumerate(content):
                tmp_trajectory_dict[name_list[idx]].append(float(state.split(', ')[0]))
    for key in trajectories_dict:
        trajectories_dict[key].append(tmp_trajectory_dict[key])
    # print(trajectories_dict)
    return trajectories_dict


def split_trajectories(trajectories_dict_list, name_list, name_idx):
    trajectories_list = list()
    for trajectories_dict in trajectories_dict_list:
        trajectories_list.append(trajectories_dict[name_list[name_idx]])
    # print(trajectories_list)
    
    return trajectories_list


def plot_long_line(x_list, y_list_list, label_name_list, figure_name, x_label, y_label, constraint=None):
    color_list = ['g', 'b']
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, y_list in enumerate(y_list_list):
        sns.lineplot(ax=ax, x=x_list, y=y_list, label=label_name_list[idx], color=color_list[idx])
    
    if constraint is not None:
        plt.axhline(y=constraint, color='r', linestyle='-')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(figure_name)
    # plt.grid()
    plt.savefig(f"all_figures/trajectories/{figure_name}.png")
    plt.close()


def plot_concrete_trajectories(trajectories_list, name_list, name_idx, mode_list, 
    safe_lower_bound_list,
    safe_upper_bound_list,
    benchmark_name):
    color_list = ['g', 'b', 'orange', 'y']
    fig, ax = plt.subplots(figsize=(10, 5))
    max_num = 1

    name = name_list[name_idx]
    for trajectories_idx, trajectories in enumerate(trajectories_list):
        mode = mode_list[trajectories_idx]
        for trajectory_idx, trajectory in enumerate(trajectories):
            x_list = [i for i in range(len(trajectory))]
            sns.lineplot(ax=ax, x=x_list, y=trajectory, color=color_list[trajectories_idx], label=mode)
            if trajectory_idx > max_num:
                break
    
    sns.lineplot(ax=ax, x=x_list[1:], y=safe_lower_bound_list, color='r', label='safe_bar')
    sns.lineplot(ax=ax, x=x_list[1:], y=safe_upper_bound_list, color='r')
    
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=100))
    plt.xlabel('Step')
    plt.ylabel('Property')
    plt.title(f"Concrete Trajectories of {benchmark_name} {name_list[name_idx]}")
    plt.savefig(f"../all_figures/concrete_trajectories_{benchmark_name}_{name_list[name_idx]}.png")

