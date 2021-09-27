import matplotlib.pyplot as plt
import os
import numpy as np

def read_value(config):
    # safety portion, data loss
    safety_portion_dict = dict()
    data_loss_dict = dict()
    for method in ['DSE', 'DiffAI+', 'Ablation']:
        subconfig = config[method]
        safety_portion_list, safety_portion_list_min, safety_portion_list_max = list(), list(), list()
        data_loss_list, data_loss_list_min, data_loss_list_max = list(), list(), list()
        safety_portion_dict[method] = dict()
        data_loss_dict[method] = dict()
        for trajectory_size in subconfig['trajectory_size_list']:
            log_path = f"{subconfig['log_path']}{subconfig['result_prefix']}_{trajectory_size}_{trajectory_size}_{subconfig['result_suffix']}"
            if os.path.isfile(log_path):
                f = open(log_path, 'r')
                sub_safety_portion_list = list()
                sub_data_loss_list = list()
                for line  in f:
                    # if 'Namespace' in line:
                    #     sub_safety_portion_list = list()
                    #     sub_data_loss_list = list()
                    if 'Target' in line or 'path_sample_size' in line:
                        continue
                    if 'verify AI' in line:
                        content = line[:-1].split(': ')
                        safety_portion = 1 - float(content[-1]) # the data is the unsafe portion
                        sub_safety_portion_list.append(safety_portion)
                    if 'test data loss' in line:
                        content = line[:-1].split(': ')
                        data_loss = float(content[-1])
                        sub_data_loss_list.append(data_loss)
                safety_portion_list.append(sum(sub_safety_portion_list)/len(sub_safety_portion_list) if len(sub_safety_portion_list) else -0.0000001)
                safety_portion_list_min.append(min(sub_safety_portion_list))
                safety_portion_list_max.append(max(sub_safety_portion_list))
                data_loss_list.append(sum(sub_data_loss_list)/len(sub_data_loss_list) if len(sub_data_loss_list) else -0.0000001)
                data_loss_list_min.append(min(sub_data_loss_list))
                data_loss_list_max.append(max(sub_data_loss_list))
            else:
                # safety_portion_list.append(-0.0)
                # data_loss_list.append(-0.0)
                raise(NameError(f"No Result!! Log path: {log_path}"))
        safety_portion_dict[method]['avg'] = safety_portion_list
        safety_portion_dict[method]['min'] = safety_portion_list_min
        safety_portion_dict[method]['max'] = safety_portion_list_max
        data_loss_dict[method]['avg'] = data_loss_list
        data_loss_dict[method]['min'] = data_loss_list_min
        data_loss_dict[method]['max'] = data_loss_list_max
    return  safety_portion_dict, data_loss_dict


def plot_data_size(safety_portion_dict, data_loss_dict, config):
    # x = [str(config['benchmark_length'] * k) for k in config['trajectory_size_list']]
    move = {
        'DSE': 0.25, 'DiffAI+': 0, 'Ablation': -0.25,
    }
    color = {
        'DSE': 'tab:orange', 'DiffAI+': 'tab:blue', 'Ablation': 'tab:red',
    }
    # x_labels = [str(k) for k in config['trajectory_size_list']]
    # x = np.arange(len(x_labels))
    # x = config['trajectory_size_list']
    x = [k * config['benchmark_length'] for k in config['trajectory_size_list']]

    fig = plt.figure(figsize=(6, 4.5), constrained_layout=True)
    plt.rcParams['font.size'] = '10'
    handles = list()
    labels = list()
    for method, safety_portion_method_dict in safety_portion_dict.items():
    # plt.plot(x, safety_portion_list)
        y, l, r = safety_portion_method_dict['avg'], \
            safety_portion_method_dict['min'], safety_portion_method_dict['max']
        handles.append(plt.fill_between(x, l, r, alpha=0.1, color=color[method], linewidth=0.0))
        labels.append(method)
        plt.plot(x, y, color=color[method], label=method)

        # bars = plt.bar(x_n+move[method], safety_portion_list, width=0.25, label=method, align='center')
        # # for index, value in enumerate(safety_portion_list):
        # #     print(value, x_n[index]+move[method])
        # #     plt.text(value, x_n[index]+move[method], str(value))
        # for rect in bars:
        #     height = rect.get_height()
        #     plt.text(rect.get_x() + rect.get_width()/2.0, height, str(round(height, 2)), ha='center', va='bottom')
    # ax.xaxis.set_ticks(x_n)
    # ax.set_xticklabels(x_labels)
    # print(len(handles), len(labels))
    # plt.legend(handles=handles, labels=labels)
    plt.xticks(np.arange(0, x[-1]+1, int(x[-1]/5)))
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Provable Safety Portion')
    plt.legend(loc='upper right')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.axhline(y=config['safety_portion_bar'], color='r', linestyle='-')
    plt.savefig(f"figures/data_size/{config['benchmark_name']}_safety_portion.png")
    plt.close()

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(6, 4.5), constrained_layout=True)
    plt.rcParams['font.size'] = '10'
    handles = list()
    labels = list()
    for method, data_loss_method_dict in data_loss_dict.items():
        y, l, r = data_loss_method_dict['avg'], \
            data_loss_method_dict['min'], data_loss_method_dict['max']
        handles.append(plt.fill_between(x, l, r, alpha=0.1, color=color[method], linewidth=0.0))
        labels.append(method)
        plt.plot(x, y, color=color[method], label=method)
        # bars = plt.bar(x_n+move[method], data_loss_list, width=0.25, label=method, align='center')
        # for rect in bars:
        #     height = rect.get_height()
        #     plt.text(rect.get_x() + rect.get_width()/2.0, height, str(round(height, 2)), ha='center', va='bottom')
    # plt.plot(x, data_loss_list)
    # ax.xaxis.set_ticks(x_n)
    # ax.set_xticklabels(x_labels)
    # print(len(handles), len(labels))
    # plt.legend(handles=handles, labels=labels)
    plt.xticks(np.arange(0, x[-1]+1, int(x[-1]/5)))
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Test Data Loss')
    plt.legend(loc='upper right')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.axhline(y=config['data_loss_bar'], color='r', linestyle='-')
    plt.savefig(f"figures/data_size/{config['benchmark_name']}_data_loss.png")
    plt.close()


def visualize_data_size(config):
    safety_portion_dict, data_loss_dict = read_value(config)
    plot_data_size(safety_portion_dict, data_loss_dict, config)


if __name__ == '__main__':
    configs = dict()
    configs['Thermostat'] = {
        'benchmark_name': "Thermostat",
        'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
        'benchmark_length': 20,
    }
    configs['AC'] = {
        'benchmark_name': "AC",
        'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000, 50000],
        'benchmark_length': 15,
    }
    configs['Racetrack'] = {
        'benchmark_name': "Racetrack",
        'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000, 50000],
        'benchmark_length': 20,
    }
    configs['Thermostat']['Ablation'] = {
       'log_path': f"../gpu_only_data/result/",
       'result_prefix': f"thermostat_new_complex_64_2_1",
       'result_suffix': f"[83.0]_volume_10000_evaluation.txt",
       'benchmark_name': "Thermostat",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 20,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['Thermostat']['DSE'] = {
       'log_path': f"../gpu_DSE/result/",
       'result_prefix': f"thermostat_new_complex_64_2_1",
       'result_suffix': f"[83.0]_volume_10000_evaluation.txt",
       'benchmark_name': "Thermostat",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 20,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['Thermostat']['DiffAI+'] = {
       'log_path': f"../gpu_DiffAI/result/",
       'result_prefix': f"thermostat_new_complex_64_2_100",
       'result_suffix': f"[83.0]_volume_10000_evaluation.txt",
       'benchmark_name': "Thermostat",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 20,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['AC']['Ablation'] = {
       'log_path': f"../gpu_only_data/result/",
       'result_prefix': f"aircraft_collision_new_1_complex_64_2_1",
       'result_suffix': f"[100000.0]_volume_10000_evaluation.txt",
       'benchmark_name': "AC",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 15,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['AC']['DSE'] = {
       'log_path': f"../gpu_DSE/result/",
       'result_prefix': f"aircraft_collision_new_1_complex_64_2_1",
       'result_suffix': f"[100000.0]_volume_10000_evaluation.txt",
       'benchmark_name': "AC",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 15,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['AC']['DiffAI+'] = {
       'log_path': f"../gpu_DiffAI/result/",
       'result_prefix': f"aircraft_collision_new_1_complex_64_2_100",
       'result_suffix': f"[100000.0]_volume_10000_evaluation.txt",
       'benchmark_name': "AC",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 15,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['Racetrack']['Ablation'] = {
       'log_path': f"../gpu_only_data/result/",
       'result_prefix': f"racetrack_relaxed_multi_complex_64_2_1",
       'result_suffix': f"[0]_volume_10000_evaluation.txt",
       'benchmark_name': "Racetrack",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 20,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['Racetrack']['DSE'] = {
       'log_path': f"../gpu_DSE/result/",
       'result_prefix': f"racetrack_relaxed_multi_complex_64_2_2",
       'result_suffix': f"[0]_volume_10000_evaluation.txt",
       'benchmark_name': "Racetrack",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 20,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }
    configs['Racetrack']['DiffAI+'] = {
       'log_path': f"../gpu_DiffAI/result/",
       'result_prefix': f"racetrack_relaxed_multi_complex_64_2_10",
       'result_suffix': f"[0]_volume_10000_evaluation.txt",
       'benchmark_name': "Racetrack",
       'trajectory_size_list': [10, 50, 100, 250, 500], # 1000, 2500, 5000],
       'benchmark_length': 20,
    #    'safety_portion_bar': 1 - 0.0001,
    #    'data_loss_bar': 0.2608449965715408,
    }

    for benchmark_name, benchmark_config in configs.items():
        visualize_data_size(benchmark_config)

