import matplotlib.pyplot as plt
import copy
import random

def read_loss(configs):
    file_name = configs['trajectory_path']
    f = open(file_name, 'r')
    f.readline()
    f.readline()
    f.readline()
    data_loss_list, safety_loss_list = list(), list()
    data_loss, safety_loss = list(), list()
    for line in f:
        if 'One train' in line:
            if len(data_loss) > 0:
                data_loss_list.append(data_loss)
                safety_loss_list.append(safety_loss)
                data_loss, safety_loss = list(), list() 
        else:
            if 'finish' in line:
                content = line[:-1].split(', ')
                q = float(content[1].split(': ')[1])
                c = float(content[-1].split(': ')[1]) # real_c in DSE
                data_loss.append(q)
                safety_loss.append(c)
    
    # print(len(data_loss_list), len(safety_loss_list))
    # print(len(data_loss_list[0]), len(safety_loss_list[0]))
    # # list of list, each list is the loss from one training
    # exit(0)
    return (data_loss_list, safety_loss_list)


def preprocess_loss(loss_dict, configs):
    for category, dict in loss_dict.items():
        for method, l in loss_dict[category].items():
            # list of list of losses
            loss_range_l_list, loss_range_r_list = list(), list()
            for loss_idx, loss_list in enumerate(l):    
                # print(len(loss_list), len(loss_range_l_list), len(loss_range_r_list))
                if configs['benchmark'] == 'Racetrack' and loss_idx == 1:
                    continue
                if len(loss_range_l_list) == 0:
                    loss_range_l_list = copy.deepcopy(loss_list)
                    loss_range_r_list = copy.deepcopy(loss_list)
                else:
                    for idx, loss in enumerate(loss_list):
                        loss_range_l_list[idx] = min(loss_range_l_list[idx], loss)
                        loss_range_r_list[idx] = max(loss_range_r_list[idx], loss)
            # print(f"{category}, {method}: \n{loss_range_l_list}\n{loss_range_r_list}")
            loss_dict[category][method] = (loss_range_l_list, loss_range_r_list)
    return loss_dict


def plot_loss(loss_dict, configs, category):
    benchmark = configs['benchmark']
    fig, ax = plt.subplots()
    handles = list()
    labels = list()
    for method, loss in loss_dict.items():
        l, r = loss[0], loss[1]
        x = [i for i in range(len(l))]
        # if a single line, reduce error a bit
        for idx, loss in enumerate(l):
            if l[idx] == r[idx]:
                l[idx] = max(0, l[idx] - 0.1)
                r[idx] = r[idx] + 0.1
                if method == 'AC': # to highlight the value
                    r[idx] = r[idx] + random.random() * 0.5
        handles.append(plt.fill_between(x, l, r, alpha=0.75))
        labels.append(method)
    
    # if category == 'Safety':
    #     ax.set_yscale('log')
    # ax.set_yscale('log')
    if category == 'Safety':
        if benchmark == 'Thermostat':
            plt.ylim(0,35)
        if benchmark == 'AC':
            plt.ylim(0,15)
        if benchmark == 'Racetrack':
            plt.xlim(0,6001)
            plt.ylim(0,10)

    plt.legend(handles=handles, labels=labels)
    plt.title(f"{configs['benchmark']} {category} Loss")
    plt.savefig(f"figures/loss_trend/{benchmark}_{category}.png")


def f_loss(configs):
    diffai_data, diffai_safety = read_loss(configs['DiffAI'])
    dse_data, dse_safety = read_loss(configs['DSE'])
    loss_dict = {
        'data': {
            'DiffAI': diffai_data,
            'DSE': dse_data,
        },
        'safety': {
            'DiffAI': diffai_safety,
            'DSE': dse_safety,
        }
    }
    loss_dict = preprocess_loss(loss_dict, configs)
    plot_loss(loss_dict['data'], configs, category='Data')
    plot_loss(loss_dict['safety'], configs, category='Safety')
    return


if __name__ == "__main__":
    benchmarks = ['only_data', 'DSE']
    trajectory_size = 10
    configs = dict()
    configs['Thermostat'] = dict()
    configs['Racetrack'] = dict()
    configs['AircraftCollision'] = dict()
    configs['Thermostat']['DiffAI'] = {
        'trajectory_path': f"../gpu_DiffAI/result/thermostat_new_complex_64_2_100_{trajectory_size}_{trajectory_size}_[83.0]_volume_10000.txt",
        'method': 'DiffAI+',
        'benchmark': 'Thermostat',
    }
    configs['Racetrack']['DiffAI'] = {
        'trajectory_path': f"../gpu_DiffAI/result/racetrack_relaxed_multi_complex_64_2_10_{trajectory_size}_{trajectory_size}_[0]_volume_10000.txt",
        'method': 'DiffAI+',
        'benchmark': 'Racetrack',
    }
    configs['AircraftCollision']['DiffAI'] = {
        'trajectory_path': f"../gpu_DiffAI/result/aircraft_collision_new_1_complex_64_2_100_{trajectory_size}_{trajectory_size}_[100000.0]_volume_10000.txt",
        'method': 'DiffAI+',
        'benchmark': 'AC',
    }
    configs['Thermostat']['DSE'] = {
        'trajectory_path': f"../gpu_DSE/result/thermostat_new_complex_64_2_1_{trajectory_size}_{trajectory_size}_[83.0]_volume_10000.txt",
        'method': 'DSE',
        'benchmark': 'Thermostat',
    }
    configs['Racetrack']['DSE'] = {
        'trajectory_path': f"../gpu_DSE/result/racetrack_relaxed_multi_complex_64_2_2_{trajectory_size}_{trajectory_size}_[0]_volume_10000.txt",
        'method': 'DSE',
        'benchmark': 'Racetrack',
    }
    configs['AircraftCollision']['DSE'] = {
        'trajectory_path': f"../gpu_DSE/result/aircraft_collision_new_1_complex_64_2_1_{trajectory_size}_{trajectory_size}_[100000.0]_volume_10000.txt",
        'method': 'DSE',
        'benchmark': 'AC',
    }
    configs['Thermostat']['benchmark'] = 'Thermostat'
    configs['Racetrack']['benchmark'] = 'Racetrack'
    configs['AircraftCollision']['benchmark'] = 'AC'
    
    f_loss(configs['Thermostat'])
    f_loss(configs['Racetrack'])
    f_loss(configs['AircraftCollision'])