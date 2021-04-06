import matplotlib.pyplot as plt
import re
import os

from args import *
import numpy as np

def read_loss(loss_path):
    q_list = list()
    c_list = list()
    with open(loss_path, 'r') as loss_f:
        loss_f.readline()
        i = 0
        for line in loss_f:
            if i % 3 == 1:
                content = line.split(",")
                q = float(content[1].split(":")[1])
                c = float(content[2].split(":")[1])
                q_list.append(q)
                c_list.append(c)
            i += 1
    return q_list, c_list


def read_loss_2(loss_path):
    q_list = list()
    c_list = list()
    with open(loss_path, 'r') as loss_f:
        # loss_f.readline()
        i = 0
        for line in loss_f:
            # print(i, line)
            if i % 2 == 0:
                content = line.split(":")
                q = float(content[1])
                q_list.append(q)
            if i % 2 == 1:
                content = line.split(",")
                c = float(content[0].split(":")[1])
                c_list.append(c)
            i += 1
    return q_list, c_list


def read_sample(sample_file):
    sample_size_list = list()
    time_list = list()
    with open(sample_file, 'r') as sample_f:
        for line in sample_f:
            content = line.split(',')
            time = float(content[0].split(':')[1])
            length = int(content[1].split(':')[1])
            sample_size_list.append(length)
            time_list.append(time)
    return sample_size_list, time_list


def read_train_log(log_file):
    q_list = list()
    c_list = list()
    with open(log_file, 'r') as log_f:
        log_f.readline()
        for line in log_f:
            # print(line)
            if 'epoch' in line and 'loss' not in line:
                content = line.split(",")
                q = float(content[1].split(":")[1])/5
                c = float(content[2].split(":")[1])/5
                # print(q, c)
                q_list.append(q)
                c_list.append(c)
                if len(q_list) >= 10:
                    break
    return q_list, c_list


def read_vary_constraint(file):
    safe_l_list, safe_r_list, p1_list, p2_list = list(), list(), list(), list()
    f = open(file, 'r')
    f.readline()
    name_line = f.readline()
    name_list = name_line[:-1].split(', ')
    name1 = name_list[-2]
    name2 = name_list[-1]
    print(name1, name2)
    for line in f:
        var = line[:-1].split(', ')
        safe_l_list.append(float(var[0]))
        safe_r_list.append(float(var[1]))
        p1_list.append(float(var[2]))
        p2_list.append(float(var[3]))
    return safe_l_list[:5], safe_r_list[:5], p1_list[:5], p2_list[:5], name1, name2


def plot_line(x_list, y_list, title, x_label, y_label, label, fig_title, c='b', log=False):
    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    plt.plot(x_list, y_list, label=label, c=c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if log is True:
        plt.yscale("log")

    plt.title(title)
    plt.legend()
    plt.savefig(fig_title)
    plt.close()


def plot_dot(x_list, y_list, title, x_label, y_label, label, fig_title, c='b', log=False):
    ax = plt.subplot(111)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    plt.plot(x_list, y_list, 'o', label=label, c=c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if log is True:
        plt.yscale("log")

    plt.title(title)
    plt.legend()
    plt.savefig(fig_title)
    plt.close()


def plot_constraint(x_list, safe_l_list, safe_r_list, p1_list, p2_list, title,  x_label, y_label, label1, label2, fig_title):
    ax = plt.subplot(111)
    # ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    
    ax2 = ax.twinx()

    lns2 = ax.plot(x_list, p1_list, marker='o', label=label1)
    lns3 = ax.plot(x_list, p2_list, marker='o', label=label2)
    ax.set_ylabel("Percentage of Safe Programs")
    ax.set_xlabel("Case Index")
    ax.grid()
    
    # ax.fill_between(x_list, safe_l_list, safe_r_list, color='C3', alpha=0.5)
    range_list = [r - safe_l_list[idx] for idx, r in enumerate(safe_r_list)]
    lns1 = ax2.plot(x_list, range_list, color='C3', label='Constraint Range')
    ax2.set_ylim([32.4,33.3])
    ax2.set_ylabel("Constraint Range")
    # ax2.legend(None)
    # ax2.grid(None)

    lns = lns2+lns3+lns1
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.xticks(range(0,5))
    
    plt.title(title)
    # plt.legend()
    # plt.grid()
    plt.savefig(fig_title)
    plt.close()


def plot_loss(loss_dir):
    for loss_file in os.listdir(loss_dir):
        loss_name = os.path.splitext(loss_file)[0]
        q_list, c_list = read_loss(loss_dir + loss_name + '.txt')
        x_list = list(range(len(q_list)))

        plot_line(x_list, q_list, title=loss_name + ' data loss', x_label='epoch', y_label='loss', label='data loss', fig_title=f"figures/loss/{loss_name}_data_loss.png", c='C0')
        plot_line(x_list, c_list, title=loss_name + ' safe loss', x_label='epoch', y_label='log(loss)', label='safe loss', fig_title=f"figures/loss/{loss_name}_safe_loss.png", c='C1')


def plot_loss_2(loss_dir):
    for loss_file in os.listdir(loss_dir):
        loss_name = os.path.splitext(loss_file)[0]
        q_list, c_list = read_loss_2(loss_dir + loss_name + '.txt')
        x_list = list(range(len(q_list)))

        plot_line(x_list, q_list, title=loss_name + ' data loss', x_label='batch', y_label='log(loss)', label='data loss', fig_title=f"figures/loss/{loss_name}_data_loss.png", c='C0', log=True)
        plot_line(x_list, c_list, title=loss_name + ' safe loss', x_label='batch', y_label='log(loss)', label='safe loss', fig_title=f"figures/loss/{loss_name}_safe_loss.png", c='C1', log=True)
    

def plot_sample(sample_file):
    sample_size_list, time_list = read_sample(sample_file)
    plot_dot(sample_size_list, time_list, title='sample time', x_label='sample size', y_label='time', label='time', fig_title=f"figures/sample/sample_time.png", c='C0')


def plot_training_loss(log_file, benchmark, log=False):
    if log:
        flag = 'log-'
    else:
        flag = ''
    q_list, c_list = read_train_log(log_file)
    x_list = list(range(len(q_list)))

    plot_line(x_list, q_list, title='training data loss', x_label='epoch', y_label=flag + 'loss', label='data loss', fig_title=f"figures/loss/{benchmark}_data_loss.png", c='C0', log=log)
    plot_line(x_list, c_list, title='training safe loss', x_label='epoch', y_label=flag + 'loss', label='safe loss', fig_title=f"figures/loss/{benchmark}_safe_loss.png", c='C1', log=log)


def plot_vary_constraint(file):
    safe_l_list, safe_r_list, p1, p2, name1, name2 = read_vary_constraint(file)
    x_list = list(range(len(safe_l_list)))

    plot_constraint(x_list, safe_l_list, safe_r_list, p1, p2, title='Percentage of Safe Programs with Variable Constraints ',  x_label='constraint', y_label='safe percentage', label1=name1, label2=name2, fig_title=f"figures/vary_constraint_{name1}_{name2}.png")


def vary_safe_bound():
    args = get_args()
    lr = args.lr
    stop_val = args.stop_val
    t_epoch = args.t_epoch
    optimizer_name = args.optimizer
    w = args.w
    benchmark_name = args.benchmark_name
    train_size = args.train_size
    test_size = args.test_size
    num_epoch = args.num_epoch
    width = args.width
    bs = args.bs
    n = args.n
    l = args.l
    nn_mode = args.nn_mode
    b = args.b
    module = args.module
    num_components = args.num_components
    use_smooth_kernel = args.use_smooth_kernel
    save = args.save
    test_mode = args.test_mode
    # safe_start_idx = args.safe_start_idx
    # safe_end_idx = args.safe_end_idx
    # path_sample_size = args.path_sample_size
    data_attr = args.data_attr
    
    method_list = ['DiffAI', 'DiffAI-Kernel', 'DSE', 'SPS']

    if benchmark_name == "thermostat":
        x_l = [55.0]
        x_r = [62.0]
        # SAFE_RANGE = [55.0, 81.34] # strict
        SAFE_RANGE = [53.0, 82.8]
        # first expr
        # safe_range_upper_bound_list = np.arange(82.0, 83.0, 0.1).tolist()
        # PHI = 0.05 # unsafe probability
        safe_range_upper_bound_list = np.arange(82.5, 83.0, 0.15).tolist()
        PHI = 0.10
        # SAFE_RANGE = [53.0, 82.0]
        # SAFE_RANGE = [52.0, 83.0] # not that strict
        # SAFE_RANGE = [50.0, 85.0] # not that loose

    train_time = 3
    result_dict = dict()
    for method in method_list:
        res_safe_upper_list = list()
        avg_verification_probability_list = list()
        safe_percentage_probability_list = list()
        if method == 'DiffAI':
            use_smooth_kernel = False
            log_file_name = f"thermostat_DiffAI_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_DiffAI/result/{log_file_name}.txt"
        if method == 'DiffAI-Kernel':
            use_smooth_kernel = True
            log_file_name = f"thermostat_DiffAI_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_DiffAI/result/{log_file_name}.txt"
        if method == 'DSE':
            use_smooth_kernel = True
            log_file_name = f"thermostat_DSE_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_DSE/result/{log_file_name}.txt"
        if method == 'SPS':
            use_smooth_kernel = True
            log_file_name = f"thermostat_SPS_{lr}_{bs}_{num_epoch}_{train_size}_{use_smooth_kernel}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{safe_range_upper_bound_list}_{PHI}"
            log_file = f"gpu_SPS/result/{log_file_name}.txt"

        f = open(log_file, 'r')
        for line in f:
            if 'safe_range_upper_bound' in line:
                safe_upper_bound = float(line[:-1].split(': ')[-1])
                res_safe_upper_list.append(safe_upper_bound)
                tmp_avg_list = list()
                tmp_safe_percentage_list = list()
            if 'Details' in line:
                if len(tmp_avg_list) < 3:
                    unsafe_probability = float(line.split(',')[0].split(': ')[1])
                    tmp_avg_list.append(unsafe_probability)
                    if unsafe_probability <= PHI:
                        tmp_safe_percentage_list.append(1.0)
                    else:
                        tmp_safe_percentage_list.append(0.0)
                if len(tmp_avg_list) == 3:
                    # remove the maximum probability
                    avg_verification_probability = (sum(tmp_avg_list) - max(tmp_avg_list))/train_time
                    safe_percentage_probability = sum(tmp_safe_percentage_list) / train_time
                    avg_verification_probability_list.append(avg_verification_probability)
                    safe_percentage_probability_list.append(safe_percentage_probability)
        f.close()
        result_dict[method] = {
            'res_safe_upper_list': res_safe_upper_list,
            'avg_verification_probability_list': avg_verification_probability_list,
            'safe_percentage_probability_list': safe_percentage_probability_list,
        }

    all_result_f = open(f"all_results/thermostat_{lr}_{bs}_{num_epoch}_{train_size}_{num_components}_{l}_{b}_{nn_mode}_{module}_{n}_{save}_{SAFE_RANGE[0]}_{PHI}.txt", 'w')
    for method in result_dict:
        all_result_f.write(f"# {method}\n")
        for key in result_dict[method]:
            all_result_f.write(f"{key}: {result_dict[method][key]}\n")
    all_result_f.close()


if __name__ == "__main__":
    # plot_loss('loss/') # the q and c loss
    # plot_loss_2('loss/')
    # plot_sample('data/sample_time.txt')
    # lr_bs_epoch_samplesize
    # plot_training_loss('result/thermostat_nn_volume_[52.0]_[85.1]_0.001_40_10_1000_5000_all_linearrelu.txt', benchmark='thermostat_nn_volume_[52.0]_[85.1]_0.001_40_10_1000_5000_all_linearrelu', log=False)
    # plot_vary_constraint('result/vary_constraint_volume_vs_point_sampling.txt')
    vary_safe_bound()





