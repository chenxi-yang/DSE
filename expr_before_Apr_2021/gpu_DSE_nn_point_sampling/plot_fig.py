import matplotlib.pyplot as plt
import re
import os

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
            if 'epoch' in line and 'loss' not in line:
                content = line.split(",")
                q = float(content[1].split(":")[1])/5
                c = float(content[2].split(":")[1])/5
                q_list.append(q)
                c_list.append(c)
                if len(q_list) >= 20:
                    break
    return q_list, c_list


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


def plot_training_loss(log_file, benchmark):
    q_list, c_list = read_train_log(log_file)
    x_list = list(range(len(q_list)))

    plot_line(x_list, q_list, title='training data loss', x_label='epoch', y_label='loss', label='data loss', fig_title=f"figures/loss/{benchmark}_data_loss.png", c='C0')
    plot_line(x_list, c_list, title='training safe loss', x_label='epoch', y_label='loss', label='safe loss', fig_title=f"figures/loss/{benchmark}_safe_loss.png", c='C1')


if __name__ == "__main__":
    # plot_loss('loss/') # the q and c loss
    # plot_loss_2('loss/')
    # plot_sample('data/sample_time.txt')
    # lr_bs_epoch_samplesize
    plot_training_loss('loss/thermostat_nn_53.00_83.00_0.001_40_20.txt', benchmark='thermostat_nn_53.00_83.00_0.001_40_20')
   





