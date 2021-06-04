import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import skewnorm
import seaborn as sns

def plot_dist(num_bar):
    # [-2, 2]: 0.24
    fig, ax = plt.subplots(figsize=(5, 5))

    x_l = np.linspace(-2.5,2.5,1000)
    y_l = norm.pdf(x_l, loc=0.0, scale=0.8)    # for example

    # x_h = [-2, -1, 0, 1, 2]
    if num_bar == 5:
        x_h = [0] * 48 + [-1]*21 + [1]*21 + [2]* 4 + [-2] * 4
    elif num_bar == 1:
        x_h = [0] * 100 + [-1]*100 + [1]*100 + [2]* 100 + [-2] * 100
    g = sns.histplot(ax=ax, x=x_h, stat="probability", discrete=True, fill='black', color='black', alpha=0.5)

    g.set(ylabel=None)
    plt.setp(ax.patches, linewidth=0)
    sns.lineplot(ax=ax, x=x_l, y=y_l, color='black')
    plt.savefig(f"figures/dist_{num_bar}.png")


def plot_dists():
    fig, ax = plt.subplots(figsize=(5, 5))

    x_l = np.linspace(-2.5,2.5,1000)
    y_l_1 = norm.pdf(x_l, loc=0.0, scale=0.8)    # for example
    y_l_2 = skewnorm.pdf(x_l, loc=0.0, scale=0.8, a=0.3)
    y_l_3 = skewnorm.pdf(x_l, loc=0.0, scale=0.8, a=-0.5)

    sns.lineplot(ax=ax, x=x_l, y=y_l_1, color='red')
    sns.lineplot(ax=ax, x=x_l, y=y_l_2, color='g')
    sns.lineplot(ax=ax, x=x_l, y=y_l_3, color='b')

    plt.savefig(f"figures/dists.png")


def plot_dists_bar():
    fig, ax = plt.subplots(figsize=(5, 5))

    x_l = np.linspace(-2.5,2.5,1000)
    y_l_1 = norm.pdf(x_l, loc=0.0, scale=0.8)    # for example
    y_l_2 = skewnorm.pdf(x_l, loc=0.0, scale=0.8, a=0.3)
    y_l_3 = skewnorm.pdf(x_l, loc=0.0, scale=0.8, a=-0.5)

    sns.lineplot(ax=ax, x=x_l, y=y_l_1, color='red')
    sns.lineplot(ax=ax, x=x_l, y=y_l_2, color='g')
    sns.lineplot(ax=ax, x=x_l, y=y_l_3, color='b')

    x_h_1 = [0] * 48 + [-1]*21 + [1]*21 + [2]* 4 + [-2] * 4
    x_h_3 = [0] * 46 + [-1]*28 + [1]*15 + [2]* 1 + [-2] * 8
    x_h_2 = [0] * 47 + [-1]*18 + [1]*25 + [2]* 6 + [-2] * 2
    g1 = sns.histplot(ax=ax, x=x_h_1, stat="probability", discrete=True, fill='red', color='red', alpha=0.3)
    g2 = sns.histplot(ax=ax, x=x_h_2, stat="probability", discrete=True, fill='g', color='g', alpha=0.3)
    g3 = sns.histplot(ax=ax, x=x_h_3, stat="probability", discrete=True, fill='b', color='b', alpha=0.3)
    g1.set(ylabel=None)
    g2.set(ylabel=None)
    g3.set(ylabel=None)
    plt.setp(ax.patches, linewidth=0)

    plt.savefig(f"figures/dists_bar.png")


def plot_dists_bar_only():
    fig, ax = plt.subplots(figsize=(5, 5))

    x_h_1 = [0] * 48 + [-1]*21 + [1]*21 + [2]* 4 + [-2] * 4
    x_h_3 = [0] * 46 + [-1]*28 + [1]*15 + [2]* 1 + [-2] * 8
    x_h_2 = [0] * 47 + [-1]*18 + [1]*25 + [2]* 6 + [-2] * 2
    g1 = sns.histplot(ax=ax, x=x_h_1, stat="probability", discrete=True, fill='red', color='red', alpha=0.3)
    g2 = sns.histplot(ax=ax, x=x_h_2, stat="probability", discrete=True, fill='g', color='g', alpha=0.3)
    g3 = sns.histplot(ax=ax, x=x_h_3, stat="probability", discrete=True, fill='b', color='b', alpha=0.3)
    g1.set(ylabel=None)
    g2.set(ylabel=None)
    g3.set(ylabel=None)
    plt.setp(ax.patches, linewidth=0)

    plt.savefig(f"figures/dists_bar_only.png")


def plot_dists_bar_max():
    fig, ax = plt.subplots(figsize=(5, 5))

    x_h_1 = [0] * 48 + [-1]*21 + [1]*21 + [2]* 4 + [-2] * 4
    x_h_2 = [0] * 46 + [-1]*28 + [1]*15 + [2]* 1 + [-2] * 8
    x_h_3 = [0] * 47 + [-1]*18 + [1]*25 + [2]* 6 + [-2] * 2
    g1 = sns.histplot(ax=ax, x=x_h_1, stat="probability", discrete=True, fill='black', color='black', alpha=1.0)
    g2 = sns.histplot(ax=ax, x=x_h_2, stat="probability", discrete=True, fill='black', color='black', alpha=1.0)
    g3 = sns.histplot(ax=ax, x=x_h_3, stat="probability", discrete=True, fill='black', color='black', alpha=1.0)
    g1.set(ylabel=None)
    g2.set(ylabel=None)
    g3.set(ylabel=None)
    plt.setp(ax.patches, linewidth=0)

    plt.savefig(f"figures/dists_bar_max.png")


if __name__ == "__main__":
    # print('test')
    # plot_dist(num_bar=5)
    # plot_dist(num_bar=1)
    plot_dists()
    plot_dists_bar()
    plot_dists_bar_only()
    plot_dists_bar_max()

    

