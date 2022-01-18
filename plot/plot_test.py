from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

def f1():
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.xlim([0, 15])
    plt.ylim([0, 15])
    n=10
    patches = []
    for i in range(0,n):
        x = random.uniform(1, 10)
        y = random.uniform(1, 10)
        patches.append(
            matplotlib.patches.Rectangle(
                (x, y),
                width=1.0,
                height=1.0,
                fill=False,
                alpha=0.1,
            )
        )
    ax.add_collection(PatchCollection(patches, match_original=True))
    plt.savefig('test.png')


def f2():
    someX, someY = 0.5, 0.5
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.add_patch(matplotlib.patches.Rectangle((someX - .1, someY - .1), 0.2, 0.2, fill=False, alpha=1))
    plt.savefig('test2.png')


def f3():
    N = 21
    x = np.linspace(0, 10, 11)
    y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) +
                            (x - x.mean())**2 / np.sum((x - x.mean())**2))

    fig, ax = plt.subplots()
    ax.plot(x, y_est, '-')
    ax.fill_between(x, y_est - y_err, y_est + y_err, color='blue', alpha=0.2)
    ax.fill_between(x, y_est + y_err, y_est + y_err + y_err, color='blue', alpha=0.2)
    # ax.plot(x, y, 'o', color='tab:brown')
    plt.savefig('test3.png')


# f1()
# f2()
f3()