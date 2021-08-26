from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import random

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

f1()
f2()