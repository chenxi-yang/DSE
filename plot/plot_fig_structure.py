import pylab
import numpy as np
from scipy.stats import norm

def plot_dist():
    x = np.linspace(-10,10,1000)
    y = norm.pdf(x, loc=2.5, scale=1.5)    # for example
    pylab.plot(x,y)
    pylab.show()

if __name__ == "__main__":
    plot_dist()


