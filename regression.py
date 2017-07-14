# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
np.random.seed(seed=1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections
from NN import *

def f_function(nb_of_samples):
    f = 1
    tt = 0

    # target
    x = np.arange(0, 2*math.pi, 0.001)
    t = np.sin(x)

    # noise
    # Create the targets t with some gaussian noise
    x1 = np.random.uniform(0,1 / f, nb_of_samples)
    noise_variance = 0.2  # Variance of the gaussian noise

    # Gaussian noise error for each sample in x
    noise = np.random.randn(x1.shape[0]) * noise_variance
    y = np.sin(2 * np.pi * f * x1 + tt) + noise

    plt.plot(x, t, 'r', label = 'target function')
    # plt.plot(x1.T, y.T, 'b', label='data')
    # plt.plot(2 * np.pi * f * x1, y, 'bo')
    plt.axis([0, 7, -2, 2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    f_function(1000)