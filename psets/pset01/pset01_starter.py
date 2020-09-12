""" Problem Set 01 starter code

Please make sure your code runs on Python version 3.5.0

Due date: 2016-02-05 13:00
"""

import numpy as np
from scipy import spatial
from scipy.stats import norm

def my_knn(X, y, k=1):
    """ Basic k-nearest neighbor functionality

    k-nearest neighbor regression for a numeric test
    matrix. Prediction are returned for the same data matrix
    used for training. For each row of the input, the k
    closest rows (using the l2 distance) in the training
    set are identified. The mean of the observations y
    is used for the predicted value of a new observation.

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy array; the observed response
      k: integer giving the number of neighbors to include

    Returns:
      a 1d numpy array of predicted responses for each row of the input matrix X
    """
    distmat = spatial.distance.pdist(X)


def my_ksmooth(X, y, sigma=1.0):
    """ Kernel smoothing function

    kernel smoother for a numeric test matrix with a Gaussian
    kernel. Prediction are returned for the same data matrix
    used for training. For each row of the input, a weighted
    average of the input y is used for prediction. The weights
    are given by the density of the normal distribution for
    the distance of a training point to the input.

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy vector; the observed response
      sigma: the standard deviation of the normal density function
        used for the weighting scheme

    Returns:
      a 1d numpy array of predicted responses for each row of the input matrix X
    """
    distmat = spatial.distance.pdist(X)
    value = 1
    norm(scale=sigma).pdf(value) # normal density at 'value'

