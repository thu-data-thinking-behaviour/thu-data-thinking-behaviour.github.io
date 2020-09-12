""" Problem Set 04 starter code

Please make sure your code runs on Python version 3.5.0

Due date: 2016-03-04 13:00
"""

import numpy as np
import scipy.optimize

def my_dual_svm(X, y, C=1):
    """ Support vector machine - Dual problem

    SVM classification for a numeric test matrix. The
    returned result is the vector of coefficients from
    the support vector machine (beta, *not* alpha!).

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy array; the observed response
      C: positive numeric value giving the cost parameter in the SVM

    Returns:
      a 1d numpy array of length p giving the coefficients of beta in
      the SVM model
    """
    return np.zeros((X.shape[1])) # correct dimension


def my_primal_svm(X, y, lam=1, k=5, T=100):
    """ Support vector machine - Dual problem

    SVM classification for a numeric test matrix. The
    returned result is the vector of coefficients from
    the support vector machine (beta, *not* alpha!).

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy array; the observed response
      lam: positive numeric value giving the tuning parameter
        in the (primal, penalized format) of the support vector machine
      k: positive integer giving the number of samples selected in
        each iteration of the algorithm
      T: positive integer giving the total number of iteration to run

    Returns:
      a 1d numpy array of length p giving the coefficients of beta in
      the SVM model
    """
    return np.zeros((X.shape[1])) # correct dimension
