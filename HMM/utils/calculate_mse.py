import numpy as np

def calculate_mse(theta, theta_true):
    '''The `calculate_mse` function calculates the mean squared error between two parameter
    vectors of the same length
    :param theta: A list or np.ndarray of parameters that are trained.
    :param theta_true: A list or np.ndarray of parameters that were used to generate the
    training data.'''
    mse = 0
    for param, true_param in zip(theta, theta_true):
        mse += (param-true_param)**2
    mse /= len(theta)
    return mse