import numpy as np

def calculate_mse(theta, theta_true):
    mse = 0
    for param, true_param in zip(theta, theta_true):
        mse += (param-true_param)**2
    mse /= len(theta)
    return mse