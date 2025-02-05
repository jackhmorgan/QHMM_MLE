import numpy as np

def calculate_volatility_bins(volatilities : list | np.ndarray):
    """
    The function `calculate_volatility_bins` calculates the bisection of consecutive elements in a list or
    numpy array of volatilities. 
    
    :param volatilities: The `volatilities` parameter is a list or numpy array containing values
    representing the center of the volatility bins.
    :type volatilities: list | np.ndarray
    :return: The function `calculate_volatility_bins` returns a list of values where each value is the
    boarder of two consecutive volatility bins.
    """
    return [(volatilities[i]+volatilities[i+1])/2 for i in range(len(volatilities)-1)]