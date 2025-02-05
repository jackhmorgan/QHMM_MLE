'''
Copyright 2025 Jack Morgan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

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