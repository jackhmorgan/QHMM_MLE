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

from scipy.stats import ncx2, gamma
import numpy as np

def calculate_pc_volatilities(theta : np.ndarray | list, ncl : int):
    """
    The function calculates the hidden state volatilities as percentiles of a gamma distribution 
    based on given parameters.
    
    :param theta: The `theta` parameter is a numpy array or list containing three values - `alpha`,
    `beta`, and `sigma`. These values are used in the calculation of `a` and `b` which are then used to
    create a gamma distribution with parameters `a` and `scale=1/b
    :type theta: np.ndarray | list
    :param ncl: The `ncl` parameter represents the number of latent states, and thus the number of percentiles 
    percentiles to compute based on the given parameters `theta`.
    :type ncl: int
    :return: The function `calculate_pc_volatilities` returns a list of quantiles calculated using the
    parameters `theta` and `ncl`. The quantiles are calculated based on the inverse of the cumulative
    distribution function of a gamma distribution with parameters `a` and `scale`.
    """
    alpha, beta, sigma = theta
    a = 2*alpha*beta / (sigma**2)
    b = 2 *alpha / (sigma**2)
    dist = gamma(a=a, scale=1/b)
    return [dist.ppf(i/(ncl+1)) for i in range(1,ncl+1)]