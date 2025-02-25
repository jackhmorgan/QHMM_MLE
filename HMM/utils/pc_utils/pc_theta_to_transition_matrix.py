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
from ..calculate_volatility_bins import calculate_volatility_bins
from scipy.stats import ncx2, gamma

def pc_theta_to_transition_matrix(theta : tuple | list | np.ndarray, 
                                  k : int, 
                                  volatilities : list | np.ndarray):
    """
    The function `pc_theta_to_transition_matrix` calculates transition matrix of the tightly
    parameterized case based on given parameters and volatility bins.
    
    :param theta: The `theta` parameter is a tuple, list, or numpy array containing three values: alpha,
    beta, and sigma. These values are used in the calculation of transition probabilities in the
    function `pc_theta_to_transition_matrix`
    :type theta: tuple | list | np.ndarray
    :param k: The parameter `k` in the function `pc_theta_to_transition_matrix` represents the number of spot 
    volatility time steps per observation time step. It is used in the formula to calculate transition probabilities
    based on the given theta values and volatilities.
    :type k: int
    :param volatilities: The `volatilities` parameter represents a list or array of volatility values 
    associated with each latent state.
    :type volatilities: list | np.ndarray
    :return: The function `pc_theta_to_transition_matrix` returns a transition matrix for the tightly parameterized 
    model given the input parameters `theta` and the model hyperparameters `k` and `volatilities.
    """
    alpha, beta, sigma = theta
    volatility_bins = calculate_volatility_bins(volatilities)
    ncl = len(volatilities)
    time_unit = 1

    def calculate_transition_probabilities(V0):
        c = 2*alpha
        c /= sigma**2 * (1-np.exp(-alpha/(time_unit*k)))

        # u = c*V0*np.exp(-self.alpha)
        # v = c*V1
        # q = 2*self.alpha*self.beta / (self.sigma**2)
        # q -= 1
        # bv = 2*np.sqrt(u*v)
        # iq = bessel(q, bv)
        # transition_probability = c * np.exp(-(u+v)) #/self.sigma**2
        # transition_probability *= (v/u)**(q/2)
        # transition_probability *= iq

        df = 4*alpha*beta / (sigma**2)
        nc = 2*c*V0*np.exp(-alpha/(time_unit*k))
        dist = ncx2(df, nc)

        transition_probabilities = []        
        prev_cdf = 0

        for bin in volatility_bins:
            cdf = dist.cdf(2*c*bin)
            transition_probabilities.append(cdf-prev_cdf)
            prev_cdf = cdf
        transition_probabilities.append(1-cdf)
        return transition_probabilities
    

    transition_matrix = np.zeros((ncl,ncl))
    
    for i, V0 in enumerate(volatilities):
        probabilities = calculate_transition_probabilities(V0)  
        transition_matrix[i] = probabilities

    return transition_matrix
    