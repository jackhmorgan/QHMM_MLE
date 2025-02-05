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

from .HMM import HMM

import numpy as np
from math import comb

from .utils.npc_utils.calculate_npc_volatilities import calculate_npc_volatilities
from .utils import calculate_emission_matrix, calculate_steady_state
from hmmlearn.hmm import CategoricalHMM

# The `NPC_HMM` class is a subclass of `HMM` that implements a Hidden Markov Model with specific
# functionalities for Non-Player Characters in a game environment.
class NPC_HMM(HMM):
    def __init__(self,
                 theta : list | np.ndarray = None,
                 k : int = 1,
                 ncl : int = None,
                 observations : list | np.ndarray = None,
                 volatilities : list | np.ndarray = None,
                 alpha : float = None,
                 delta : float = None,
                 mean : float = 0
                 ):
        """
        This Python function initializes a non-parameterized hidden Markov model object with specified parameters, including theta, k, ncl,
        observations, volatilities, alpha, delta, and mean, with default values and calculations for
        certain parameters if not provided.
        
        :param theta: The `theta` parameter is a list or numpy array representing the model parameters.
        :type theta: list | np.ndarray
        :param k: The parameter `k` is the number of spot volatilities per time step. If not provided 
        explicitly, defaults to 1.
        :type k: int (optional)
        :param ncl: The `ncl` parameter represents the number of classical latent states. If
        the `ncl` parameter is not provided when initializing an instance of the class, it will be
        calculated based on the number of parameters provided in `theta`.
        :type ncl: int
        :param observations: Observations are the data points that occupy the discrete number of observable 
        states.
        :type observations: list | np.ndarray
        :param volatilities: The `volatilities` parameter in the `__init__` method is used to specify a
        list or numpy array of volatilities. If this parameter is not provided, the code calculates the
        volatilities using the `calculate_npc_volatilities` function based on the values of `alpha' and 'delta'.
        :type volatilities: list | np.ndarray
        :param alpha: Alpha is a parameter used in the calculation of the volatility value associated with each
        observation bin.
        :type alpha: float
        :param delta: The `delta` parameter is used in the calculation of the volatility value associated with 
        each observation bin.
        :type delta: float
        :param mean: The `mean` parameter is a float that represents the mean of the normal distributions that
        are used to determine the emission probabilities.
        :type mean: float (optional)
        """
        self.k = k
        self.observations = observations
        self.mean = mean
        if ncl is None:
            ncl = np.ceil(np.sqrt(len(theta)))
        self.ncl = ncl

        if volatilities is None:
            if alpha is None or delta is None:
                alpha = -0.057
                delta = 1.358
            volatilities = calculate_npc_volatilities(ncl,
                                                      alpha,
                                                      delta)            
        self.volatilities = volatilities
         
        if not theta is None:
            self.update_theta(theta=theta)

    def calculate_transition_matrix(self, 
                                    theta : np.ndarray | list[list]):
        """
        The function calculates a transition matrix based on a given theta parameter.
        
        :param theta: Parameters used to construct the loosely parameterized transition matrix.
        :return: The function `calculate_transition_matrix` returns a transition matrix based on the
        input parameters int `theta`. The transition matrix is a square matrix with dimensions `ncl x ncl`,
        where `ncl` is the number of latent states.
        """
        matrix = np.zeros((self.ncl, self.ncl), dtype=np.float64)
        index = 0
        for row in range(self.ncl):
            for column in range(self.ncl-1):
                matrix[row][column] = theta[index]
                index += 1
            matrix[row][-1] = 1-np.sum(matrix[row])
        return matrix

    def update_theta(self,
                     theta):
        """
        The function `update_theta` updates the parameters of a Hidden Markov Model using the given
        theta values.
        
        :param theta: The `theta` parameter in the `update_theta` method represents the transition
        probabilities between different states in a Hidden Markov Model (HMM). It is a list of length
        `ncl*(ncl-1)`, where `ncl` is the number of hidden states in the HMM
        """
        if not len(theta) == (self.ncl*(self.ncl-1)):
            raise ValueError('Invalid theta length and ncl')
        
        transition_matrix = self.calculate_transition_matrix(theta)
        steady_state = calculate_steady_state(transition_matrix)
        emission_matrix = calculate_emission_matrix(transition_matrix, 
                              self.volatilities,
                              self.observations,
                              self.k,
                              mean=self.mean,
                              )
        
        model = CategoricalHMM(n_components=self.ncl,
                               init_params='',
                               params='',
                               )
        model.startprob_ = steady_state
        model.transmat_ = transition_matrix
        model.emissionprob_ = emission_matrix.T

        self._model = model

    def log_likelihood(self, sequence):
        super().log_likelihood(sequence)
        return self._model.score(sequence)
    
    def generate_sequence(self, length):
        super().generate_sequence(length)
        X, _ = self._model.sample(length)
        return X