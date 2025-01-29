from .HMM import HMM

import numpy as np
from math import comb

from .utils.npc_utils.calculate_npc_volatilities import calculate_npc_volatilities
from .utils import calculate_emission_matrix, calculate_steady_state
from hmmlearn.hmm import CategoricalHMM

class NPC_HMM(HMM):
    def __init__(self,
                 theta=None,
                 k=1,
                 ncl=None,
                 observations=None,
                 volatilities = None,
                 alpha = None,
                 delta = None,
                 mean = 0
                 ):
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

    def calculate_transition_matrix(self, theta):
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