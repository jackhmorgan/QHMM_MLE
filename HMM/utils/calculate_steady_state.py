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
from scipy.linalg import null_space

def calculate_steady_state(transition_matrix : np.ndarray | list[list]):
        """
        The function calculates the steady state of the latent state Markov chain based on the transition matrix.
        :return: The `calculate_steady_state` function returns the steady state probabilities of a
        Markov chain represented by the transition matrix stored in the `self.transition_matrix`
        attribute.
        """
        transition_matrix = np.array(transition_matrix).T
        n = transition_matrix.shape[0]
        
        Q = transition_matrix - np.eye(n)
        steady_state = null_space(Q).flatten()
        steady_state /= steady_state.sum()
        return steady_state