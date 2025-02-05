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