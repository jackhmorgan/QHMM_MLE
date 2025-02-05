import numpy as np

def npc_theta_to_transition_matrix(theta: list | np.ndarray,
                                    ncl: int):
    """
    The function `npc_theta_to_transition_matrix` takes a list or NumPy array of transition
    probabilities and converts it into a transition matrix for a given number of latent states.
    The array of parameters theta holds the transition probability to all but the final state, which 
    is determined using the row-stochastic quality of the matrix.
    
    :param theta: The `theta` parameter is expected to be a list or numpy array containing transition
    probabilities for each state in a Markov chain. These probabilities should be between 0 and 1, and 
    the sum of each row should be less than 1.
    :type theta: list | np.ndarray
    :param ncl: The parameter `ncl` represents the number of latent states in the transition matrix. It is
    used to determine the dimensions of the transition matrix, which will be an ncl x ncl matrix
    :type ncl: int
    :return: The function `npc_theta_to_transition_matrix` returns a transition matrix based on the
    input `theta` and the number of latent states `ncl`.
    """
    matrix = np.zeros((ncl, ncl), dtype=np.float64)
    index = 0
    for row in range(ncl):
        for column in range(ncl-1):
            matrix[row][column] = theta[index]
            index += 1
        matrix[row][-1] = 1-np.sum(matrix[row])
    return matrix