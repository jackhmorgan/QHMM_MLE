from .HMM import HMM
from .utils.pc_utils import calculate_pc_volatilities, pc_theta_to_transition_matrix
from .utils import calculate_emission_matrix, calculate_steady_state
from hmmlearn.hmm import CategoricalHMM

# The `PC_HMM` class extends `HMM` and implements methods for updating model parameters, calculating
# log likelihood, and generating sequences based on a hidden Markov model with principal component
# analysis.
class PC_HMM(HMM):
    def __init__(self,
                 theta=None,
                 k=1,
                 ncl=None,
                 observations=None,
    ):
        """
        This Python function initializes attributes `k`, `observations`, and `ncl`, and updates `theta`
        if it is not None.
        
        :param theta: Theta is a parameter that can be passed to the constructor of the class. If a
        value is provided for theta when creating an instance of the class, the `update_theta` method
        will be called with that value to update the theta attribute of the instance
        :param k: The `k` parameter determines the number of spot volatilities for every integrated
        volatility, defaults to 1.
        :param ncl: The `ncl` parameter determines the number of latent states.
        :param observations: The `observations` parameter determines the center of the observable bins
        associated with each emitted state.
        """
        super().__init__()
        self.k = k
        self.observations = observations
        self.ncl = ncl

        if not theta is None:
            self.update_theta(theta)

    def update_theta(self,
                     theta):
        volatilities = calculate_pc_volatilities(theta=theta,
                                                 ncl=self.ncl)
        transition_matrix = pc_theta_to_transition_matrix(theta=theta,
                                                          volatilities=volatilities,
                                                          k = self.k)
        steady_state = calculate_steady_state(transition_matrix)
        emission_matrix = calculate_emission_matrix(transition_matrix=transition_matrix,
                                                    volatilities=volatilities,
                                                    observations = self.observations,
                                                    k = self.k)
        
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
        