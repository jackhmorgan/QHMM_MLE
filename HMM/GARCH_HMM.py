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
from .utils.garch_utils.garch_theta_to_joint_matrix import garch_theta_to_joint_matrix
from .utils.garch_utils.calculate_garch_stationary_distribution import calculate_stationary_distribution
import numpy as np

# The `GARCH_HMM` class extends `HMM` and implements a GARCH(1,1) discretized to `ncl` latent variance
# states, the GARCH counterpart of `PC_HMM` (which discretizes the CIR process).
class GARCH_HMM(HMM):
    def __init__(self,
                 theta=None,
                 ncl=None,
                 observations=None,
                 coupled=False,
                 n_grid=1000,
    ):
        """
        This Python function initializes attributes `ncl`, `observations` and `coupled`, and updates
        `theta` if it is not None.

        :param theta: The GARCH(1,1) parameters (omega, alpha, beta) for returns in decimal units.
        :param ncl: The `ncl` parameter determines the number of latent variance states.
        :param observations: The `observations` parameter determines the center of the observable bins
        associated with each emitted state.
        :param coupled: If False (default) the model is a standard HMM like `PC_HMM` and `NPC_HMM`: the
        emission and the transition are independent given the state. If True the observation and the
        next state are drawn jointly from the same GARCH shock, which is exact GARCH(1,1) behaviour
        but no longer a standard HMM.
        :param n_grid: The number of fine-grid points used to compute the stationary variance
        distribution; each latent state is a cell of this grid with probability 1/ncl. It should be
        well above `ncl`.
        """
        super().__init__()
        self.ncl = ncl
        self.observations = observations
        self.coupled = coupled
        self.n_grid = n_grid

        if not theta is None:
            self.update_theta(theta)

    def update_theta(self,
                     theta):
        joint_matrix, variances = garch_theta_to_joint_matrix(theta=theta,
                                                              ncl=self.ncl,
                                                              observations=self.observations,
                                                              n_grid=self.n_grid)
        emission_matrix = joint_matrix.sum(axis=2)
        transition_matrix = joint_matrix.sum(axis=1)
        if not self.coupled:
            joint_matrix = emission_matrix[:, :, None] * transition_matrix[:, None, :]

        self.theta = list(theta)
        self.variances = variances
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix
        self.joint_matrix = joint_matrix
        self.steady_state = calculate_stationary_distribution(transition_matrix)

    def log_likelihood(self, sequence):
        """
        Forward algorithm with per-step normalization, so long sequences do not underflow.
        """
        sequence = np.asarray(sequence).flatten().astype(int)
        super().log_likelihood(sequence)
        forward = self.steady_state.copy()
        log_likelihood = 0.0
        for observation in sequence:
            forward = forward @ self.joint_matrix[:, observation, :]
            scale = forward.sum()
            if scale <= 0:
                return float('-inf')
            log_likelihood += np.log(scale)
            forward /= scale
        return float(log_likelihood)

    def generate_sequence(self, length):
        super().generate_sequence(length)
        ncl = len(self.variances)
        state = np.random.choice(ncl, p=self.steady_state / self.steady_state.sum())
        sequence = np.zeros((length, 1), dtype=int)
        for t in range(length):
            probabilities = self.joint_matrix[state].ravel()
            index = np.random.choice(probabilities.size, p=probabilities / probabilities.sum())
            sequence[t, 0], state = divmod(index, ncl)
        return sequence
