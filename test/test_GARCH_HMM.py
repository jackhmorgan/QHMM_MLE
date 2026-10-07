import unittest
from ddt import ddt, data, unpack
import numpy as np
from hmmlearn.hmm import CategoricalHMM

from HMM import GARCH_HMM
from HMM.utils.garch_utils import minimize_garch_hmm

OBSERVATIONS = [-0.006313589141205697, -0.0010981613895532023, 0.0022960204279199436, 0.007239523585948188]


@ddt
class TestGARCH_HMM(unittest.TestCase):

    @data(
        ([1.9e-6, 0.10, 0.88], False),
        ([1.9e-6, 0.10, 0.88], True),
        ([1.76e-6, 0.0987, 0.8865], False),
    )
    @unpack
    def test_probabilities(self, theta, coupled):
        hmm = GARCH_HMM(theta=theta, ncl=16, observations=OBSERVATIONS, coupled=coupled)

        self.assertTrue(np.all(hmm.joint_matrix >= 0))
        self.assertTrue(np.allclose(hmm.joint_matrix.sum(axis=(1, 2)), 1))
        self.assertTrue(np.allclose(hmm.transition_matrix.sum(axis=1), 1))
        self.assertTrue(np.allclose(hmm.emission_matrix.sum(axis=1), 1))
        self.assertTrue(np.all(np.diff(hmm.variances) > 0))
        # States are cells of equal stationary probability
        self.assertTrue(np.allclose(hmm.steady_state, 1 / 16, atol=0.01))
        # No absorbing states
        self.assertTrue(np.all(np.diag(hmm.transition_matrix) < 0.95))

    def test_coupled_and_decoupled_share_marginals(self):
        theta = [1.9e-6, 0.10, 0.88]
        coupled = GARCH_HMM(theta=theta, ncl=16, observations=OBSERVATIONS, coupled=True)
        decoupled = GARCH_HMM(theta=theta, ncl=16, observations=OBSERVATIONS, coupled=False)

        self.assertTrue(np.allclose(coupled.transition_matrix, decoupled.transition_matrix))
        self.assertTrue(np.allclose(coupled.emission_matrix, decoupled.emission_matrix))
        self.assertTrue(np.allclose(decoupled.joint_matrix,
                                    decoupled.emission_matrix[:, :, None] * decoupled.transition_matrix[:, None, :]))
        self.assertFalse(np.allclose(coupled.joint_matrix, decoupled.joint_matrix))

    def test_decoupled_likelihood_matches_hmmlearn(self):
        hmm = GARCH_HMM(theta=[1.9e-6, 0.10, 0.88], ncl=16, observations=OBSERVATIONS)
        np.random.seed(0)
        sequence = hmm.generate_sequence(length=200)

        reference = CategoricalHMM(n_components=16, init_params='', params='')
        reference.startprob_ = hmm.steady_state
        reference.transmat_ = hmm.transition_matrix
        reference.emissionprob_ = hmm.emission_matrix

        self.assertAlmostEqual(hmm.log_likelihood(sequence), reference.score(sequence), places=8)

    @data(False, True)
    def test_generate_sequence(self, coupled):
        hmm = GARCH_HMM(theta=[1.9e-6, 0.10, 0.88], ncl=16, observations=OBSERVATIONS, coupled=coupled)
        sequence = hmm.generate_sequence(length=50)

        self.assertEqual(sequence.shape, (50, 1))
        self.assertTrue(all(bit in (0, 1, 2, 3) for bit in sequence.flatten()))

    @data(
        # Format: (theta values, sequence, coupled, expected log likelihood)
        ([1.9e-6, 0.10, 0.88], [0, 3, 3, 1, 2, 0], False, -8.10517499622565),
        ([1.9e-6, 0.10, 0.88], [0, 3, 3, 1, 2, 0], True, -8.157689188308321),
        ([1.76e-6, 0.0987, 0.8865], [1, 2, 1, 2, 0, 3], False, -9.07712403457921),
        ([1.76e-6, 0.0987, 0.8865], [1, 2, 1, 2, 0, 3], True, -9.031579319386196),
    )
    @unpack
    def test_likelihood_matches_expected(self, theta, sequence, coupled, expected_log_likelihood):
        hmm = GARCH_HMM(theta=theta, ncl=16, observations=OBSERVATIONS, coupled=coupled)

        self.assertAlmostEqual(hmm.log_likelihood(sequence), expected_log_likelihood, places=4)

    def test_minimize_improves_likelihood(self):
        true_theta = [1.9e-6, 0.10, 0.88]
        hmm = GARCH_HMM(theta=true_theta, ncl=8, observations=OBSERVATIONS)
        np.random.seed(1)
        sequence = hmm.generate_sequence(length=200)

        theta_0 = [3e-6, 0.05, 0.80]
        hmm.update_theta(theta_0)
        initial = hmm.log_likelihood(sequence)

        theta, _, _, training_curve = minimize_garch_hmm(model=hmm,
                                                         sequence=sequence,
                                                         theta_0=theta_0,
                                                         max_iter=20)
        hmm.update_theta(theta)

        self.assertGreater(hmm.log_likelihood(sequence), initial)
        self.assertLess(theta[1] + theta[2], 1)
        self.assertAlmostEqual(-training_curve[0], initial, places=6)


if __name__ == '__main__':
    unittest.main()
