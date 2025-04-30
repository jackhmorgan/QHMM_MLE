import unittest
from ddt import ddt, data, unpack

from HMM import PC_HMM


@ddt
class TestPCMMTrueLikelihood(unittest.TestCase):

    def setUp(self):
        # Initial state (superposition)
        observations = [-0.03, 0.04]

        self.hmm = PC_HMM(k=2, 
                          ncl=4,
                          observations=observations
        )

    @data(
        ([2.5, 0.7, 1.1], 3),
        ([5.5, 0.3, 0.9], 3),
    )
    @unpack
    def test_qhmm_real_sequence(self, theta, length):

        hmm = self.hmm

        hmm.update_theta(theta=theta)

        sequence = hmm.generate_sequence(length=length)
        self.assertEqual(len(sequence), length)
        self.assertTrue(all(bit in (0, 1) for bit in sequence))

    @data(
        # Format: (theta values, sequence, expected log likelihood)
        ([2.5, 0.7, 1.1], 
         [1, 0, 0, 0],
         -2.759039227248806 
        ),
        ([5.5, 0.3, 0.9], 
         [0, 1, 0, 1], 
         -2.7730317503362834  
        ),
    )
    @unpack
    def test_likelihood_matches_expected(self, theta, sequence, expected_log_likelihood):
        hmm = self.hmm

        hmm.update_theta(theta=theta)

        log_likelihood = hmm.log_likelihood(sequence)

        # Check it's close to expected (within ~1e-2)
        self.assertAlmostEqual(log_likelihood, expected_log_likelihood, places=4)


if __name__ == '__main__':
    unittest.main()