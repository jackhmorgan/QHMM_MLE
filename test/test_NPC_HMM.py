import unittest
from ddt import ddt, data, unpack

from HMM import NPC_HMM


@ddt
class TestNPCMMTrueLikelihood(unittest.TestCase):

    def setUp(self):
        # Initial state (superposition)
        observations = [-0.03, 0.04]

        self.hmm = NPC_HMM(k=2, 
                          ncl=4,
                          observations=observations
        )

    @data(
        ([0.24999511446730874, 0.44054164765994347, 0.26430864981328733, 0.2305099407117467, 0.43682328169610857, 0.10490557895969388, 0.2093377597470978, 0.3619604523843043, 0.10035696342885003, 0.2385286820594175, 0.32156584835962815, 0.09039096424769398], 
         3),
        ([0.10392668931818926, 0.1777037898353118, 0.32727906100729354, 0.513745720180152, 0.1410865477143382, 0.0579793078546543, 0.1355180263810542, 0.3970891427779065, 0.2021663143271303, 0.32784529019651476, 0.29523410831811664, 0.1845893473812538],
        3),
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
        ([0.24999511446730874, 0.44054164765994347, 0.26430864981328733, 0.2305099407117467, 0.43682328169610857, 0.10490557895969388, 0.2093377597470978, 0.3619604523843043, 0.10035696342885003, 0.2385286820594175, 0.32156584835962815, 0.09039096424769398], 
         [0, 0, 0, 1],
         -2.7618284042806076
        ),
        ([0.10392668931818926, 0.1777037898353118, 0.32727906100729354, 0.513745720180152, 0.1410865477143382, 0.0579793078546543, 0.1355180263810542, 0.3970891427779065, 0.2021663143271303, 0.32784529019651476, 0.29523410831811664, 0.1845893473812538], 
         [0, 1, 1, 0], 
         -2.7726244318358084  
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