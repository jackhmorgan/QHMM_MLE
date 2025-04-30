import unittest
from ddt import ddt, data, unpack
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2

from HMM import QHMM  # Adjust path
from HMM.utils.qhmm_utils import statevector_result_getter


@ddt
class TestQHMMTrueLikelihood(unittest.TestCase):

    def setUp(self):
        # Initial state (superposition)
        self.initial_state = QuantumCircuit(1)
        self.initial_state.h(0)

        # Ansatz with two parameters
        self.ansatz = efficient_su2(2, reps=1, entanglement='pairwise', su2_gates=['ry'])

        self.qhmm = QHMM(
            result_getter=statevector_result_getter(),
            initial_state=self.initial_state,
            ansatz=self.ansatz,
        )

    @data(
        ([10.525141585117083, 13.10640697260181, 7.851319643368781, 14.096626513899015], 3),
        ([7.349274114289958, 18.763391543756633, 10.968501894680365, 12.80941094929162], 3),
    )
    @unpack
    def test_qhmm_real_sequence(self, theta, length):

        qhmm = self.qhmm

        qhmm.update_theta(theta=theta)

        sequence = qhmm.generate_sequence(length=length)
        self.assertEqual(len(sequence), length)
        self.assertTrue(all(bit in (0, 1) for bit in sequence))

    @data(
        # Format: (theta values, sequence, expected log likelihood)
        ([10.309768728110232, 16.780104028979764, 18.778602622231126, 6.579089601936728], 
         [1, 1, 0, 0],
         -3.2502622740622225 
        ),
        ([7.349274114289958, 18.763391543756633, 10.968501894680365, 12.80941094929162], 
         [1, 1, 1, 1], 
         -0.4242041218446928  
        ),
        (
            [10.525141585117083, 13.10640697260181, 7.851319643368781, 14.096626513899015], 
            [0, 1, 0, 1],
            -3.3095456809564228 # Both qubits in equal superposition, 1 has 0.5 amp → 0.5^2 = 0.25
        ),
    )
    @unpack
    def test_likelihood_matches_expected(self, theta, sequence, expected_log_likelihood):
        qhmm = self.qhmm

        qhmm.update_theta(theta=theta)

        log_likelihood = qhmm.log_likelihood(sequence, save_state=True)

        # Check it's close to expected (within ~1e-2)
        self.assertAlmostEqual(log_likelihood, expected_log_likelihood, places=2)


if __name__ == '__main__':
    unittest.main()