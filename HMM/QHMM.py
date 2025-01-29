from .HMM import HMM
from .utils.qhmm_utils import result_getter
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
import numpy as np

class QHMM(HMM):
    def __init__(self,
                 result_getter: result_getter,
                 initial_state: QuantumCircuit,
                 ansatz: QuantumCircuit,
                 theta: list | np.ndarray | ParameterVector = None,
                 ):
        self.result_getter = result_getter
        self.initial_state = initial_state
        self.ansatz = ansatz
        self.hidden_qubits = initial_state.num_qubits
        self.observed_qubits = ansatz.num_qubits - initial_state.num_qubits
        self.theta = None
        if not theta is None:
            self.update_theta(theta)

    def construct_circuit(self,
                          num_time_steps,
                          save_state=False):
        if self.theta is None:
            raise ValueError('theta must be set in order to construct a circuit')

        circ = QuantumCircuit()
        hidden_register = QuantumRegister(self.hidden_qubits, 'Hidden')
        circ.add_register(hidden_register)
        observed_register = QuantumRegister(self.observed_qubits, 'Observed')
        circ.add_register(observed_register)

        circ.append(self.initial_state, hidden_register)

        for step in range(num_time_steps):
            circ.barrier()
            circ.append(self.ansatz, circ.qubits)
            if save_state:
                circ.save_statevector(label='step_'+str(step))
            else:
                creg = ClassicalRegister(self.observed_qubits, "step_"+str(step))
                circ.add_register(creg)
                circ.measure(observed_register, creg)
            if not step == num_time_steps-1:
                circ.reset(observed_register)

        circ.assign_parameters(self.theta, inplace=True)

        return circ

    def update_theta(self,
                     theta):
        self.theta = theta

    def generate_sequence(self, length):
        super().generate_sequence(length)
        circuit = self.construct_circuit(length, save_state=False)
        result = self.result_getter.generate_sequence(circuit)
        return result
    
    def log_likelihood(self, sequence):
        super().log_likelihood(sequence)
        length = len(sequence)
        circuit = self.construct_circuit(length, save_state=True)
        result = self.result_getter.log_likelihood(circuit, sequence)
        return result