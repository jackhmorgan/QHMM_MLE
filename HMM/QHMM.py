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
from .utils.qhmm_utils import result_getter
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
import numpy as np
# The `QHMM` class extends the `HMM` class and implements methods for constructing quantum circuits,
# updating parameters, generating sequences, and calculating log-likelihood in a quantum hidden Markov
# model.

class QHMM(HMM):
    def __init__(self,
                 result_getter: result_getter,
                 initial_state: QuantumCircuit,
                 ansatz: QuantumCircuit,
                 theta: list | np.ndarray | ParameterVector = None,
                 ):
        """
        This function initializes a quantum variational circuit with specified components and
        parameters.
        
        :param result_getter: The `result_getter` parameter is a function or object that is responsible
        for obtaining the results of quantum circuit measurements. It could be a function that executes
        the quantum circuit on a simulator or a quantum device and retrieves  and formats the measurement
        outcomes for the QHMM object.
        :type result_getter: result_getter
        :param initial_state: The `initial_state` parameter is a QuantumCircuit
        object representing the initial state of the latnet quantum system.
        :type initial_state: QuantumCircuit
        :param ansatz: The `ansatz` parameter in the `__init__` method is a QuantumCircuit object. It
        represents the quantum circuit that defines the variational circuit that will apply the unitary 
        operator in the unitary definition of a QHMM.
        :type ansatz: QuantumCircuit
        :param theta: The `theta` parameter is a list, numpy array, or
        ParameterVector that is used to initialize the quantum circuit object. It is optional and can be
        provided when creating an instance of the class, or applied later with the `update_theta` method.
        :type theta: list | np.ndarray | ParameterVector
        """
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
                          save_state=False) -> QuantumCircuit:
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

    def generate_sequence(self, 
                          length : int) -> list[int]:
        """
        The function generates a sequence of integers based on a given length using a constructed
        circuit and a result getter.
        
        :param length: The `length` parameter in the `generate_sequence` method represents the number of
        time steps in the sequence that you want to generate.
        :type length: int
        :return: The `generate_sequence` method is returning the result obtained by generating a
        sequence using the circuit constructed based on the given length.
        """
        super().generate_sequence(length)
        circuit = self.construct_circuit(length, save_state=False)
        result = self.result_getter.generate_sequence(circuit)
        return result
    
    def log_likelihood(self, sequence : list[int], save_state : bool=False):
        """
        This function calculates the log likelihood of a given sequence using a constructed circuit and
        the instance's result_getter.log_likelihood method.
        
        :param sequence: The `sequence` parameter is a list of integers that represents a sequence of
        observations. This method determines the likelihood of the provided sequence being produced by 
        by the model
        :type sequence: list[int]
        :param save_state: The `save_state` parameter in the `log_likelihood` method is a boolean flag
        that indicates whether the quantum state should be saved or measured. If an exact likelihood is calculated
        using the statevector simulator then this flag should be set to true, defaults to False
        :type save_state: bool (optional)
        :return: The `log_likelihood` method returns the log likelihood of a given sequence using a 
        constructed circuit and the `result_getter` object.
        """
        super().log_likelihood(sequence)
        if not save_state:
            if hasattr(self.result_getter, 'save_state'):
                save_state = self.result_getter.save_state
        length = len(sequence)
        circuit = self.construct_circuit(length, save_state=save_state)
        result = self.result_getter.log_likelihood(circuit, sequence)
        return result