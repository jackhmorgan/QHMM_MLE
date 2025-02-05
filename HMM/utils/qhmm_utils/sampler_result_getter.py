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

from .result_getter import result_getter
from qiskit_ibm_runtime import SamplerV2
from qiskit import transpile, QuantumCircuit
import numpy as np

# This class `sampler_result_getter` is designed to generate sequences from a quantum circuit using a
# sampler, and calculate the log likelihood of a given sequence based on the circuit execution
# results.
class sampler_result_getter(result_getter):
    def __init__(self,
                 sampler: SamplerV2,
                 num_shots: int =1000,
                 max_shots: int =10000):
        super().__init__()
        self.backend = sampler.backend()
        self.sampler = sampler
        self.num_shots = num_shots
        self.max_shots = max_shots

    def generate_sequence(self, 
                          circuit : QuantumCircuit):
        """
        The function generates a sequence of measurement outcomes from a quantum circuit executed on a
        backend.
        
        :param circuit: The `generate_sequence` method takes a quantum circuit as input
        and generates a sequence of measurement outcomes from running the circuit on a quantum backend.
        The method transpiles the circuit, runs it on the backend, retrieves the results, and then
        extracts a sequence of measurement outcomes from the results.
        :return: The `generate_sequence` function returns a list of integers representing the
        measurement outcomes of a quantum circuit that has been transpiled and executed on a backend
        using a sampler.
        """
        transpiled = transpile(circuit, self.backend)
        job = self.sampler.run([transpiled], shots=1)
        results = job.result()[0].data
        sequence = []
        for key in results.keys():
            sequence.append(int(results[key].array[0,0]))
        return sequence

    def log_likelihood(self, 
                       circuit : QuantumCircuit, 
                       sequence : list[int]):
        """
        This function calculates the log likelihood of a given quantum circuit generating a specific
        sequence of measurement outcomes.
        
        :param circuit: The `log_likelihood` method calculates the log likelihood
        of a given sequence based on the execution results of a quantum circuit on a backend. The method
        transpiles the circuit, runs it on the backend, and then compares the results to the expected
        sequence to calculate the likelihood
        :param sequence: The 'sequence' parameter is a list of integers corresponding to the measurement
        outcomes of each time step we are evaluating the log_likelihood of.
        :return: The `log_likelihood` method returns the natural logarithm of the probability calculated
        based on the number of shots in the circuit that match the given sequence. If the probability is
        0, it returns negative infinity.
        """
        super().log_likelihood(sequence)
        transpiled = transpile(circuit, self.backend)
        num_shots_circuit = 0
        num_shots_sequence = 0
        while num_shots_sequence == 0 and num_shots_circuit <= self.max_shots-self.num_shots:
            job = self.sampler.run([transpiled], shots=self.num_shots)
            result = job.result()[0]
            num_shots_sequence_job=0
            for shot in range(self.num_shots):
                is_sequence = True
                for i, step in enumerate(transpiled.cregs):
                    if not int(result.data[step.name].array[shot,0]) == sequence[i]:
                        is_sequence=False
                if is_sequence:
                    num_shots_sequence_job += 1
            num_shots_sequence += num_shots_sequence_job
            num_shots_circuit += self.num_shots
        probability = num_shots_sequence/num_shots_circuit
        
        if probability == 0:
            return float('-inf')
        
        else:
            return np.log(probability)