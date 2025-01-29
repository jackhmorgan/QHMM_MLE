from .result_getter import result_getter
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

class qiskit_simulator_result_getter(result_getter):
    def __init__(self):
        self.simulator = AerSimulator()

    def generate_sequence(self, circuit):
        super().generate_sequence()
        transpiled = transpile(circuits=circuit, backend=self.simulator)
        result = self.simulator.run(transpiled, shots=1).result()
        sequence = [int(binary, 2) for binary in next(iter(result.get_counts())).split()]
        return sequence
    
    def log_likelihood(self, circuit, sequence):
        transpiled = transpile(circuit, backend=self.simulator)
        result = self.simulator.run(transpiled).result()
        likelihood = 1
        observed_qargs = [i for i in range(circuit.qregs[0].size, circuit.num_qubits)]
        for step, observation in enumerate(sequence):
            sv = result.data()['step_'+str(step)]
            probabilities = sv.probabilities(observed_qargs)
            prob = probabilities[observation]
            likelihood *= prob
        
        return np.log(likelihood)