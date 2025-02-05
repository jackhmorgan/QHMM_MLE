from .result_getter import result_getter
from qiskit_ibm_runtime import SamplerV2
from qiskit import transpile
import numpy as np

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

    def generate_sequence(self, circuit):
        transpiled = transpile(circuit, self.backend)
        job = self.sampler.run([transpiled], shots=1)
        results = job.result()[0].data
        sequence = []
        for key in results.keys():
            sequence.append(int(results[key].array[0,0]))
        return sequence

    def log_likelihood(self, circuit, sequence):
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