from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

# The `result_getter` class is an abstract base class with methods for generating sequences and
# calculating log likelihood.
class result_getter(ABC):
    @abstractmethod
    def generate_sequence(circuit : QuantumCircuit):
        pass
    def log_likelihood(circuit : QuantumCircuit, 
                       sequence : list[int]):
        pass