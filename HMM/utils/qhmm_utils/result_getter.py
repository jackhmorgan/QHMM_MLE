from abc import ABC, abstractmethod

class result_getter(ABC):
    @abstractmethod
    def generate_sequence(circut):
        pass
    def log_likelihood(circuit, sequence):
        pass