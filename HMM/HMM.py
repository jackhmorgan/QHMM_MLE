from abc import ABC, abstractmethod

class HMM(ABC):
    @abstractmethod
    def log_likelihood(self, sequence):
        pass
    @abstractmethod
    def generate_sequence(self, length):
        pass