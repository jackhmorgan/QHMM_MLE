import numpy as np
from scipy.stats import entropy

def generate_sequences(model, n_sequences, length):
    """Generate sequences from an HMM model."""
    sequences = []
    for _ in range(n_sequences):
        seq = model.generate_sequence(length)
        if not type(seq, list):
            seq = seq.tolist()
        sequences.append(seq)
    return sequences

def calculate_kl_divergence(model1, model2, sequences):
    """Calculate the KL divergence between two HMM models using generated sequences."""
    log_prob_model1 = np.array([model1.log_likelihood(seq) for seq in sequences])
    log_prob_model2 = np.array([model2.log_likelihood(seq) for seq in sequences])
    
    # Convert log probabilities to probabilities
    prob_model1 = np.exp(log_prob_model1)
    prob_model2 = np.exp(log_prob_model2)
    
    # Normalize the probabilities to get a distribution
    prob_model1 /= prob_model1.sum()
    prob_model2 /= prob_model2.sum()
    
    # Calculate KL divergence
    kl_divergence = entropy(prob_model1, prob_model2)
    return kl_divergence

def kl_divergence_between_hmm(model1, model2, n_sequences=1000, sequence_length=10):
    """
    Calculate the KL divergence between two HMM models.
    
    Parameters:
    - model1: First HMM model
    - model2: Second HMM model
    - n_sequences: Number of sequences to generate
    - sequence_length: Length of each sequence
    
    Returns:
    - kl_divergence: The KL divergence between the two models
    """
    # Generate sequences from the first model
    sequences = generate_sequences(model1, n_sequences, sequence_length)
    
    # Calculate KL divergence
    kl_divergence = calculate_kl_divergence(model1, model2, sequences)
    
    return kl_divergence
