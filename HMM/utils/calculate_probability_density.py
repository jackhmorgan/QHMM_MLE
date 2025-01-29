from .calculate_steady_state import calculate_steady_state
from itertools import combinations_with_replacement
import numpy as np

def calculate_probability_density(transition_matrix, 
                                  volatilities,
                                  k):

    steady_state = calculate_steady_state(transition_matrix)
    ncl = transition_matrix.shape[0]

    # store sequence as a tuple of the state indexes, and the probability of that sequence
    current_probability_dictionary = {(vl,) : steady_state[vl] for vl in range(ncl)}
    
    # DP loop
    for _ in range(k-1):
        next_probability_dictionary = {}
        for sequence, probability in current_probability_dictionary.items():
            previous_state = int(sequence[-1])
            for state, transition_prob in enumerate(transition_matrix[previous_state]):
                new_sequence = sequence + (state,)
                new_probability = probability*transition_prob
                next_probability_dictionary[new_sequence] = new_probability
        current_probability_dictionary = next_probability_dictionary

    # given the spot volatility vl, what is the probability of being in a sequence that results in vi
    conditional_probability_densities = {vl : {np.sum(np.array(vls))/k : 0 for vls in combinations_with_replacement(volatilities, k)} for vl in range(ncl)}

    # add probability of each vl in each sequence
    for sequence, sequence_probability in current_probability_dictionary.items():
        integrated_volatility = np.sum([volatilities[int(spot_volatility)] for spot_volatility in sequence])
        integrated_volatility /= k
        for spot_volatility in sequence:
            closest_key = min(conditional_probability_densities[spot_volatility].keys(), key=lambda k: abs(k - integrated_volatility))
            conditional_probability_densities[spot_volatility][closest_key] += sequence_probability
        
    # Normalize the conditional probabilities
    for spot_volatility, probability_dictionary in conditional_probability_densities.items():
        normalization_constant = np.sum([prob for prob in probability_dictionary.values()])
        for integrated_volatility, probability in probability_dictionary.items():
            probability_dictionary[integrated_volatility] = probability/normalization_constant
    
    return conditional_probability_densities