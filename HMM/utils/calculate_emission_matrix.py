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

from .calculate_steady_state import calculate_steady_state
from .calculate_probability_density import calculate_probability_density
from .calculate_integrated_emission_probability import calculate_integrated_emission_probability
from itertools import combinations_with_replacement
import numpy as np

def calculate_emission_matrix(transition_matrix : list | np.ndarray, 
                              volatilities : list | np.ndarray,
                              observations : list | np.ndarray,
                              k : int,
                              mean : float = 0):
    """
    The function `calculate_emission_matrix` calculates the emission matrix based on transition
    probabilities, volatilities, observation bins, number of spot volatilities per time step, 
    and the mean of the normal distribution we are taking draws from.
    
    :param transition_matrix: The `transition_matrix` parameter represents the transition probabilities
    between different states in the latent Markovian process.
    :param volatilities: The `volatilities` parameter is a list or array of the volatility assocaiated with
    each latent state.
    :param observations: Observations are the centers of each observation bin that will be used to determine
    the emission probability
    :param k: The parameter `k` is the number of spot volitilities used to determine each integrated volatility.
    :param mean: The `mean` parameter in the `calculate_emission_matrix` function represents the mean
    of the normal distribution used to determine the emission probabilities for all volatilities.
    :return: The function `calculate_emission_matrix` returns an emission matrix that represents the
    probabilities of observing each state (spot volatility) given a sequence of observations and the
    model parameters.
    """
    ncl = transition_matrix.shape[0]
    num_observations = len(observations)
    steady_state = calculate_steady_state(transition_matrix=transition_matrix)

    probability_density = calculate_probability_density(transition_matrix,
                                                        volatilities,
                                                        k)
    integrated_emission_probability = calculate_integrated_emission_probability(volatilities=volatilities,
                                              observations=observations,
                                              k=k,
                                              mean=mean)
    # calculate integrated probability density
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

    emission_matrix = np.zeros((num_observations, ncl))

    for spot_volatility, probability_dictionary in probability_density.items():
        for integrated_volatility, probability in enumerate(probability_dictionary.values()):
            emission_matrix[:,spot_volatility] += probability*integrated_emission_probability[:,integrated_volatility]

    return emission_matrix