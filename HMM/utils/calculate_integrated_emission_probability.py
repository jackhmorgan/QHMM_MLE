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

import numpy as np
from scipy.stats import norm
from itertools import combinations_with_replacement

def calculate_integrated_emission_probability(volatilities,
                                              observations,
                                              k,
                                              mean=0):
        """
        The function calculates the integrated emission probability based on scaled integrated
        volatilities and observation edges.
        :return: The function `calculate_integrated_emission_probability` returns a numpy array
        `integrated_emission_probabilty` which contains the calculated emission probabilities for each
        observation bin and integrated volatility.
        """
        num_observation_bins = len(observations)
        observation_edges = [(observations[i] + observations[i+1])/2 for i in range(len(observations)-1)]
        integrated_volatilities = [np.sum(np.array(svs))/k for svs in combinations_with_replacement(volatilities, k)]
        integrated_emission_probabilty = np.zeros((num_observation_bins, len(integrated_volatilities)))

        for i, integrated_volatility in enumerate(integrated_volatilities):
            # calculate std
            sigma = np.sqrt(integrated_volatility)

            #emission_probs = [norm.pdf(value, scale=sigma) for value in self.observation_values]
            #emission_probs = emission_probs / np.sum(emission_probs)
            # -inf cdf
            prev_cdf = 0.0
            emission_probs = []
            # cdf of the bin
            for bin in observation_edges:
                current_cdf = norm.cdf(bin, loc=mean, scale = sigma)# - norm.cdf(-bin, scale = sigma)
                emission_probabilty = current_cdf - prev_cdf
                emission_probs.append(emission_probabilty)
                prev_cdf = current_cdf

            emission_probs.append(1-prev_cdf)
            integrated_emission_probabilty[:,i] = emission_probs

        return integrated_emission_probabilty