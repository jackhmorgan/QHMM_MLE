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

from scipy.stats import gamma
import numpy as np

def calculate_npc_volatilities(ncl: int,
                               alpha: float,
                               delta: float):
    """
    The function calculates the volatility bins based on alpha, delta, and the number of volatility bins.
    The volatilities are distributed from e^(alpha-delta) to e^(alpha+delta)

    :param ncl: The `ncl` parameter represents the number of classical latent states
    :param alpha: The `alpha` parameter determining volatility values.
    :param delta: The `delta` parameter determining volatility values
    :return: The function `calculate_npc_volatilities` returns a list of calculated volatilities based
    on the input parameters `ncl`, `alpha`, and `delta`. The volatilities are calculated using the
    formula `np.exp(alpha+delta*e)` for each `e` value in the list `e_values`.
    """
    e_values = [(2*(i+1) - (ncl+1))/(ncl-1) for i in range(ncl)]
    return [np.exp(alpha+delta*e) for e in e_values]