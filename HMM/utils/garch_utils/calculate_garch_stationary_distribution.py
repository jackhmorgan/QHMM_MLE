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
from scipy.special import erf


def calculate_stationary_distribution(transition_matrix : np.ndarray):
    """
    Solves p (P - I) = 0 with sum(p) = 1 for the stationary distribution of a transition matrix.
    """
    n = transition_matrix.shape[0]
    system = transition_matrix.T - np.eye(n)
    system[-1, :] = 1
    rhs = np.zeros(n)
    rhs[-1] = 1
    try:
        stationary = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        # Several absorbing points (e.g. alpha close to 0); take the least-squares solution
        stationary = np.linalg.lstsq(system, rhs, rcond=None)[0]
    stationary = np.clip(stationary, 0, None)
    return stationary / stationary.sum()


def calculate_garch_stationary_distribution(theta : np.ndarray | list,
                                            n_grid : int = 1000,
                                            upper_multiple : float = 1000.0):
    """
    The function computes the stationary distribution of the GARCH(1,1) conditional variance on a
    fine deterministic grid.

    The variance X_{t+1} = omega + (alpha*w_t^2 + beta)*X_t, w_t ~ N(0,1), is a Markov chain whose
    stationary distribution has no closed form. The grid is log-spaced from the lower end of the
    stationary support, omega/(1-beta), to `upper_multiple` times the unconditional variance, and the
    transition probabilities between grid cells are exact chi-squared(1) probabilities.

    :param theta: The GARCH(1,1) parameters omega, alpha and beta, for returns in decimal units.
    Requires omega > 0, alpha > 0, beta >= 0 and alpha + beta < 1.
    :type theta: np.ndarray | list
    :param n_grid: The number of grid points.
    :type n_grid: int
    :param upper_multiple: The top of the grid as a multiple of the unconditional variance.
    :type upper_multiple: float
    :return: The grid points, the n_grid-1 cell edges between them, and the stationary probability of
    each grid point.
    """
    omega, alpha, beta = theta
    unconditional_variance = omega / (1 - alpha - beta)
    lower = omega / (1 - beta)
    grid = np.geomspace(lower, unconditional_variance * upper_multiple, n_grid)
    edges = np.sqrt(grid[:-1] * grid[1:])

    # P(X_{t+1} <= edge | X_t = x) = P(w^2 <= c) = erf(sqrt(c/2)), c = (edge - omega - beta*x)/(alpha*x)
    threshold = (edges[None, :] - omega - beta * grid[:, None]) / (alpha * grid[:, None])
    cdf = erf(np.sqrt(np.clip(threshold, 0, None) / 2))
    cdf = np.concatenate([np.zeros((n_grid, 1)), cdf, np.ones((n_grid, 1))], axis=1)
    transition_matrix = np.diff(cdf, axis=1)

    return grid, edges, calculate_stationary_distribution(transition_matrix)
