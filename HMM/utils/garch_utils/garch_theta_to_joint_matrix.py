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
from scipy.special import ndtr
from .calculate_garch_stationary_distribution import calculate_garch_stationary_distribution


def calculate_joint_probabilities(theta, variances, observations, next_state_edges):
    """
    For each starting variance x, the joint probability of (observation bin j, next variance in
    cell k). With Y_t = sqrt(x)*w_t and X_{t+1} = omega + beta*x + alpha*x*w_t^2, both are functions
    of the same shock w_t, so each entry is the standard normal probability of the set of w that
    puts Y_t in bin j and X_{t+1} in cell k.

    :param variances: The starting variances, shape (n,).
    :param observations: The centers of the observation bins; bin edges are the midpoints between
    consecutive centers, as in `calculate_integrated_emission_probability`.
    :param next_state_edges: The interior edges of the next-state cells, increasing.
    :return: An array of shape (n, n_observations, n_cells).
    """
    omega, alpha, beta = theta
    x = np.asarray(variances, dtype=float)[:, None]
    observations = np.asarray(observations, dtype=float)
    n = len(x)

    # Observation bin j <=> w in [w_lo, w_hi]
    observation_edges = (observations[:-1] + observations[1:]) / 2
    w_edges = observation_edges[None, :] / np.sqrt(x)
    w_edges = np.concatenate([np.full((n, 1), -np.inf), w_edges, np.full((n, 1), np.inf)], axis=1)
    w_lo = w_edges[:, :-1, None]
    w_hi = w_edges[:, 1:, None]

    # Next cell k <=> X_{t+1} in [edge_k, edge_{k+1}) <=> |w| in [s_lo, s_hi]
    s = np.sqrt(np.clip((np.asarray(next_state_edges)[None, :] - omega - beta * x) / (alpha * x), 0, None))
    s = np.concatenate([np.zeros((n, 1)), s, np.full((n, 1), np.inf)], axis=1)
    s_lo = s[:, None, :-1]
    s_hi = s[:, None, 1:]

    positive = np.clip(ndtr(np.minimum(w_hi, s_hi)) - ndtr(np.maximum(w_lo, s_lo)), 0, None)
    negative = np.clip(ndtr(np.minimum(w_hi, -s_lo)) - ndtr(np.maximum(w_lo, -s_hi)), 0, None)
    return positive + negative


def garch_theta_to_joint_matrix(theta : tuple | list | np.ndarray,
                                ncl : int,
                                observations : list | np.ndarray,
                                n_grid : int = 1000):
    """
    The function `garch_theta_to_joint_matrix` discretizes a GARCH(1,1) into `ncl` latent variance
    states and returns, for each state, the joint probability of the observation bin and the next
    state.

    The states are `ncl` cells of the stationary variance distribution with equal stationary
    probability. Each cell's probabilities are the stationary-weighted average over the fine grid
    points inside it (lumping). Evaluating the transitions at a single point per state instead loses
    the downward drift when cells are wide compared with one day's change in variance, and can make
    the top state absorbing.

    :param theta: The GARCH(1,1) parameters omega, alpha and beta (decimal-return units).
    :param ncl: The number of latent states.
    :param observations: The centers of the observation bins.
    :param n_grid: The number of fine grid points; it should be well above `ncl`.
    :return: The joint matrix of shape (ncl, n_observations, ncl), whose [i, j, k] entry is
    P(observation bin j, next state k | current state i), and the stationary mean variance of each
    state. Summing the joint matrix over j gives the transition matrix and over k the emission
    matrix (states as rows).
    """
    grid, edges, stationary = calculate_garch_stationary_distribution(theta, n_grid=n_grid)

    # Assign each fine point to the cell containing the midpoint of its cumulative probability
    cumulative = np.cumsum(stationary)
    cell = np.minimum((ncl * (cumulative - stationary / 2)).astype(int), ncl - 1)
    cell_mass = np.bincount(cell, weights=stationary, minlength=ncl)
    if np.any(cell_mass == 0):
        raise ValueError(f"n_grid={n_grid} is too coarse for ncl={ncl}; increase n_grid.")

    # Cells are contiguous runs of fine points, so the next-state edges are fine edges at boundaries
    boundaries = np.flatnonzero(np.diff(cell))
    next_state_edges = edges[boundaries]

    fine_joint = calculate_joint_probabilities(theta, grid, observations, next_state_edges)
    weighted = fine_joint * stationary[:, None, None]
    joint_matrix = np.zeros((ncl,) + fine_joint.shape[1:])
    np.add.at(joint_matrix, cell, weighted)
    joint_matrix /= cell_mass[:, None, None]

    variances = np.bincount(cell, weights=stationary * grid, minlength=ncl) / cell_mass
    return joint_matrix, variances
