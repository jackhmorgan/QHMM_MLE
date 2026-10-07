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
import time
from scipy.optimize import minimize, LinearConstraint

# omega is O(1e-6) for daily returns in decimal units; the optimizer works on omega/OMEGA_SCALE so
# all three parameters are O(1) for the finite-difference gradient.
OMEGA_SCALE = 1e-6

def minimize_garch_hmm(model,
                       sequence : list,
                       theta_0 : list | np.ndarray,
                       max_iter : int = 100,
                       tol : float = 1e-6):
    """
    The function `minimize_garch_hmm` fits a GARCH_HMM by maximum likelihood with SLSQP, enforcing
    omega > 0, alpha > 0, beta >= 0 and alpha + beta < 1 (covariance stationarity).

    :param theta_0: The initial (omega, alpha, beta), in decimal-return units.
    :return: The function `minimize_garch_hmm` returns four values, as `minimize_pc_hmm` does:
    1. `trained_theta`: The optimized (omega, alpha, beta).
    2. `training_time`: The number of seconds taken for training the model.
    3. `nit`: The number of optimizer iterations.
    4. `training_curve`: The running best negative log-likelihood during training.
    """

    training_curve = []

    def to_theta(z):
        return [z[0] * OMEGA_SCALE, z[1], z[2]]

    def neg_log_likelihood(z):
        try:
            model.update_theta(to_theta(z))
            likelihood = model.log_likelihood(sequence)
        except (ValueError, np.linalg.LinAlgError):
            # Degenerate parameters (e.g. too few grid points per state); treat as infeasible
            return 1e10
        if not np.isfinite(likelihood):
            return 1e10
        if len(training_curve) == 0 or -likelihood < training_curve[-1]:
            training_curve.append(-likelihood)
        return -likelihood

    z_0 = [theta_0[0] / OMEGA_SCALE, theta_0[1], theta_0[2]]
    bounds = [(1e-6, None), (1e-6, 1), (0, 1)]
    stationarity = LinearConstraint([[0, 1, 1]], -np.inf, 1 - 1e-6)

    start_time = time.time()
    result = minimize(neg_log_likelihood,
                      z_0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=[stationarity],
                      tol=tol,
                      options = {'maxiter': max_iter},
                      )
    training_time = time.time() - start_time

    trained_theta = to_theta(result.x)
    nit = result.nit

    return trained_theta, training_time, nit, training_curve
