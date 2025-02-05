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

from scipy.optimize import minimize
import time
import numpy as np
from ...QHMM import QHMM

def minimize_qhmm(model : QHMM,
                  sequence : list,
                  theta_0 : list | np.ndarray,
                  max_iter : int = 100,
                  tol : float = 1e-6):
    """
    The function `minimize_qhmm` optimizes a quantum hidden Markov model using a given model, sequence data,
    initial parameters, maximum iterations, and tolerance level.
    
    :param theta: The `theta` parameter in the `minimize_qhmm` function represents the initial values
    of the parameters that will be optimized during the training process. These parameters are specific
    to the quantm hidden Markov model (QHMM) being used in the function. The optimization algorithm will adjust
    these parameters to maximize
    :return: The function `minimize_qhmm` returns three values: 
    1. `trained_theta`: The optimized theta values for the model after training on the given sequence.
    2. `training_time`: The number of seconds taken for training the model.
    3. `training_curve`: A list containing the negative log-likelihood values at each iteration during
    training.
    """
    
    training_curve = []

    def neg_log_likelihood(theta):
        penalty = 0
        likelihood = 0
        penalizer = len(sequence)**4
        for param in theta:
            if param < 0:
                penalty += (1-param)*penalizer
            
            if param > 8*np.pi:
                penalty += (1+param-8*np.pi)*penalizer

        if penalty == 0:
            model.update_theta(theta)
            likelihood = model.log_likelihood(sequence)
        
        if not likelihood == 0:
            training_curve.append(-likelihood)

        return -likelihood + penalty
    
    start_time = time.time()
    # Optimize
    result = minimize(neg_log_likelihood, 
                      theta_0, 
                      method='Nelder-Mead',
                      tol=tol,
                      options = {'maxiter': max_iter},
                      )
    training_time = time.time() - start_time
    
    trained_theta = result.x.tolist()

    return trained_theta, training_time, training_curve