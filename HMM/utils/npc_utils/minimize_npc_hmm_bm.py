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


def minimize_npc_hmm_bm(model,
                        sequence : list,
                        max_iter : int = 100,
                        tol : float = 1e-6):
    """
    The function `minimize_npc_hmm_bm` optimizes a non-parameterized hidden Markov model using the 
    Baum-Welch algorithm by directly calling hmmlearn's fit function.
    
    :param model: An NPC_HMM model instance to train
    :type model: NPC_HMM
    :param sequence: The observation sequence to train on
    :type sequence: list
    :param max_iter: Maximum number of iterations for the Baum-Welch algorithm
    :type max_iter: int
    :param tol: Tolerance level for convergence
    :type tol: float
    :return: A tuple containing:
    1. `trained_theta`: The optimized theta values extracted from the trained transition matrix.
    2. `training_time`: The number of seconds taken for training the model.
    3. `n_iter`: The number of iterations performed by the Baum-Welch algorithm.
    4. `training_curve`: A list containing the log-likelihood values at each iteration during training.
    """
    
    # Convert sequence to the format expected by hmmlearn (n_samples, n_features)
    if isinstance(sequence, list):
        sequence = np.array(sequence).reshape(-1, 1)
    elif sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)
    
    # Start timing
    start_time = time.time()
    
    # Get the internal hmmlearn model from the NPC_HMM model
    hmm_model = model._model
    
    # Set the fit parameters
    hmm_model.n_iter = max_iter
    hmm_model.tol = tol
    
    # Fit using Baum-Welch algorithm (this is the standard fit method)
    hmm_model.fit(sequence, lengths=None)
    
    # End timing
    training_time = time.time() - start_time
    
    # Extract the number of iterations performed
    n_iter = hmm_model.n_iter_ if hasattr(hmm_model, 'n_iter_') else max_iter
    
    # Extract trained parameters from the fitted model
    trained_transmat = hmm_model.transmat_
    
    # Convert transition matrix back to theta format
    # theta contains all elements except the last column of each row
    trained_theta = []
    ncl = model.ncl
    for row in range(ncl):
        for column in range(ncl - 1):
            trained_theta.append(trained_transmat[row][column])
    
    # Get the training curve (log-likelihood at each iteration)
    # Note: hmmlearn doesn't expose per-iteration likelihood, so we compute it
    training_curve = [hmm_model.score(sequence)]
    
    return trained_theta, training_time, n_iter, training_curve
