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
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')


def minimize_GARCH(model,
                   sequence : list | np.ndarray,
                   p : int = None,
                   q : int = None,
                   max_iter : int = 200,
                   max_p : int = 3,
                   max_q : int = 3,
                   grid_search : bool = False):
    """
    Fits a GARCH model to a sequence using maximum likelihood estimation.
    
    This function can either:
    1. Fit a pre-configured GARCH model to the sequence (if p and q are provided via model)
    2. Perform a grid search over (p, q) specifications to find the best GARCH model
    
    :param model: The GARCH model instance to fit (or used as template for grid search)
    :type model: GARCH
    :param sequence: The observation sequence (time series data)
    :type sequence: list | np.ndarray
    :param p: AR order (optional, overrides model.p if provided)
    :type p: int
    :param q: MA order (optional, overrides model.q if provided)
    :type q: int
    :param max_iter: Maximum number of optimization iterations
    :type max_iter: int
    :param max_p: Maximum AR order to search when grid_search=True
    :type max_p: int
    :param max_q: Maximum MA order to search when grid_search=True
    :type max_q: int
    :param grid_search: Whether to perform grid search over (p, q) values
    :type grid_search: bool
    :return: Dictionary containing:
        - 'trained_model': The fitted GARCH model
        - 'fit_results': Dictionary with loglikelihood, aic, bic, params, convergence_flag
        - 'training_time': Time in seconds for fitting
        - 'p': AR order of the best model
        - 'q': MA order of the best model
        - 'training_curve': List of log-likelihoods (for compatibility with minimize_pc_hmm)
    :rtype: dict
    """
    
    if isinstance(sequence, list):
        sequence = np.array(sequence)
    
    if sequence.ndim > 1:
        sequence = sequence.flatten()
    
    start_time = time.time()
    
    if grid_search:
        # Grid search over (p, q) specifications
        best_bic = np.inf
        best_fit = None
        best_p = None
        best_q = None
        training_curve = []
        
        for p_val in range(1, max_p + 1):
            for q_val in range(1, max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        garch_model = arch_model(sequence,
                                                vol='Garch',
                                                p=p_val,
                                                q=q_val,
                                                mean=model.mean)
                        fitted_model = garch_model.fit(disp='off',
                                                      options={'maxiter': max_iter},
                                                      show_warning=False)
                    
                    bic = fitted_model.bic
                    training_curve.append(fitted_model.loglikelihood)
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_fit = {
                            'loglikelihood': fitted_model.loglikelihood,
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'params': fitted_model.params.to_dict(),
                            'convergence_flag': fitted_model.convergence_flag
                        }
                        best_p = p_val
                        best_q = q_val
                
                except Exception as e:
                    continue
        
        training_time = time.time() - start_time
        
        if best_fit is None:
            raise ValueError("Grid search failed to fit any GARCH model")
        
        return {
            'fit_results': best_fit,
            'training_time': training_time,
            'p': best_p,
            'q': best_q,
            'training_curve': training_curve,
            'grid_search': True
        }
    
    else:
        # Fit a single GARCH model
        p_val = p if p is not None else model.p
        q_val = q if q is not None else model.q
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            garch_model = arch_model(sequence,
                                    vol='Garch',
                                    p=p_val,
                                    q=q_val,
                                    mean=model.mean)
            fitted_model = garch_model.fit(disp='off',
                                          options={'maxiter': max_iter},
                                          show_warning=False)
        
        training_time = time.time() - start_time
        
        fit_results = {
            'loglikelihood': fitted_model.loglikelihood,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'params': fitted_model.params.to_dict(),
            'convergence_flag': fitted_model.convergence_flag
        }
        
        # Create a simple training curve (single point since arch handles optimization internally)
        training_curve = [fitted_model.loglikelihood]
        
        return {
            'fit_results': fit_results,
            'training_time': training_time,
            'p': p_val,
            'q': q_val,
            'training_curve': training_curve,
            'grid_search': False
        }
