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

from .HMM import HMM

import numpy as np
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')


class GARCH(HMM):
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model 
    for modeling volatility in financial time series.
    
    This class follows the HMM interface and uses the GARCH model to predict
    log-likelihood and generate sequences.
    """
    
    def __init__(self,
                 p : int = 1,
                 q : int = 1,
                 mean : str = 'Zero',
                 vol : str = 'Garch',
                 observations : list | np.ndarray = None):
        """
        Initialize the GARCH model.
        
        :param p: The lag order of the GARCH component (AR order)
        :type p: int
        :param q: The lag order of the GARCH component (MA order)
        :type q: int
        :param mean: The mean model ('Zero', 'Constant', 'AR', 'ARX')
        :type mean: str
        :param vol: The volatility model ('Garch', 'EGARCH', 'ConstantMean')
        :type vol: str
        :param observations: Observations or returns data
        :type observations: list | np.ndarray
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.observations = observations
        
        # Initialize the GARCH model
        if observations is not None:
            observations_array = np.array(observations)
            if observations_array.ndim > 1:
                observations_array = observations_array.flatten()
            self._model = arch_model(observations_array, 
                                    vol='Garch', 
                                    p=p, 
                                    q=q,
                                    mean=mean)
            self._fitted_model = None
        else:
            self._model = None
            self._fitted_model = None
    
    def fit(self, 
            disp : str = 'off',
            max_iter : int = 200,
            show_warning : bool = False) -> dict:
        """
        Fit the GARCH model to the observations using maximum likelihood estimation.
        
        :param disp: Display level ('off', 'final', or 'all')
        :type disp: str
        :param max_iter: Maximum number of iterations for the optimizer
        :type max_iter: int
        :param show_warning: Whether to show warnings during fitting
        :type show_warning: bool
        :return: Dictionary containing fit results including parameters and fit information
        :rtype: dict
        """
        if self._model is None:
            raise ValueError("Model not initialized. Provide observations during initialization.")
        
        if not show_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fitted_model = self._model.fit(disp=disp, 
                                                    options={'maxiter': max_iter},
                                                    show_warning=False)
        else:
            self._fitted_model = self._model.fit(disp=disp, 
                                                options={'maxiter': max_iter},
                                                show_warning=show_warning)
        
        # Return fit summary
        return {
            'loglikelihood': self._fitted_model.loglikelihood,
            'aic': self._fitted_model.aic,
            'bic': self._fitted_model.bic,
            'params': self._fitted_model.params.to_dict(),
            'convergence_flag': self._fitted_model.convergence_flag
        }
    
    def log_likelihood(self, sequence : list | np.ndarray) -> float:
        """
        Calculate the log-likelihood of a sequence under the fitted GARCH model.
        
        :param sequence: The observation sequence
        :type sequence: list | np.ndarray
        :return: The log-likelihood value
        :rtype: float
        """
        if isinstance(sequence, list):
            sequence = np.array(sequence)
        
        if sequence.ndim > 1:
            sequence = sequence.flatten()
        
        if self._fitted_model is None:
            # If not fitted, fit first
            if self._model is None:
                self._model = arch_model(sequence, 
                                        vol='Garch', 
                                        p=self.p, 
                                        q=self.q,
                                        mean=self.mean)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fitted_model = self._model.fit(disp='off', show_warning=False)
        
        # Calculate log-likelihood for the given sequence
        # Re-fit on the sequence to get its likelihood
        temp_model = arch_model(sequence, 
                               vol='Garch', 
                               p=self.p, 
                               q=self.q,
                               mean=self.mean)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_fit = temp_model.fit(disp='off', show_warning=False)
        
        return temp_fit.loglikelihood
    
    def generate_sequence(self, length : int) -> np.ndarray:
        """
        Generate a synthetic sequence from the fitted GARCH model.
        
        :param length: The length of the sequence to generate
        :type length: int
        :return: The generated sequence
        :rtype: np.ndarray
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before generating sequences. Call fit() first.")
        
        # Simulate from the fitted model
        simulations = self._fitted_model.simulate(nobs=length, nsimulations=1, seed=None)
        
        # Extract the simulated returns (first column)
        return simulations.values.flatten()
    
    def get_parameters(self) -> dict:
        """
        Get the estimated parameters of the fitted GARCH model.
        
        :return: Dictionary of model parameters
        :rtype: dict
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        return self._fitted_model.params.to_dict()
    
    def get_volatility(self) -> np.ndarray:
        """
        Get the estimated volatility (conditional standard deviation) from the fitted model.
        
        :return: Array of estimated volatilities
        :rtype: np.ndarray
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        return self._fitted_model.conditional_volatility.values
    
    def get_residuals(self) -> np.ndarray:
        """
        Get the standardized residuals from the fitted model.
        
        :return: Array of standardized residuals
        :rtype: np.ndarray
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        return self._fitted_model.std_resid.values
