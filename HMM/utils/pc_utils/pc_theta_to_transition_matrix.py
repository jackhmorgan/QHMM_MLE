import numpy as np
from ..calculate_volatility_bins import calculate_volatility_bins
from scipy.stats import ncx2, gamma

def pc_theta_to_transition_matrix(theta, k, volatilities):
        alpha, beta, sigma = theta
        volatility_bins = calculate_volatility_bins(volatilities)
        ncl = len(volatilities)

        def calculate_transition_probabilities(V0):
            c = 2*alpha
            c /= sigma**2 * (1-np.exp(-alpha/k))

            # u = c*V0*np.exp(-self.alpha)
            # v = c*V1
            # q = 2*self.alpha*self.beta / (self.sigma**2)
            # q -= 1
            # bv = 2*np.sqrt(u*v)
            # iq = bessel(q, bv)
            # transition_probability = c * np.exp(-(u+v)) #/self.sigma**2
            # transition_probability *= (v/u)**(q/2)
            # transition_probability *= iq

            df = 4*alpha*beta / (sigma**2)
            nc = 2*c*V0*np.exp(-alpha/k)
            dist = ncx2(df, nc)

            transition_probabilities = []        
            prev_cdf = 0

            for bin in volatility_bins:
                cdf = dist.cdf(2*c*bin)
                transition_probabilities.append(cdf-prev_cdf)
                prev_cdf = cdf
            transition_probabilities.append(1-cdf)
            return transition_probabilities
        

        transition_matrix = np.zeros((ncl,ncl))
        
        for i, V0 in enumerate(volatilities):
            probabilities = calculate_transition_probabilities(V0)  
            transition_matrix[i] = probabilities

        return transition_matrix
    