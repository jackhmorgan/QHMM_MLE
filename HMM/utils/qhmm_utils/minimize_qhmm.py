from scipy.optimize import minimize
import time
import numpy as np

def minimize_qhmm(model,
                   sequence,
                   theta_0,
                   max_iter,
                   tol):
    
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