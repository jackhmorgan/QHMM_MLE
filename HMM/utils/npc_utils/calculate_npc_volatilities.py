from scipy.stats import gamma
import numpy as np

def calculate_npc_volatilities(ncl,
                               alpha,
                               delta):
    e_values = [(2*(i+1) - (ncl+1))/(ncl-1) for i in range(ncl)]
    return [np.exp(alpha+delta*e) for e in e_values]