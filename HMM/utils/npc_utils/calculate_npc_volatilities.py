from scipy.stats import gamma
import numpy as np

def calculate_npc_volatilities(ncl: int,
                               alpha: float,
                               delta: float):
    """
    The function calculates the volatility bins based on alpha, delta, and the number of volatility bins.
    The volatilities are distributed from e^(alpha-delta) to e^(alpha+delta)

    :param ncl: The `ncl` parameter represents the number of classical latent states
    :param alpha: The `alpha` parameter determining volatility values.
    :param delta: The `delta` parameter determining volatility values
    :return: The function `calculate_npc_volatilities` returns a list of calculated volatilities based
    on the input parameters `ncl`, `alpha`, and `delta`. The volatilities are calculated using the
    formula `np.exp(alpha+delta*e)` for each `e` value in the list `e_values`.
    """
    e_values = [(2*(i+1) - (ncl+1))/(ncl-1) for i in range(ncl)]
    return [np.exp(alpha+delta*e) for e in e_values]