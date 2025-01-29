from scipy.stats import ncx2, gamma

def calculate_pc_volatilities(theta, ncl):
    alpha, beta, sigma = theta
    a = 2*alpha*beta / (sigma**2)
    b = 2 *alpha / (sigma**2)
    dist = gamma(a=a, scale=1/b)
    return [dist.ppf(i/(ncl+1)) for i in range(1,ncl+1)]