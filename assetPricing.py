from math import exp, sqrt, log
import numpy as np
from scipy.stats import norm

def blackScholes(s, x, sigma, r, t, q = 0.0, optionType = 'c'): 
    d1 = 1 / (sigma * np.sqrt(t)) * (np.log(s / x) + (r - q + 0.5 * sigma ** 2) * t)
    d2 = d1 - sigma * np.sqrt(t)

    if optionType == 'c': 
        call = s * np.exp(-q * t) * norm.cdf(d1) - x * np.exp(-r * t) * norm.cdf(d2)
        return call
    elif optionType == 'p': 
        put = x * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)
        return put