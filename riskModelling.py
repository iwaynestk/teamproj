from multiprocessing import Value
from numpy.core.arrayprint import str_format
from numpy.core.fromnumeric import std
import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
from scipy.stats.morestats import anderson_ksamp
from wiener import wiener


def gbmVaR(v0, mu, sigma, p, t, position = 'long'): 
    """
    Calculate VaR of a single stock following GBM. 
    This function supports long and short position. 

    Args: 
        v0: initial portfolio value
        mu: drift
        sigma: volatility
        p: percentile to calculate VaR (Must be within (0, 1).  Example: 99% VaR: p = 0.99)
        t: period of time to look ahead (in years. Example: 5 days -> t = 5/252)
        position: string 'long' or 'short'. 'long' by default

    Returns: 
        A double value of VaR corresponding to that stock. 
    """

    if position == 'long': 
        VaR = v0 - v0 * exp(sigma * sqrt(t) * norm.ppf(1 - p) + (mu - 0.5 * sigma**2) * t)
        return VaR
    elif position == 'short': 
        VaR = -v0 + v0 * exp(sigma * sqrt(t) * norm.ppf(p) \
            + (mu - sigma ** 2 / 2) * t)
        return VaR
    else: 
        raise ValueError("position can only be \'long\' or \'short\'")
    # unit test passed



def gbmES(v0, mu, sigma, p, t, position = 'long'): 
    """
    Calculate expected shortfall of a single stock following GBM. 
    This function supports both long and short positions. 

    Args: 
        v0: initial portfolio value
        mu: drift
        sigma: volatility
        p: percentile to calculate VaR (Must be within (0, 1).  Example: 99% ES: p = 0.99)
        t: period of time to look ahead (in years. Example: 5 days -> t = 5/252)
        position: string 'long' or 'short'. 'long' by default

    Returns: 
        A double value of ES corresponding to that stock. 

    """

    if position == 'long': 
        ES = v0 * (1 - exp(mu*t) / (1 - p) \
            * norm.cdf(norm.ppf(1 - p) - sqrt(t)*sigma))
        return ES
    elif position == 'short': 
        ES = (-gbmES(v0, mu, sigma, 0.0, t, 'long') + p*gbmES(v0, mu, sigma, 1 - p, t, 'long')) \
            / (1 - p)
        return ES
    else: 
        raise ValueError("position can only be 'long' or 'short'. ")


        
def parametricVaR(a, s, mu, sigma, covm, p, t, position = 'long'): 
    """
    Compute the VaR of a portfolio of stocks that follows correlated GBMs by assuming
    that the portfolio is normally distributted. 

    Args: 
        a: position vector (number of shares of each stock)
        s: initial stock price 
        mu: drift vector
        sigma: volatility vector
        covm: covariance matrix
        p: percentile to calculate VaR (99% VaR: p = 0.99)
        t: period of time to look ahead (in years)
        position: string 'long' or 'short'. 'long' by default

    Returns: 
        A double value which represents value at risk. 
    """
    # position values: 
    a_s = a * s

    # V_0
    v0 = a_s.sum()

    # E[V_t]
    # in numpy, @ means matrix multiplication
    evt = a_s @ np.transpose(np.exp(mu * t)) 

    # E[V_t^2]
    evt2 = np.exp(mu * t) @ (np.exp(covm * t) * (np.transpose(a_s) @ a_s)) @ np.transpose(np.exp(mu * t))

    #var[V_t]
    varvt = evt2 - evt ** 2

    # sd(V_t)
    sdvt = np.sqrt(varvt)

    if position == 'long': 
        VaR = v0 - (evt + norm.ppf(1 - p) * sdvt)
        return VaR
    elif position == 'short': 
        VaR = -v0 - (-evt + norm.ppf(p) * sdvt)
        return VaR
    else: 
        raise ValueError("Only 'long' or 'short' positions are allowed. ")



def parametricES(a, s, mu, sigma, covm, p, t, position = 'long'): 
    """
    Compute the ES of a portfolio of stocks that follows correlated GBMs by assuming
    the portfolio is normally distributted. 

    Args: 
        a: position vector (number of shares of each stock)
        s: initial stock price 
        mu: drift vector
        sigma: volatility vector
        covm: covariance matrix
        p: percentile to calculate VaR (99% ES: p = 0.99)
        t: period of time to look ahead (in years)
        position: string 'long' or 'short'. 'long' by default

    Returns: 
        A double value which represents the expected shortfall. 
    """
    # position values: 
    a_s = a * s

    # V_0
    v0 = a_s.sum()

    # E[V_t]
    # in numpy, @ means matrix multiplication
    evt = a_s @ np.transpose(np.exp(mu * t)) 

    # E[V_t^2]
    evt2 = np.exp(mu * t) @ (np.exp(covm * t) * (np.transpose(a_s) @ a_s)) @ np.transpose(np.exp(mu * t))

    #var[V_t]
    varvt = evt2 - evt ** 2

    # sd(V_t)
    sdvt = np.sqrt(varvt)

    if position == 'long':
        ES = v0 - evt + sdvt*norm.pdf(norm.ppf(p))/(1 - p)
        return ES
    elif position == 'short': 
        ES = -v0 + evt + sdvt/(1 - p)*(norm.pdf(norm.ppf(p)))
        return ES
    else: 
        raise ValueError("Only 'long' or 'short' positions are allowed. ")



def gbmPrice(s, mu, sigma, t, a = None, covm = None): 
    """
    Calculate projected future value of a portfolio/stock based on one sample path, 
    assuming that it follows GBM. 

    Args: 
        v0: initial stock price
        mu: drift vector
        sigma: volatility vector
        t: time steps (in years)

    Returns: 
        A double value which corresponds to the projected value in time t
    """

    bm = wiener(1)
    v0 = s
    vt = v0 * np.exp((mu - 0.5*sigma**2)*t + sigma*np.sqrt(t)*bm.process)
    return vt



def MCVaR(s, v0, mu, sigma, p, t, trials = 1000, position = 'long'): 
    """
    Use Monte-Carlo simulation to generate a set of paths following GBM. 
    Use the projected price from sample paths to calculate value at risk. 

    Args: 
        s: initial stock price 
        mu: drift
        sigma: volatility
        p: percentile (example: 99% VaR -> p = 0.99)
        t: time steps (in years)
        trials: number of trials for simulations. 1000 by default. 
        position: string 'long' or 'short'. 'long' by default
    
    Returns: 
        A double value which represents VaR based on simulations. 
    """
    if position == 'long': 
        # this is a single stock case

        # for a portfolio valued as v0, the total number needed is 
        num = v0 / s

        results = []
        for i in range(trials): 
            projPrice = gbmPrice(s, mu, sigma, t)
            loss = s - projPrice
            results.append(loss)
        
        results = np.array(results)
        VaR = num * np.quantile(results, p)
        return VaR

    elif position == 'short': 

        num = v0 / s
        results = []
        for i in range(trials): 
            projPrice = gbmPrice(s, mu, sigma, t)
            loss = projPrice - s
            results.append(loss)

        results = np.array(results)
        VaR = num * np.quantile(results, p)
        return VaR

    else: 
        raise ValueError("Position should be 'long' or 'short'. ")



def MCES(s, v0, mu, sigma, p, t, a = None, covm = None, trials = 1000, position = 'long'): 
    """
    Use Monte-Carlo simulation to generate a set of paths following GBM. 
    Use the projected price from sample paths to calculate expected shortfall. 

    Args: 
        s: initial stock price 
        mu: drift
        sigma: volatility
        p: percentile (example: 99% ES -> p = 0.99)
        t: time steps (in years)
        trials: number of trials for simulations. 1000 by default. 
        position: string 'long' or 'short'. 'long' by default
    
    Returns: 
        A double value which represents ES based on simulations. 
    """
    if position == 'long': 
        # this is a single stock case

        # for a portfolio valued as v0, the total number needed is 
        num = v0 / s

        results = []
        for i in range(trials): 
            projPrice = gbmPrice(s, mu, sigma, t)
            loss = s - projPrice
            results.append(loss)

        results = np.array(results)
        results[::-1].sort()
        cumSumResults = np.cumsum(results)
        weight = np.arange(1, len(cumSumResults) + 1)
        ESlist = cumSumResults / weight
        return np.quantile(ESlist, p) * num

    elif position == 'short': 
        # this is a single stock case

        # for a portfolio valued as v0, the total number needed is 
        num = v0 / s

        results = []
        for i in range(trials): 
            projPrice = gbmPrice(s, mu, sigma, t)
            loss = projPrice - s
            results.append(loss)

        results = np.array(results)
        results[::-1].sort()
        cumSumResults = np.cumsum(results)
        weight = np.arange(1, len(cumSumResults) + 1)
        ESlist = cumSumResults / weight
        return np.quantile(ESlist, p) * num

    else: 
        raise ValueError("Position can only be 'long' or 'short'. ")


