from multiprocessing import Value
from numpy.core.arrayprint import str_format
from numpy.core.fromnumeric import std
import pandas as pd
import numpy as np
from math import sqrt, exp, log, floor
from scipy.stats import norm
import matplotlib.pyplot as plt
from assetPricing import blackScholes
from strGenerator import riskStrGenerator
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

def MCVaR_corr(sigma1, sigma2, mu1, mu2, n1, n2, v1, v2, rho, dt, p, numberOfTrials = 1000, v0 = 100000): 
    """
    This function calculates the VaR of a portfolio consisting of 2 correlated stocks. 

    Args: 
        sigma1: volatility of stock 1
        sigma2: volatility of stock 2
        mu1: GBM mu parameter of stock 1
        mu2: GBM mu parameter of stock 2
        n1: number of shares of stock 1
        n2: number of shares of stock 2
        v1: current price of stock 1
        v2: current price of stock 2
        rho: correlation
        dt: time step (in years. 5 / 252 for 5 days)
        p: quantile (must be within (0, 1)) for VaR (99% VaR -> p = 0.99)
        numberOfTrials: total number of trials to conduct Monte Carlo simulations
        v0: initial fund
        
    Returns: 
        A double value representing VaR of the portfolio based on Monte Carlo simulations. 
    """

    num1 = v0*(n1*v1)/(n1*v1+n2*v2)/v1
    num2 = v0*(n2*v2)/(n1*v1+n2*v2)/v2

    trialResults = []

    for i in range(numberOfTrials + 1): 
        w1 = np.random.normal(0, 1)
        w2 = rho * w1 + sqrt(1 - rho**2) * w1

        vt = v1 * num1 * exp((mu1 - 0.5*sigma1**2) * dt + sigma1 * w1 * sqrt(dt)) \
            + v2 * num2 * exp((mu2 - 0.5*sigma2**2) * dt + sigma2 * w2 * sqrt(dt))

        trialResults.append(vt)

    trialResultNP = np.array(trialResults)
    VaR = v0 - np.percentile(trialResultNP, 1 - p)
    return VaR


def MCES_corr(sigma1, sigma2, mu1, mu2, n1, n2, v1, v2, rho, dt, p, numberOfTrials = 1000, v0 = 100000): 
    """
    This function calculates the ES of a portfolio consisting of 2 correlated stocks. 

    Args: 
        sigma1: volatility of stock 1
        sigma2: volatility of stock 2
        mu1: GBM mu parameter of stock 1
        mu2: GBM mu parameter of stock 2
        n1: number of shares of stock 1
        n2: number of shares of stock 2
        v1: current price of stock 1
        v2: current price of stock 2
        rho: correlation
        dt: time step (in years. 5 / 252 for 5 days)
        p: quantile (must be within (0, 1)) for VaR (99% VaR -> p = 0.99)
        numberOfTrials: total number of trials to conduct Monte Carlo simulations
        v0: initial fund
        
    Returns: 
        A double value representing ES of the portfolio based on Monte Carlo simulations. 
    """

    num1 = v0*(n1*v1)/(n1*v1+n2*v2)/v1
    num2 = v0*(n2*v2)/(n1*v1+n2*v2)/v2

    trialResults = []

    # numberOfTrials trials on each date
    for i in range(numberOfTrials + 1): 
        w1 = np.random.normal(0, 1)
        w2 = rho * w1 + sqrt(1 - rho ** 2) * w1
        value = v1 * num1 * np.exp((mu1 - 0.5*sigma1**2) * dt + sigma1 * w1 * sqrt(dt)) \
            + v2 * num2 * np.exp((mu2 - 0.5*sigma2**2) * dt + sigma2 * w2 * sqrt(dt))
        trialResults.append(value)
        
    trialResultNP = np.sort(v0 - np.array(trialResults))
    trialResultNP = trialResultNP[::-1]
    cumSumTrialResultNP = np.cumsum(trialResultNP)
    weightList = np.arange(1, len(cumSumTrialResultNP) + 1)
    ESlist = cumSumTrialResultNP / weightList
    return np.percentile(ESlist, p)



def liquidateVaR_MC(v0, s0, mu, sigma, iv, dt, p, liquidate, r, trial = 100): 
    
    """
    Returns VaR of portfolio given the weight of portfolio liquidated to buy put option. 
    
    Args: 
        v0: portfolio initial fund
        s0: stock price
        mu: drift parameter
        sigma: volatility parameter for GBM
        iv: implied volatility for option
        dt: time interval (in years)
        p: quantile for VaR e.g. 0.99 for 99% VaR
        liquidate: percentage of portfolio to be liquidated to buy put option
        r: interest rate
        trial: number of trials to run Monte Carlo simulation. 1000 by default
        
    Returns: 
        A double value of VaR
    """
    
    opt0 = blackScholes(s = s0, x = s0, sigma = iv, r = r, t = 1, optionType = 'p')
    
    numOpt = v0 * liquidate / opt0
    numStock = v0 * (1 - liquidate) / s0
    
    
    trialResults = []
    for i in range(0, trial + 1): 
        s_t = gbmPrice(s0, mu, sigma, dt)
        opt_t = blackScholes(s = s_t, x = s0, sigma = iv, r = r, t = 1 - dt, optionType = 'p')
        v_t = s_t * numStock + opt_t * numOpt
        loss = v0 - v_t
        trialResults.append(loss)
        
    trialResultsArr = np.array(trialResults)
    VaR = np.quantile(trialResultsArr, p)
    return VaR




def liquidateProportion(v0, s0, mu, sigma, iv, dt, p, r, targetVaR, left, right, e, trial = 100): 
    
    """
    Returns the percentage of a portfolio to be liquidated to achieve target VaR
        
    Args: 
        v0: initial fund of portfolio
        s0: initial stock price
        mu: drift parameter for GBM
        sigma: volaitlity parameter for GBM
        iv: implied volatility
        dt: time interval (in years)
        p: quantile for VaR (0.99 for 99% VaR)
        r: risk free rate
        targetVaR: target VaR
        left: left starting point
        right: right starting point
        e: tolerence (0.001 if you want your result to be as acurate as 0.1%)
        trial: number of trials for each position. 1000 by default

    
    Returns: 
        A double value in range [0, 1] which indicates the percentage to liquidate to buy put option
    """
    
    liquidate = []
    results = []
    for i in range(floor((right - left) / e) + 1): 
        liquidate.append(left + e * i)
        VaR = liquidateVaR_MC(v0, s0, mu, sigma, iv, dt, p, left + i * e, r, trial = trial)
        
        # the difference between current VaR and target VaR, in absolute value
        difference = abs(VaR - targetVaR)
        results.append(difference)
        
    minDifference = min(results)
    minIndex = results.index(minDifference)
    
    return left + minIndex * e