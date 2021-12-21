from numpy.lib.function_base import quantile
import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
from scipy.stats.stats import combine_pvalues
import yfinance as yf
from strGenerator import riskStrGenerator, paramStrGenerator
import riskModelling


def readTickerList():
    """
    A function that reads stock tickers from users. 

    Args: None

    Returns: 
        A list of string which corresponds to the ticker of each stock. 
    """
    numberOfStocks = input("Please enter the total number of stocks you would like to observe: ")
    numberOfStocks = int(numberOfStocks)
    tickerList = []
    for i in range(numberOfStocks): 
        tickerList.append(input("Please enter stock ticker (AAPL for Apple): "))
    return tickerList



def readNumList(tickerList): 
    """
    A function that reads number of stocks to buy from users. 

    Args: 
        tickerList: list of string which corresponds to the ticker of each stock. 

    Returns: 
        A list of integers which corresponds to the number of shares of each stock. 
    """
    numberOfStocks = len(tickerList)
    numberOfStocks = int(numberOfStocks)
    numList = []
    for i in range(numberOfStocks): 
        displayStr = "Please enter the number of stocks you would like to buy for " + tickerList[i] + ". Must be an integer. "
        numList.append(int(input(displayStr)))

    return numList



def positionCalc(df, tickerList, numList): 
    """
    A function that add columns with number of shares of each stock to original dataframe. 

    Args: 
        df: original DataFrame object with price data in columns labeled with ticker
        tickerList: list of string that corresponds to the ticker of each stock
        numList: number of shares that corresponds to the ticker of each stock

    Returns:
        None
    """
    for i in range(len(tickerList)): 
        numLabel = paramStrGenerator(tickerList[i], 'num')
        
        # we assume that the number of each stock follows the same proportion throughout. 
        df[numLabel] = numList[i]
    return



def portfolioCalc(df, tickerList, numList): 
    """
    A function that add portfolio value as a new column to original dataframe. 

    Args: 
        df: original DataFrame object with price data in columns labeled with ticker
        tickerList: list of string that corresponds to the ticker of each stock
        numList: number of shares that corresponds to the ticker of each stock

    Returns: 
        None
    """
    df['portfolio'] = 0
    for i in range(len(tickerList)): 
        ticker = tickerList[i]
        df['portfolio'] += df[ticker] * numList[i]
    return



def dwnloadAndJoin(tickerList): 
    """
    A function that downloads data using yfinance (yahoo finance) 
    and join data of different stocks into one single dataframe. 

    Args: 
        tickerList: list of string that corresponds to tickers of stocks to be observed. 
    Returns: 
        A pandas.DataFrame consisting of all data of all stocks. 
    """
    df = pd.DataFrame()

    # join dateframe one by one
    for i in range(len(tickerList)): 
        print(tickerList[i])
        dfTicker = yf.download(tickerList[i], period = 'max')
        dfTicker = pd.DataFrame(data = dfTicker['Adj Close'])

        # if there's no data in df, start from the first dfTicker we have
        if len(df) == 0: 
            df = dfTicker
            df = df.rename({'Adj Close': tickerList[i]}, axis = 1)
            continue

        df = df.join(dfTicker, how = 'outer')
        df = df.rename({'Adj Close': tickerList[i]}, axis = 1)
    return df



def logRtnCalc(df, ticker, lag): 
    """
    A function that calculates log return given the ticker of the stock and time interval. 

    Args:
        df: pandas.DataFrame
        ticker: stock ticker
        lag: 'lag'-day log return

    Returns: 
        None. 
    """    
    paramName = str(lag) + 'dlogRtn'
    logRtnLabel = paramStrGenerator(ticker, paramName)
    df[logRtnLabel] = np.log(df[ticker]) - np.log(df[ticker].shift(lag))
    return



def gbmParamCalc(df, ticker, window, method, lam = None): 
    
    """
    A function that calculates the estimated mu and sigma for GBM. 
    This function supports both window and exponential weighting methods. 

    Args: 
        df: original DataFrame object
        ticker: stock ticker
        window: window period in years (2, 5 or 10)
        method: estimation method ('window' or 'exp')
        lam: Equivalent lambda, only used for 'exp' type estimation

    Returns: 
        None. Changes are made to original dataframe. 
    """

    # generate labels
    logRtnLabel = paramStrGenerator(ticker, '1dlogRtn') # e.g. jnj_logRtn
    muHatLabel = paramStrGenerator(ticker, 'mu_hat', window, method) # e.g. jnj_mu_hat_5y_exp
    sigmaHatLabel = paramStrGenerator(ticker, 'sigma_hat', window, method) # e.g. jnj_sigma_hat_2y_window
    muLabel = paramStrGenerator(ticker, 'mu', window, method) # e.g. jnj_sigma_10y_exp
    sigmaLabel = paramStrGenerator(ticker, 'sigma', window, method) # e.g. jnj_mu_2y_window


    if method == 'window': 
        # mu_hat and sigma_hat
        df[muHatLabel] = df[logRtnLabel].rolling(window = window * 252).mean()
        df[sigmaHatLabel] = df[logRtnLabel].rolling(window = window * 252).std()
        df[sigmaLabel] = df[sigmaHatLabel] * sqrt(252)
        df[muLabel] = df[muHatLabel] * 252 + 0.5 * df[sigmaLabel] ** 2

        return

    elif method == 'exp': 

        df[muHatLabel] = df[logRtnLabel].ewm(alpha = 1 - lam, adjust = False).mean() * 252
        df[sigmaLabel] = df[logRtnLabel].ewm(alpha = 1 - lam, adjust = False).std() * sqrt(252)
        df[muLabel] = df[muHatLabel] + 0.5 * df[sigmaLabel] ** 2

        return

    else: 
        raise ValueError("Only 'window' or 'exp' methods are supported. ")



def PARAMriskCalc(df, v0, ticker, p, t, estWindow, riskType = 'VaR', estMethod = 'window', position = 'long'): 
    """
    Generate new columns in original DataFrame which calculates parametric VaR/ES. 
    
    Args: 
        df: original df
        ticker: stock ticker
        p: percentile (example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindow: length of historical window of data to look back
        riskType: 'VaR' or 'ES'
        estMethod: 'window' or 'exp'
        position: the position you would like to take: 'long' or 'short'
    Returns: 
        None. Changes are made to original dataframe. 
    """
    riskLabel = riskStrGenerator('PARAM', ticker, p, int(t * 252), riskType=riskType, estWindow=estWindow, estMethod=estMethod, position = position)
    muLabel = paramStrGenerator(ticker, 'mu', window = estWindow, method = estMethod)
    sigmaLabel = paramStrGenerator(ticker, 'sigma', window=estWindow, method=estMethod)


    if riskType == 'VaR': 
        df[riskLabel] = df.apply(lambda x: riskModelling.gbmVaR(v0, x[muLabel], x[sigmaLabel], p, t, position = position), axis = 1)
        return
    elif riskType == 'ES': 
        df[riskLabel] = df.apply(lambda x: riskModelling.gbmES(v0, x[muLabel], x[sigmaLabel], p, t, position = position), axis = 1)
        return
    else: 
        raise ValueError("Only 'VaR' and 'ES' are allowed. ")



def MCriskCalc(df, v0, ticker, p, t, estWindow, riskType = 'VaR', estMethod = 'window', position = 'long', trials = 1000): 
    """
    Generate new columns in original DataFrame which calculates VaR/ES based on Monte Carlo simulations
    
    Args: 
        df: original df
        ticker: stock ticker
        p: percentile (example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindow: length of historical window of data to look back
        riskType: 'VaR' or 'ES'
        estMethod: 'window' or 'exp'
        position: the position you would like to take: 'long' or 'short'
    Returns: 
        None. Changes are made to original dataframe.
    """

    riskLabel = riskStrGenerator('MC', ticker, p, int(t * 252), riskType=riskType, estWindow=estWindow, estMethod=estMethod, position = position)
    muLabel = paramStrGenerator(ticker, 'mu', window = estWindow, method = estMethod)
    sigmaLabel = paramStrGenerator(ticker, 'sigma', window=estWindow, method=estMethod)
    numLabel = paramStrGenerator(ticker, 'num')


    if riskType == 'VaR': 
        df[riskLabel] = df.apply(lambda x: riskModelling.MCVaR(x[numLabel], v0, x[muLabel], x[sigmaLabel], p, t, position = position, trials = trials), axis = 1)
        return
    elif riskType == 'ES': 
        df[riskLabel] = df.apply(lambda x: riskModelling.MCES(x[numLabel], v0, x[muLabel], x[sigmaLabel], p, t, position = position, trials = trials), axis = 1)
        return
    else: 
        raise ValueError("Only 'VaR' and 'ES' are allowed. ")



def HISTriskCalc(df, v0, ticker, p, t, estWindow): 
    """
    Generate new columns in original DataFrame which calculates VaR/ES based on Monte Carlo simulations
    
    Args: 
        df: original df
        ticker: stock ticker
        p: percentile (example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindow: length of historical window of data to look back
        riskType: 'VaR' or 'ES'
    Returns: 
        None. Changes are made to original dataframe.    
    """
    VaRLabel = riskStrGenerator('HIST', ticker, p, int(t * 252), 'VaR', 5)
    ESLabel = riskStrGenerator('HIST', ticker, p, int(t * 252), 'ES', 5)

    lag = int(252 * t)

    logRtnTail = str(lag) + 'dlogRtn'
    lagLogRtnLabel = paramStrGenerator(ticker, logRtnTail)

    df[VaRLabel] = v0 - v0 * np.exp(df[lagLogRtnLabel].rolling(estWindow * 252).quantile(1 - p, interpolation = 'lower'))
    df[ESLabel] = v0 - v0 * np.exp(df[lagLogRtnLabel].rolling(estWindow * 252).apply(lambda x: conditionalMean(x, p)))
    return



def conditionalMean(df, p): 
    """
    Calculate the mean of data that rank within (1-p)th quantile of the entire series. 

    Args: 
        df: One-column dataframe
        p: quantile (must be within (0, 1))
    Returns: 
        The conditional mean of data lying within (1-p)th quantile of df. 
    """
    threshold = df.quantile(1 - p)
    mean = df[df <= threshold].mean()
    return mean



def targetLiquidate(df, ticker, v0, p, t, iv):
    """
    Calculate the proportion of v0 (originally 100% in stock/portfolio) to be liquidated to buy
    put option in order to decrease VaR by 20%. 

    Args: 
        df: the original DataFrame containing stock price and VaR
        ticker: the ticker of the stock you would like to observe
        v0: initial fund
        p: quantile (must be within (0, 1)) for VaR (99% VaR -> p = 0.99)
        t: time horizon of VaR (5d VaR -> t = 5)
        iv: implied volatility

    Returns: 
        None
    """

    # get the label of VaR column in df
    VaRLabel = riskStrGenerator('PARAM', ticker, p, t, 'VaR', 5, 'window', 'long')

    # generate the label of target VaR column in df and calculate their values
    targetVaRLabel = 'target_' + VaRLabel
    df[targetVaRLabel] = 0.8 * df[VaRLabel]

    # generate the label of proportion to be liquidated
    liquidateLabel = 'liquidate_' + VaRLabel

    # label factory here... 
    stockLabel = ticker
    muLabel = paramStrGenerator(ticker, 'mu', 5, 'window')
    sigmaLabel = paramStrGenerator(ticker, 'sigma', 5, 'window')

    df[liquidateLabel] = df.apply(lambda x: riskModelling.liquidateProportion\
        (v0, x[stockLabel], x[muLabel], x[sigmaLabel], iv, t / 252, p, 0.005, \
            x[targetVaRLabel], 0.01, 0.1, 0.01), axis = 1)
    return


def rhoCalc(df, tickerList): 
    """
    This function calculate the correlation coefficient of two stocks based on rolling window of length 5 years. 

    Args: 
        df: the original DataFrame with daily log return of individual stocks
        tickerList: a list of tickers of stocks. Must be of length 3 with ticker 'portfolio' included

    Returns: 
        None
    """

    if len(tickerList) != 3: 
        raise ValueError("This model only supports rho of 2 stocks as a value. ")

    logRtnLabel1 = paramStrGenerator(tickerList[0], '1dlogRtn')
    logRtnLabel2 = paramStrGenerator(tickerList[1], '1dlogRtn')

    combinedStr = tickerList[0] + '_' +  tickerList[1]

    covmLabel = paramStrGenerator(combinedStr, 'covm')
    rhoLabel = paramStrGenerator(combinedStr, 'rho')

    df[covmLabel] = df[logRtnLabel1].rolling(5 * 252).cov(df[logRtnLabel2])

    sigmaLabel1 = paramStrGenerator(tickerList[0], 'sigma', window=5, method='window')
    sigmaLabel2 = paramStrGenerator(tickerList[1], 'sigma', window=5, method='window')

    df[rhoLabel] = 252 * df[covmLabel] / (df[sigmaLabel1]*df[sigmaLabel2])

    return

  


def MCriskCalc_corr(df, tickerList, p, t, estWindow, riskType = 'VaR', estMethod = 'window', trials = 1000, v0 = 100000): 
    """
    This function calculates VaR or ES of the portfolio assuming underlying correlated stocks follow GBM, based on Monte Carlo simulations. 

    Args: 
        df: the original DataFrame containing stock price and VaR
        tickerList: a list of tickers of stocks. (ticker 'portfolio' must be inside this list)
        p: quantile (must be within (0, 1)) for VaR (99% VaR -> p = 0.99)
        t: time horizon of VaR (5d VaR -> t = 5 / 252)
        estWindow: length of window which corresponds to the parameters (mu, sigma)
        riskType: 'VaR' or 'ES'
        estMethod: 'window' or 'exp'
        trials = number of trials to conduct Monte Carlo simulations. 1000 by default
        v0: initial fund. 100000 by default 
    
    Returns: 
        None
    """

    if len(tickerList) != 3: 
        raise ValueError("MC calculation for correlated stocks only supports portfolios with exactly 2 stocks. ")

    position = 'long'

    # getting labels
    combinedStr = tickerList[0] + '_' +  tickerList[1]

    rhoLabel = paramStrGenerator(combinedStr, 'rho')

    riskLabel = riskStrGenerator('MC', 'corr_portfolio', p, int(t * 252), riskType=riskType, estWindow=estWindow, estMethod=estMethod, position = position)
    
    muLabel1 = paramStrGenerator(tickerList[0], 'mu', window = estWindow, method = estMethod)
    muLabel2 = paramStrGenerator(tickerList[1], 'mu', window = estWindow, method = estMethod)
    
    sigmaLabel1 = paramStrGenerator(tickerList[0], 'sigma', window=estWindow, method=estMethod)
    sigmaLabel2 = paramStrGenerator(tickerList[1], 'sigma', window=estWindow, method=estMethod)

    numLabel1 = paramStrGenerator(tickerList[0], 'num')
    numLabel2 = paramStrGenerator(tickerList[1], 'num')

    if riskType == 'VaR': 
        df[riskLabel] = df.apply(lambda x: riskModelling.MCVaR_corr(x[sigmaLabel1], x[sigmaLabel2], x[muLabel1], x[muLabel2], x[numLabel1], x[numLabel2], \
            x[tickerList[0]], x[tickerList[1]], x[rhoLabel], t, p, numberOfTrials=trials, v0 = v0), axis = 1)
        return
    elif riskType == 'ES': 
        df[riskLabel] = df.apply(lambda x: riskModelling.MCVaR_corr(x[sigmaLabel1], x[sigmaLabel2], x[muLabel1], x[muLabel2], x[numLabel1], x[numLabel2], \
            x[tickerList[0]], x[tickerList[1]], x[rhoLabel], t, p, numberOfTrials=trials, v0 = v0), axis = 1)
        return
    else: 
        raise ValueError("Only 'VaR' and 'ES' are allowed. ")











    


