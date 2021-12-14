import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
import wiener
import dataFactory
import matplotlib.pyplot as plt
from strGenerator import riskStrGenerator


def riskPlot(df, ax_obj, riskLabel): 
    """
    A function that plot a subsplot

    Args: 
        df: original pandas.DataFrame object
        ax_obj: matplotlib.pyplot.axes object, which corresponds to one subplot of the main plot. 
        riskLabel: the label (which is also column name) used to look up data
    Returns: 
        matplotlib.pyplot.subplots
    """

    ax_obj.plot(df[riskLabel], label = riskLabel)

def fMethodDiffWinPlot(df, bigType, ticker, p, t, estWindowList, estMethod, position = 'long'): 
    """
    A function that plots and compares different window length under the same estimation method. 
    fMethodDiffWinPlot stands for 'fixed method different window. '

    Args: 
        df: original pandas,DataFrame
        bigType: 'MC' or 'PARAM' or 'HIST'
        ticker: stock ticker
        p: percentile (example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindowList: a list of window length to be compared (example: [2, 5, 10])
        estMethod: 'window' or 'exp'
    
    Returns: 
        A plot with VaR comparison at the top and ES comparison at the bottom. 
    """
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 32))

    for i in range(len(estWindowList)): 
        VaRLabel = riskStrGenerator(bigType, ticker, p, t, 'VaR', estWindowList[i], estMethod, position)
        ESLabel = riskStrGenerator(bigType, ticker, p, t, 'ES', estWindowList[i], estMethod, position)
        riskPlot(df, ax[0], VaRLabel)
        ax[0].legend(loc = 'upper right')
        riskPlot(df, ax[1], ESLabel)
        ax[1].legend(loc = 'upper right')

    plt.show()    
    # for the same method, we would have a list of different window length
    return

def fWinDiffMethodPlot(df, bigType, ticker, p, t, estWindow, position = 'long'): 
    """
    Fixed window different methods. 
    A function that compares different method ('window' and 'exp') based on the same historical window
    (example: 2-y rolling window and 2-y equivalent lambda)

    For the same window, we only have two methods, rolling window and exponential weighting

    Args: 
        df: original pandas.DataFrame
        bigType: The main method (Monte Carlo, historical and parametric) used to calculate VaR and ES. Must be 'MC' or 'PARAM' or 'HIST'
        ticker: stock ticker
        p: percentile (Must be within (0, 1). Example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindow: length of historical window of data to look back
        position: 'long' or 'short'
    
    Returns: 
        A plot with VaR comparison at the top and ES comparison at the bottom. 
    """
    
    
    VaRLabel_window = riskStrGenerator(bigType, ticker, p, t, 'VaR', estWindow, 'window', position)
    VaRLabel_exp = riskStrGenerator(bigType, ticker, p, t, 'VaR', estWindow, 'exp', position)
    ESLabel_window = riskStrGenerator(bigType, ticker, p, t, 'ES', estWindow, 'window', position)
    ESLabel_exp = riskStrGenerator(bigType, ticker, p, t, 'ES', estWindow, 'exp', position)

    labels = [[VaRLabel_window, VaRLabel_exp], [ESLabel_window, ESLabel_exp]]

    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 32))

    for i in range(len(ax)): 
        for j in range(2): 
            riskPlot(df, ax[i], riskLabel=labels[i][j])
        ax[i].legend(loc = 'upper right')

    plt.show()
    return
    # unit test passed

def fWinFMethodDiffPosition(df, bigType, ticker, p, t, estWindow, estMethod): 
    """
    Fixed window, fixed method, different positions. 
    A function that compares VaR and ES with different positions based on same historical window and estimation method. 

    For the same method and historical window, we only have 'long' and 'short' position. 

    Args: 
        df: original pandas.DataFrame
        bigType: The main method (Monte Carlo, historical and parametric) used to calculate VaR and ES. Must be 'MC' or 'PARAM' or 'HIST'
        ticker: stock ticker
        p: percentile (Must be within (0, 1). Example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindow: length of historical window of data to look back
        estMethod: 'window' or 'exp'

    Returns: 
        A plot with VaR comparison at the top and ES comparison at the bottom. 
    """

    # VaR labels for long and short positions
    VaRLabel_long = riskStrGenerator(bigType, ticker, p, t, 'VaR', estWindow, estMethod, 'long')
    VaRLabel_short = riskStrGenerator(bigType, ticker, p, t, 'VaR', estWindow, estMethod, 'short')

    # ES labels for long and short positions
    ESLabel_long = riskStrGenerator(bigType, ticker, p, t, 'ES', estWindow, estMethod, 'long')
    ESLabel_short = riskStrGenerator(bigType, ticker, p, t, 'ES', estWindow, estMethod, 'short')

    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 32))

    riskPlot(df, ax[0], VaRLabel_long)
    riskPlot(df, ax[0], VaRLabel_short)
    ax[0].legend(loc = 'upper right')

    riskPlot(df, ax[1], ESLabel_long)
    riskPlot(df, ax[1], ESLabel_short)
    ax[1].legend(loc = 'upper right')

    return
    # unit test passed

def paramMC(df, ticker, p, t, estWindow, estMethod, position): 
    """
    A function that generates plots to compare VaR and ES from parametric method and Monte Carlo simulations. 
    'Fixed window, fixed estimation method, fixed position'
    
    Args: 
        df: original pandas.DataFrame
        ticker: stock ticker
        p: percentile (Must be within (0, 1). Example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindow: length of historical window of data to look back
        estMethod: 'window' or 'exp'
        position: 'long' or 'short' 
    Returns: 
        A plot with VaR comparison at the top and ES comparison at the bottom. 
    """

    VaRLabel_MC = riskStrGenerator('MC', ticker, p, t, 'VaR', estWindow, estMethod, position)
    VaRLabel_PARAM = riskStrGenerator('PARAM', ticker, p, t, 'VaR', estWindow, estMethod, position)

    ESLabel_MC = riskStrGenerator('MC', ticker, p, t, 'ES', estWindow, estMethod, position)
    ESLabel_PARAM = riskStrGenerator('PARAM', ticker, p, t, 'ES', estWindow, estMethod, position)

    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 32))

    riskPlot(df, ax[0], VaRLabel_MC)
    riskPlot(df, ax[0], VaRLabel_PARAM)
    ax[0].legend(loc = 'upper right')

    riskPlot(df, ax[1], ESLabel_MC)
    riskPlot(df, ax[1], ESLabel_PARAM)
    ax[1].legend(loc = 'upper right')

    return

def PARAM_MC_HIST(df, ticker, p, t, estWindow, position = 'long'): 
    """
    A function that generates plots to compare VaR and ES from parametric method, Monte Carlo simulations and historical method. 
    'Fixed window, fixed estimation method, long position'

    Args: 
        df: original pandas.DataFrame
        ticker: stock ticker
        p: percentile (Must be within (0, 1). Example: 99% ES -> p = 0.99)
        t: period of time to look ahead (in years)
        estWindow: length of historical window of data to look back
        position: 'long' or 'short' 

    Returns: 
        A plot with VaR comparison at the top and ES comparison at the bottom. 
    """
    
    VaRLabel_PARAM = riskStrGenerator('PARAM', ticker, p, t, 'VaR', estWindow, 'window', position)
    VaRLabel_MC = riskStrGenerator('MC', ticker, p, t, 'VaR', estWindow, 'window', position)
    VaRLabel_HIST = riskStrGenerator('HIST', ticker, p, t, 'VaR', estWindow, 'window', position)

    ESLabel_PARAM = riskStrGenerator('PARAM', ticker, p, t, 'ES', estWindow, 'window', position)
    ESLabel_MC = riskStrGenerator('MC', ticker, p, t, 'ES', estWindow, 'window', position)
    ESLabel_HIST = riskStrGenerator('HIST', ticker, p, t, 'ES', estWindow, 'window', position)

    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 32))

    riskPlot(df, ax[0], VaRLabel_MC)
    riskPlot(df, ax[0], VaRLabel_PARAM)
    riskPlot(df, ax[0], VaRLabel_HIST)
    ax[0].legend(loc = 'upper right')

    riskPlot(df, ax[1], ESLabel_MC)
    riskPlot(df, ax[1], ESLabel_PARAM)
    riskPlot(df, ax[1], ESLabel_HIST)
    ax[1].legend(loc = 'upper right')

    return
