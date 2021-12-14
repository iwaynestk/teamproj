def paramStrGenerator(ticker, parameter, window = '', method = ''):
    """
    A function that generates parameter labels, which will be used as DataFrame column names. 

    Args: 
        ticker: stock ticker
        parameter: parameter name (mu, sigma, logRtn, num, etc,.)
        window: length of historical window period (2, 5, 10. In years)
        method: estimation method ('window' or 'exp')

    Returns: 
        A string with all the information aggregated from input. 

    Examples: 
        'AAPL_num', 'MSFT_mu_2y_exp', 'TSLA_sigma_10y_window'
    """

    # if window and method are not emtpy
    if window and method: 
        returnedString = ticker + '_' + parameter + '_' + str(window) + 'y_' + method
        return returnedString
    else: 
        returnedString = ticker + '_' + parameter
        return returnedString

def riskStrGenerator(bigType, ticker, p, t, riskType, estWindow, estMethod = None, position = None): 
    """
    A function that generate label for VaR and ES. 
    
    Args: 
        bigType: 'MC' or 'PARAM' or 'HIST'
        ticker: stock ticker
        p: percentile to compute VaR or ES (99%ES: p = 0.99)
        t: period of time to look ahead (e.g. 5d VaR), in days
        riskType: 'VaR' or 'ES'
        estWindow: the length of historical window that estimation was based on (in years)
        estMethod: 'window' or 'exp'
        position: 'long' or 'short'
    
    Returns: 
        A string that represent the label of the specified risk parameter. 
    
    Examples: 
        AAPL_0.99_5dVaR_5y_window_short
        MSFT_0.975_10dES_10y_exp_long (exp method does not have estWindow)
    """
    if bigType == 'HIST': 
        returnedString = bigType + '_' + ticker + '_' + str(p) + '_' + str(t) + 'd' + riskType \
            + '_' + str(estWindow) + 'yData'
        return returnedString

    if estMethod == 'window' or 'exp': 
        returnedString = bigType + '_' + ticker + '_' + str(p) + '_' + str(t) + 'd' + riskType \
            + '_' + str(estWindow) + 'y_' + estMethod + '_' + position
        # example: AAPL_0.99_5dVaR_5y_window_long
        # example: MSFT_0.975_3dES_10y_exp_short
        return returnedString
    else: 
        raise ValueError("Only 'window' or 'exp' methods are allowed for single stocks. ")