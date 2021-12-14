from os import EX_SOFTWARE
import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
import yfinance as yf
import dataFactory
import strGenerator

import unittest

class testStrGenerator(unittest.TestCase): 

    def testParamStrGenerator(self): 
        """
        Unit test for strGenerator.paramGenerator(ticker, parameter, window, method)

        Args: 
            None
        Returns: 
            Test results
        """
        ticker = 'AAPL'
        parameter = 'mu'
        window = 5
        method = 'exp'

        self.assertEqual(strGenerator.paramStrGenerator(ticker, parameter, window, method), 'AAPL_mu_5y_exp')
        self.assertEqual(strGenerator.paramStrGenerator(ticker, parameter), 'AAPL_mu')

        ticker1 = 'AAPL'
        parameter1 = 'mu'
        window1 = 2
        method1 = 'window'
        self.assertEqual(strGenerator.paramStrGenerator(ticker1, parameter1, window1, method1), 'AAPL_mu_2y_window')

    def testRiskStrGenerator(self): 
        """
        Unit test for strGenerator.riskStrGenerator(ticker, p, t, riskType, estWindow, estMethod)

        Args: 
            None
        Returns: 
            Test results
        """
        actualStr = strGenerator.riskStrGenerator(ticker = 'MSFT', p = 0.99, t = 10, riskType = 'ES', estWindow = 2, estMethod = 'exp', position = 'short')
        expectedStr = 'MSFT_0.99_10dES_2y_exp_short'
        self.assertEqual(actualStr, expectedStr)
        actualStr2 = strGenerator.riskStrGenerator(ticker = 'portfolio', p = 0.95, t = 5, riskType = 'VaR', estWindow = 2, estMethod = 'exp', position = 'long')
        expectedStr2 = 'portfolio_0.99_5dVaR_2y_exp_long'


if __name__ == '__main__': 
    unittest.main()
        