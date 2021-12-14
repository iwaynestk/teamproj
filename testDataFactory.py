import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
import yfinance as yf
import dataFactory
import gbm

import unittest
from pandas._testing import assert_frame_equal, assert_series_equal

class testDataFactory(unittest.TestCase): 

    def testDwnloadAndJoin(self): 
        """
        Unit test for function dataFactory.dwnloadAndJoin
        Args: 
            None
        Returns: 
            Test results
        """
        # numberOfStocks = 2
        # ticker1 = 'AAPL'
        # ticker2 = 'MSFT'
        # df1_expected = yf.download(ticker1, period = 'max')
        # df1_expected = pd.DataFrame(data = df1_expected['Adj Close'])
        # df1_expected = df1_expected.rename({'Adj Close': ticker1}, axis = 1)
        
        # # df2_expected = yf.download(ticker2, period = 'max')
        # # df2_expected = pd.DataFrame(data = df2_expected['Adj Close'])
        # # df2_expected = df2_expected.rename({'Adj Close': ticker1}, axis = 1)

        # df1_actual = dataFactory.dwnloadAndJoin([ticker1])
        # # df2_actual = dataFactory.dwnloadAndJoin(ticker2)

        # assert_frame_equal(df1_actual, df1_expected, check_less_precise = True)
        # # assert_frame_equal(df2_actual, df2_expected)
    
        pass


    def testLogRtnCalc(self): 
        """
        Unit test for function dataFactory.logRtnCalc(df, ticker)
        Args: 
            None
        Returns: 
            Test results
        """
        ticker = 'AAPL'
        df_actual = pd.DataFrame(np.random.randint(0, 100, size = (100, 2)), columns = ['date', 'AAPL'])

        df_expected = df_actual.copy(deep = True)
        dataFactory.logRtnCalc(df_actual, ticker)

        df_expected['AAPL_rtn'] = df_expected['AAPL'].pct_change(1)
        df_expected['AAPL_logRtn'] = np.log(df_expected['AAPL']) - np.log(df_expected['AAPL'].shift(1))

        assert_frame_equal(df_expected, df_actual, check_less_precise = True)


    def testGbmParamCalc(self): 
        """
        Unit test for dataFactory.gbmParamCalc(df, ticker, window, method, lam)
        Args: 
            None
        Returns: 
            Test results
        """
        ticker = 'AAPL'

        """ unit test for window method """
        #generate random dataframe of shape (1000, 2)
        df_actual = pd.DataFrame(np.random.randint(0, 100, size = (1000, 2)), columns = ['date', 'AAPL'])
        df_expected = df_actual.copy(deep = True)

        dataFactory.logRtnCalc(df_actual, ticker)
        dataFactory.gbmParamCalc(df_actual, ticker, window = 1, method = 'window')

        dataFactory.logRtnCalc(df_expected, 'AAPL')

        # manually calculate expected results
        df_expected['AAPL_mu_hat_1y_window'] = df_expected['AAPL_logRtn'].rolling(window = 252).mean()
        df_expected['AAPL_sigma_hat_1y_window'] = df_expected['AAPL_logRtn'].rolling(window = 252).std()
        df_expected['AAPL_sigma_1y_window'] = df_expected['AAPL_sigma_hat_1y_window'] * sqrt(252)
        df_expected['AAPL_mu_1y_window'] = df_expected['AAPL_mu_hat_1y_window'] * 252 + 0.5 * df_expected['AAPL_sigma_1y_window'] ** 2

        assert_frame_equal(df_actual, df_expected)

        """ unit test for exp weighted method """
        df1_actual = pd.DataFrame(np.random.randint(0, 100, size = (1000, 2)), columns = ['date', 'AAPL'])
        df1_expected = df1_actual.copy(deep = True)

        lam = 0.997
        dataFactory.logRtnCalc(df1_actual, ticker)
        dataFactory.gbmParamCalc(df1_actual, ticker, window = 1, method = 'exp', lam = lam)

        dataFactory.logRtnCalc(df1_expected, 'AAPL')

        df1_expected['AAPL_mu_hat_1y_exp'] = df1_expected['AAPL_logRtn'].ewm(alpha = 1 - 0.997, adjust = False).mean() * 252
        df1_expected['AAPL_sigma_1y_exp'] = df1_expected['AAPL_logRtn'].ewm(alpha = 1 - 0.997, adjust = False).std() * sqrt(252)
        df1_expected['AAPL_mu_1y_exp'] = df1_expected['AAPL_mu_hat_1y_exp'] + 0.5 * df1_expected['AAPL_sigma_1y_exp'] ** 2

        assert_frame_equal(df1_actual, df1_expected)


        # def testCovm(self): 
        #     df = pd.DataFrame(np.random.randint(0, 100, size = (1000, 5)), columns = ['date', 'data1_logRtn', 'data2_logRtn', 'data3_logRtn', 'data4_logRtn'])
        #     tickerList = ['data1', 'data2', 'data3', 'data4']
        #     covmdf = df[['data1_logRtn', 'data2_logRtn', 'data3_logRtn', 'data4_logRtn']].rolling(252).cov(pairwise=True)


        def testPARAMriskCalc(self): 
            df = pd.DataFrame(np.random.randint(100, 120, size = (1000, 5)), \
                columns = ['AAPL', 'MSFT', 'TSLA', 'JNJ', 'PFE'])
            df = df.join(pd.DataFrame(np.arange(1000), columns = ['date']))

            tickerList = ['AAPL', 'MSFT', 'TSLA', 'JNJ', 'PFE']
            for i in range(len(tickerList)): 
                dataFactory.logRtnCalc(df, tickerList[i])

            for i in range(len(tickerList)): 
                dataFactory.gbmParamCalc(df, tickerList[i], window = 2, method = 'window')

            df['expected'] = df.apply(lambda x: gbm.gbmVaR(10000, x['AAPL_mu_2y_window'], x['AAPL_sigma_2y_window'], 0.98, 10 / 252), axis = 1)
            dataFactory.riskCalc(df, 10000, 'AAPL', 0.98, 10/252, 2, 'VaR', 'window', 'long')
            assert_series_equal(df['expected'], df['AAPL_0.98_10dVaR_2y_window_long'])



        def testMCriskCalc(self): 
            pass


if __name__ == '__main__': 
    unittest.main()