
from unittest import result
from unittest.case import expectedFailure
import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
from wiener import wiener

import unittest
import riskModelling


class testGBM(unittest.TestCase): 

    def testParametricVaR(self): 
        """
        
        """
        # one-stock case
        a = np.array([1])
        s = np.array([100])
        mu = np.array([0.02])
        sigma = np.array([0.3])
        covm = np.array([sigma**2])
        p = 0.99
        t = 5 / 252
        position = 'long'
        a_s = a*s
        v0 = a_s.sum()
        evt = a_s @ np.transpose(np.exp(mu * t)) 
        evt2 = np.exp(mu * t) @ (np.exp(covm * t) * (np.transpose(a_s) @ a_s)) @ np.transpose(np.exp(mu * t))
        varvt = evt2 - evt ** 2
        sdvt = np.sqrt(varvt)
        expected = v0 - (evt + norm.ppf(1 - p) * sdvt)
        self.assertEqual(a * s, 100)
        actual = riskModelling.parametricVaR(a, s, mu, sigma, covm, p, t, position)
        self.assertEqual(expected, actual)

        # multiple-stock case
        a = np.array([100, 200])
        s = np.array([150, 180])
        mu = np.array([0.02, 0.03])
        sigma = np.array(([0.3, 0.2]))
        rho = np.array([[1, 0.25], [0.25, 1]])
        p = 0.99
        t = 5 / 252
        position = 'long'
        # covariance matrix: 
        covm = sigma * rho * np.transpose(sigma)
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
        varvt = evt2 - evt**2
        # sd(V_t)
        sdvt = np.sqrt(varvt)
        expected = v0 - (evt + norm.ppf(1 - p) * sdvt)
        actual = riskModelling.parametricVaR(a, s, mu, sigma, covm, p, t, position = 'long')
        self.assertEqual(actual, expected)
        # unit test passed



    def testParametricES(self): 
        # one-stock case
        a = np.array([1])
        s = np.array([100])
        mu = np.array([0.02])
        sigma = np.array([0.3])
        covm = np.array([sigma**2])
        p = 0.99
        t = 5 / 252
        position = 'long'
        a_s = a*s
        v0 = a_s.sum()
        evt = a_s @ np.transpose(np.exp(mu * t)) 
        evt2 = np.exp(mu * t) @ (np.exp(covm * t) * (np.transpose(a_s) @ a_s)) @ np.transpose(np.exp(mu * t))
        varvt = evt2 - evt ** 2
        sdvt = np.sqrt(varvt)
        expected = v0 - evt +  sdvt/(1 - p)*(norm.pdf(norm.ppf(p)))
        self.assertEqual(a * s, 100)
        actual = riskModelling.parametricES(a, s, mu, sigma, covm, p, t, position)
        self.assertEqual(expected, actual)

        # multiple-stock case
        a = np.array([100, 200])
        s = np.array([150, 180])
        mu = np.array([0.02, 0.03])
        sigma = np.array(([0.3, 0.2]))
        rho = np.array([[1, 0.25], [0.25, 1]])
        p = 0.99
        t = 5 / 252
        position = 'short'
        # covariance matrix: 
        covm = sigma * rho * np.transpose(sigma)
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
        varvt = evt2 - evt**2
        # sd(V_t)
        sdvt = np.sqrt(varvt)
        expected = -v0 + evt + sdvt/(1 - p)*(norm.pdf(norm.ppf(p)))
        actual = riskModelling.parametricES(a, s, mu, sigma, covm, p, t, position = position)
        self.assertEqual(actual, expected)
        # unit test passed



    def testGbmVaR(self): 
        v0 = 100000
        mu = 0.015
        sigma = 0.2
        p = 0.99
        t = 5 / 252
        position = 'short'

        expected = -v0 + v0 * exp(sigma * sqrt(t) * norm.ppf(p) \
            + (mu - sigma ** 2 / 2) * t)
        actual = riskModelling.gbmVaR(v0, mu, sigma, p, t, 'short')
        self.assertEqual(actual, expected)

        position1 = 'long'
        expected1 = v0 - v0 * exp(sigma * sqrt(t) * norm.ppf(1 - p) + (mu - 0.5 * sigma**2) * t)
        actual1 = riskModelling.gbmVaR(v0, mu, sigma, p, t, position1)
        self.assertEqual(actual1, expected1)



    def testGbmES(self): 
        v0 = 100000
        mu = 0.015
        sigma = 0.2
        p = 0.99
        t = 5 / 252
        position = 'long'

        expected =  v0 * (1 - exp(mu*t) / (1 - p) \
            * norm.cdf(norm.ppf(1 - p) - sqrt(t)*sigma))
        actual = riskModelling.gbmES(v0, mu, sigma, p, t, position)
        self.assertEqual(actual, expected)
        # unit test passed


if __name__ == '__main__': 
    unittest.main()