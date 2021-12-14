import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm


class wiener: 
    def __init__(self, length = 1, covm = None): 
        self.covm = covm
        if covm is None: 
            if length != 1: 
                raise SyntaxError("Please input correlation matrix")
            else: 
                self.length = 1
                self.process = np.array(np.random.normal(0, 1))
        else: 
            if length != len(covm): 
                raise SyntaxError("Length of corrMatrix need to match length of process vector. ")
            else: 
                self.length = length
                mean = np.zeros(len(covm))
                self.process = np.random.multivariate_normal(mean = mean, cov = covm)

