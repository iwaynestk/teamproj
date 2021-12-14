import pandas as pd
import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm
import matplotlib.pyplot as plt

import unittest
from wiener import wiener

class testWiener(unittest.TestCase): 

    def testOneWiener(self): 
        # one wiener process
        processList = []
        for i in range(100000): 
            process = wiener()
            processList.append(process.process)

        processArray = np.array(processList)
        self.assertAlmostEqual(processArray.mean(), 0, places = 2)
        self.assertAlmostEqual(processArray.var(), 1.0, places = 2)

    def testMultiWiener(self): 
        pass


if __name__ == '__main__': 
    unittest.main()