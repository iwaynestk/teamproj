# Project
This model supports the calculation of VaR (value at risk) and ES (expected shortfall) of any given number of stocks (identified with their tickers) and the portfolio as a whole, based on 3 main methods: 
- **Parametric Method:** VaR and ES are directly calculated with closed form equations. 
- **Monte Carlo simulations:** VaR and ES are calculated based on simulated projected results. 
# Features
## Arbitrary Stock Ticker Input
When you first started to run the program, you will be asked to input the tickers of stocks you are interested in. For example, 'AAPL' for Apple. 
These stocks need to be currently publicly traded to make sure that data would be successfully fetched from Yahoo finance. 
## Graphical Comparison
Functions in [demo.py](./demo.py) would allow users to compare the impact of different estimation methods, or window periods, or positions on the final evaluation of VaR and ES. 
## Customization
All functions are publicly accessible to users. You might tailer the functions or combine them as you need. 

# Installation
To run this project, you need to install the following packages: 
- pandas: https://pandas.pydata.org
- numpy: https://numpy.org
- scipy: https://scipy.org
- yfinance: https://pypi.org/project/yfinance/

# Example
A detailed and interactive example could be found in [Model.py](./Model.ipynb). 

# Attention
Because Monte Carlo simulations could consume a large amount of computing power, unlike historical and parametrical VaR and ES (they are computed in advance and ready-to-use), estimation of VaR and ES based on simulations would operate in a call-and-calculate manner. 

# Contribution
This project and the corresponding model is accomplished by Yanjie Liu, Qingyi Yan and Ming Yin, students of Math GR5320, Fall 2021 at Columbia University. 

# General Structure


# Function Documentation
## [strGenerator.py](./strGenerator.py)


## [dataFactory.py](./dataFactory.py)

## [riskModelling.py](./riskModelling.py)

## [demo.py](./demo.py)