#region imports
from AlgorithmImports import *
#endregion
from sklearn import linear_model
import numpy as np
import pandas as pd
from scipy import stats
from math import floor
from datetime import timedelta
#For testing coint. 
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


class PairsTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        
        self.SetStartDate(2009,1,1)
        self.SetEndDate(2022,10,10)
        self.SetCash(10000)
        self.numdays = 250  # set the length of training period
        tickers = ["XOM", "CVX"]
       
        self.symbols = []
        
        self.threshold = 1.5
        for i in tickers:
            self.symbols.append(self.AddSecurity(SecurityType.Equity, i, Resolution.Daily).Symbol)
        for i in self.symbols:
            i.hist_window = RollingWindow[TradeBar](self.numdays) 


    def OnData(self, data):
        if not (data.ContainsKey("CVX") and data.ContainsKey("XOM")): return

        for symbol in self.symbols:
            symbol.hist_window.Add(data[symbol])
        price_x = pd.Series([float(i.Close) for i in self.symbols[0].hist_window], 
                             index = [i.Time for i in self.symbols[0].hist_window])    
        price_y = pd.Series([float(i.Close) for i in self.symbols[1].hist_window], 
                             index = [i.Time for i in self.symbols[1].hist_window])
        if len(price_x) < 251: return
        for symbol in self.symbols:
            symbol.hist_window.remove(0)
        spread = self.regr(np.log(price_x), np.log(price_y))
        mean = np.mean(spread)
        std = np.std(spread)
        ratio = floor(self.Portfolio[self.symbols[1]].Price / self.Portfolio[self.symbols[0]].Price)
        if spread[-1] > mean + self.threshold * std:
            if not self.Portfolio[self.symbols[0]].Quantity > 0 and not self.Portfolio[self.symbols[0]].Quantity < 0:
                self.Sell(self.symbols[1], 100) 
                self.Buy(self.symbols[0],  ratio * 100)
        elif spread[-1] < mean - self.threshold * std:
            if not self.Portfolio[self.symbols[0]].Quantity < 0 and not self.Portfolio[self.symbols[0]].Quantity > 0:
                self.Sell(self.symbols[0], 100)
                self.Buy(self.symbols[1], ratio * 100) 
        else:
            self.Liquidate()

    





    def regr(self,x,y):
        regr = linear_model.LinearRegression()
        x_constant = np.column_stack([np.ones(len(x)), x])
        regr.fit(x_constant, y)
        beta = regr.coef_[0]
        alpha = regr.intercept_
        spread = y - x*beta - alpha
        return spread
    
    def find_cointegrated_pairs(data):
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2, maxlag=1)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs.append((keys[i], keys[j]))
        return score_matrix, pvalue_matrix, pairs
        