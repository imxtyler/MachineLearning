#/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def ols_eval(X,y):
    est = sm.OLS(y, X).fit()
    print(est.summary())
    print(est.params)
    return est

def plot(est,X,y):
    y_fitted = est.fittedvalues
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(X, y, 'o', label='data')
    ax.plot(X, y_fitted, 'r--.', label='OLS')
    ax.legend(loc='best')
    plt.show()

def analysis_corr(dataset,X,y,constant=False):
    #print(dataset.shape)
    #print(dataset.ndim)
    #dataset = dataset[:, np.newaxis]
    cor = np.corrcoef(dataset, rowvar=0)[:,dataset.shape[1]-1]
    print("the corrcoef is:",cor,"\n")
    if(constant):
        X = sm.add_constant(X)
    est = ols_eval(X, y)
    plot(est,X,y)