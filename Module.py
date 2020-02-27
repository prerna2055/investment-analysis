#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns and Computes and returns a DataFrame that contains:
    Wealth Index
    Previous peaks
    % Drawdowns
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

    
def semideviation(r):
    """
    Returns the semi deviation or negative semideviation of r must be a series or a DataFrame
    """
    
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    """
    An alternative to the scipy.stats.skew()
    Computes skewness of the supplied series of DF
    Returns a float or a Series
    """
    demeaned_r = r-r.mean()
    sigma_r=r.std(ddof=0) #use population Std. , so, set dof=0, i.e. no (n-1) correction
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    An alternative to the scipy.stats.kurtosis()
    Computes kurtosis of the supplied series of DF
    Returns a float or a Series
    """
    demeaned_r = r-r.mean()
    sigma_r=r.std(ddof=0) #use population Std. , so, set dof=0, i.e. no (n-1) correction
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


import scipy.stats
def is_normal(r, level = 0.01):
    """
    Applies Jarque-Bera test to determine if series is normal or not
    Test is applied at the 1% level by default
    Returns True if hypothesis of normailty is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value>level

def annualize_return(r,periods_per_year):
    """
    Annualizes the set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = len(r)
    return compounded_growth**(periods_per_year/n_periods)-1
    
def annualize_vol(r, periods_per_year):
    """
    Annualizes the volume of a set of returns"""
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
#convert the annual riskfree rate to per period

    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r-rf_per_period
    ann_ex_ret = annualize_return(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> returns
    """
    return weights.T @ returns

def portfolio_volatility(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**0.5

"""
    Plots the 2-Asset Efficient Frontier
    """
def plot_EF2(n_points, er, cov, style = ".-", title= 'Two Asset Efficient Frontier'):
    
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_EF can plot only two asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    Rets = [portfolio_return(w,er) for w in weights]
    Vols = [portfolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": Rets, "Volatility": Vols})
    
    return ef.plot.line(x = "Volatility", y = "Returns", style = style)



"""
    Plots the Multi-Asset Efficient Frontier
    """

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_volatility, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    """
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef_multi(n_points, er, cov, style = ".-"):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov) # not yet implemented!
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_volatility(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-', title= 'Multi-Asset Efficient Frontier')


import numpy as np
def var_hist(r, level=5):
    """
    Returns the historic VaR at a specified level, i.e. returns the number such that "level" percent of the returns fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_hist, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be series or DataFrame")
        

from scipy.stats import norm
def var_gaussian(r, level=5):
    """
    Returns the Parametric Gaussian(Variance Covariance approach) VaR of Seris or DataFrame
    """
    #assuming Z score to be Gaussian
    z= norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof = 0))
    
    
def var_modified(r, level = 5):
    """
    Returns Parametric Gausian VaR of a Series or DataFrame 
    If "modified" is TRUE, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    #compute the Z score assuming it was Gausian
    z= norm.ppf(level/100)
    #if modified:
        # modify the Z score based on observed skewness and kurtosis
    s=skewness(r)
    k=kurtosis(r)
    z=(z +
           (z**2-1)*s/6 + 
           (z**3 -3*z)*(k-3)/24 -
           (2*z**3 - 5*z)*(s**2)/36
      )
    return -(r.mean() + z*r.std(ddof = 0))

#CVaR
def cvar_hist(r, level=5):
    """
    Returns the historic VaR at a specified level, i.e. returns the number such that "level" percent of the returns fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_hist, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_hist(r, level=level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be series or DataFrame")
