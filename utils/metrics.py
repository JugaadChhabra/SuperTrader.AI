import numpy as np
import pandas as pd


def rolling_vol_ewma(returns,span=3):
    returns=pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    
    lam=(span-1)/(span+1)
    ewma_var=(returns**2).ewm(alpha=1-lam,adjust=False).mean()
    return np.sqrt(ewma_var.values[-1])


def sharpe(returns,span=3,ann_factor=252,r_f=0):
    returns=pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    
    alpha=2/(span+1)
    mu=returns.ewm(alpha=alpha,adjust=False).mean()
    sigma2=((returns-mu.shift(1))**2).ewm(alpha=alpha,adjust=False).mean()
    
    return ((mu-r_f)/np.sqrt(sigma2)).values[-1]*np.sqrt(ann_factor)


def sortino(returns,ann_factor=252,rf=0.0):
    returns=pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    
    excess_returns=returns-rf
    mean_excess=excess_returns.mean()
    negative_excess=excess_returns[excess_returns<0]
    downside_std=negative_excess.std(ddof=0)

    if downside_std==0:
        return np.nan
    
    return (mean_excess/downside_std)*np.sqrt(ann_factor)


def max_drawdown(equity_curve):
    equity=pd.Series(equity_curve).dropna()
    if equity.empty:
        return np.nan
    
    cumulative_max=equity.cummax()
    drawdowns=(equity-cumulative_max)/cumulative_max

    return drawdowns.min()


def calmar(returns,equity_curve,annual_factor=252):
    returns=pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    
    annual_return=returns.mean()*annual_factor
    mdd=abs(max_drawdown(equity_curve))

    if mdd==0:
        return np.nan
    
    return annual_return/mdd


def turnover(positions):
    positions=pd.Series(positions).fillna(0)
    if positions.empty:
        return np.nan
    
    total_changes=np.sum(np.abs(np.diff(positions)))
    avg_position=np.mean(np.abs(positions))

    if avg_position==0:
        return np.nan
    
    return total_changes/avg_position
