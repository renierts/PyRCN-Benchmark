"""
Preprocessing utilities required to reproduce the results in the paper
'Template Repository for Research Papers with Python Code'.
"""
# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

import pandas as pd
from pandas import DataFrame
import numpy as np
from statsmodels.tsa.tsatools import lagmat


def ts2super(ts: DataFrame, n_lags: int, h: int):
    """
    Construction of supervised learning dataset (2D) from tim series (1D)
    ts: a dataframe with datetime index and a column with numeric values
    n_lags: number of lags to be used as inputs
    h: forecasting horizon
    """
    # Creating dataframe with lags
    lags = range(1, n_lags)
    data_super_lags = DataFrame(ts).assign(
        **{f'{col} (t-{t})': ts[col].shift(t) for t in lags for col in ts})
    # Reversing column orders in dataframe with lags
    data_super_lags = data_super_lags[data_super_lags.columns[::-1]]
    # Slicing dataframe with legs to get a single column with current values
    ts_at_zero = pd.DataFrame(data_super_lags.iloc[:, -1])
    # Creating dataframe with future values
    future = range(1, h + 1)
    data_super_future = ts_at_zero.assign(
        **{f'{col} (t+{t})': ts_at_zero[col].shift(-t)
           for t in future for col in ts_at_zero})
    # Joining lags and future values
    data_super = data_super_lags.join(data_super_future.iloc[:, 1:]).dropna()
    data_super = data_super.reset_index(drop=True)
    return data_super


def compute_average_volatility(X, window_length):
    """
    Compute the moving average volatility over a window of samples.

    Parameters
    ----------
    X : np.ndarray, shape=(n_samples, 1)
        one-dimensional input time series.
    window_length : int

    Returns
    -------
    X_new : np.ndarray, shape=(n_samples, 1)
        moving average across the input time series.
    """
    # Make sure to use the mode "full" but to return len(X) samples.
    X_new = np.convolve(X.ravel(), np.ones(window_length) / window_length,
                        mode="full").reshape(-1, 1)[:len(X)]

    """
    lag_array = lagmat(X, window_length)
    X_new = np.mean(lag_array[:, 0: window_length], 1).reshape(-1, 1)
    """
    return X_new
