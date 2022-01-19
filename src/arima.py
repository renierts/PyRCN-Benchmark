"""Functions for ARIMA model."""
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
import warnings
from itertools import product
from joblib import Parallel, delayed


warnings.filterwarnings("ignore")


# evaluate an ARIMA model for a given order (p,d,q)
def fit_predict(ts_train, ts_test, arima_order, H):
    """
    ARIMA predictions H steps ahead given:
    Args:
    ts_train: array with training time series
    ts_test: array with test time series
    arima_order: (p, d, q) tuple
    H: forecasting horizon
    Returns:
    pred: predictions H steps ahead
    """
    train_model = ARIMA(ts_train, order=arima_order)
    train_model_fit = train_model.fit()
    train_pred = train_model_fit.predict().reshape(-1, 1)

    history = [x for x in ts_train]
    predictions = list()
    for t in range(len(ts_test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=H)[0]
        predictions.append(yhat)
        history.append(ts_test[t])

    test_pred = np.array(predictions[:-H]).reshape(-1, 1)

    return train_pred, test_pred


def optimization_step(p, d, q, ts_train, ts_test, H):
    order = (p, d, q)
    print(order)
    try:
        _, y_hat = fit_predict(ts_train, ts_test, order, H)
        mse = mean_squared_error(ts_test[H:], y_hat)
        print('ARIMA%s MSE=%.3f' % (order, mse))
        return order, mse
    except:
        print('ARIMA%s MSE=%.3f' % (order, float("inf")))
        return order, float("inf")


def grid_search(ts_train, ts_test, H, p_grid, d_grid, q_grid):
    """
    Grid search to find best ARIMA orders.
    """
    best_score, best_cfg = float("inf"), None
    result = Parallel(n_jobs=-1)(
        delayed(optimization_step)(p, d, q, ts_train, ts_test, H)
        for p, d, q in product(p_grid, d_grid, q_grid))

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

    return best_cfg


def arima_tscv(n_splits, ts, H, p_grid, d_grid, q_grid):
    """
    Time-series cross-validation
    Args:
    n_splits: int, number of cv splits
    X: array of float, inputs matrix (nsamples x nfeat)
    Y: array of float, output array (nsamples x 1)
    H: forecasting horizon
    """

    # data split for cross-validation (time slice for time series)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Loop over data splits
    results_list = list()
    for train_index, test_index in tscv.split(ts):

        # Slicing data
        dev = ts[train_index]
        test = ts[test_index]

        # Data scaling to [0, 1]
        dev_max = np.max(dev, axis=0)
        dev_min = np.min(dev, axis=0)
        dev_scld = (dev - dev_min) / (dev_max - dev_min)
        test_scld = (test - dev_min) / (dev_max - dev_min)

        # Finding ARIMA orders through grid search
        train = dev_scld[:round(0.8*len(dev_scld))]
        valid = dev_scld[round(0.8*len(dev_scld)):]
        order = grid_search(train, valid, H, p_grid, d_grid, q_grid)

        # Prediction over test set
        try:
            start_time = time.time()
            pred_train, pred_test = fit_predict(dev_scld, test_scld, order, H)
            prediction_time = time.time() - start_time
        except:
            pred_train = 'convergence failed'
            pred_test = 'convergence failed'
            prediction_time = 'convergence failed'

        # Unscale prediction
        if type(pred_train) != str:
            pred_train_unscld = pred_train*(dev_max - dev_min) + dev_min
            pred_test_unscld = pred_test*(dev_max - dev_min) + dev_min

        # Measuring accuracy
        if type(pred_train) != str:
            mse_test = mean_squared_error(test[H:], pred_test_unscld)
            r2_test = r2_score(test[H:], pred_test_unscld)
            mse_train = mean_squared_error(dev[:len(pred_train_unscld)],
                                               pred_train_unscld)
            r2_train = r2_score(dev[:len(pred_train_unscld)],
                                    pred_train_unscld)
        else:
            mse_test = 'convergence failed'
            r2_test = 'convergence failed'
            mse_train = 'convergence failed'
            r2_train = 'convergence failed'

        # store results
        results_list.append({
                            'Prediction time': prediction_time,
                            'MSE train': mse_train,
                            'MSE test': mse_test,
                            'R2 train': r2_train,
                            'R2 test': r2_test,
                            'order(p,d,q)': order,
                            })
    return results_list
