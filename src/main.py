"""
Main Code to reproduce the results in the paper
'Template Repository for Research Papers with Python Code'.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

import logging
from src.file_handling import load_data
import numpy as np
import src.arima as arima
from src.preprocessing import ts2super
from src.file_handling import export_results
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from collections import OrderedDict, deque


LOGGER = logging.getLogger(__name__)


def main(fit_arima: bool = False, fit_mlp: bool = False,
         fit_esn: bool = False) -> None:
    """
    This is the main function to reproduce all visualizations and models for
    the paper "Template Repository for Research Papers with Python Code".

    It is controlled via command line arguments:

    Parameters
    ----------
    fit_arima : bool, default=False
        Fit ARIMA models.
    fit_mlp: default=False
        Fit MLP models.
    fit_esn:
        Fit ESN models.
    """
    datasets = OrderedDict({"CAT": 0, "EBAY": 1, "MSFT": 2})
    data = [None] * len(datasets)

    LOGGER.info("Loading the training dataset...")
    for dataset, k in datasets.items():
        data[k] = load_data(f"./data/{dataset}.csv")["t0"].to_frame()
    LOGGER.info("... done!")

    # Index for model selection
    idx_matrix = deque(datasets.values())

    # Hyperparameters tuning
    p_grid = list(range(3))
    d_grid = list(range(2))
    q_grid = list(range(3))
    # Forecasting horizon
    for H in [1, 5, 21]:
        if fit_arima:
            results = {'R2 train': [], 'R2 test': [],
                       'MSE train': [], 'MSE test': []}

            # Run model selection
            LOGGER.info(f"Starting the experiment with the horizon {H}...")
            for experiment in range(len(datasets)):
                LOGGER.info(f"    Fold {experiment}: ")

                # Experiment indexes
                ts_train = data[idx_matrix[0]]
                ts_val = data[idx_matrix[1]]
                ts_test = data[idx_matrix[2]]
                LOGGER.info("        MinMaxScaling:")
                # Data scaling to [0, 1]
                try:
                    x_scaler = load(f'./results/x_scaler_arima_grid_'
                                    f'{experiment}_h{H}.joblib')
                except FileNotFoundError:
                    x_scaler = MinMaxScaler().fit(ts_train)
                    dump(x_scaler, f'./results/x_scaler_arima_grid_'
                                   f'{experiment}_h{H}.joblib')
                ts_train_scaled = x_scaler.transform(ts_train)
                ts_val_scaled = x_scaler.transform(ts_val)
                ts_test_scaled = x_scaler.transform(ts_test)
                LOGGER.info("        ... done!")
                # Finding ARIMA orders through grid search
                LOGGER.info("        GridSearch:")
                order = arima.grid_search(
                    ts_train_scaled, ts_val_scaled, H, p_grid, d_grid, q_grid)
                LOGGER.info("        ... done!")
                LOGGER.info("        Prediction:")
                try:
                    pred_train, pred_test = arima.fit_predict(
                        ts_train_scaled, ts_test_scaled, order, H)
                    LOGGER.info("        ... done!")
                except:
                    pred_train = 'convergence failed'
                    pred_test = 'convergence failed'
                    LOGGER.info("        ... failed!")
                # Unscale prediction
                if type(pred_train) != str:
                    pred_train_unscaled = x_scaler.inverse_transform(
                        pred_train)
                    pred_test_unscaled = x_scaler.inverse_transform(pred_test)

                # Measuring accuracy
                if type(pred_train) != str:
                    mse_test = mean_squared_error(
                        ts_test[H:], pred_test_unscaled)
                    r2_test = r2_score(
                        ts_test[H:], pred_test_unscaled)
                    mse_train = mean_squared_error(
                        ts_train[:len(pred_train_unscaled)],
                        pred_train_unscaled)
                    r2_train = r2_score(
                        ts_train[:len(pred_train_unscaled)],
                        pred_train_unscaled)
                else:
                    mse_test = 'convergence failed'
                    r2_test = 'convergence failed'
                    mse_train = 'convergence failed'
                    r2_train = 'convergence failed'

                # store results
                results['MSE train'].append(mse_train)
                results['MSE test'].append(mse_test)
                results['R2 train'].append(r2_train)
                results['R2 test'].append(r2_test)

                # Save list of dictionaries with results
                export_results(
                    results, f'./results/arima_grid_{experiment}_h{H}.csv')
                LOGGER.info("    ... done!")
                idx_matrix.rotate(1)
            LOGGER.info("... done!")
        if fit_mlp:
            results = {'R2 train': [], 'R2 test': [],
                       'MSE train': [], 'MSE test': []}

            # Run model selection
            LOGGER.info(f"Starting the experiment with the horizon {H}...")
            for experiment in range(len(datasets)):
                LOGGER.info(f"    Fold {experiment}: ")

                # Experiment indexes
                trn = data[idx_matrix[0]]
                val = data[idx_matrix[1]]
                tst = data[idx_matrix[2]]
                LOGGER.info("        Feature Extraction:")
                trn_feat = dp.get_ts_features(trn, 0, H, fs='pacf',dummies=False)
                feat = trn_feat[3]
                feat_cols = [-x for x in feat]
                val_super = ts2super(val, trn_feat[4], H)
                tst_super = ts2super(tst, trn_feat[4], H)
                val_selected = val_super[trn_feat[2]]
                tst_selected = tst_super[trn_feat[2]]
                # Experiment indexes
                X_train = trn_feat[0]
                Y_train = trn_feat[1]
                X_val = val_selected
                Y_val = val_super.iloc[:, -1].values.reshape(-1, 1)
                X_test = tst_selected
                Y_test = tst_super.iloc[:, -1].values.reshape(-1, 1)
                LOGGER.info("        MinMaxScaling:")
                # Data scaling to [0, 1]
                xdelta = np.max(X_train, axis=0) - np.min(X_train, axis=0)
                ydelta = np.max(Y_train, axis=0) - np.min(Y_train, axis=0)
                X_train_scld = (X_train - np.min(X_train, axis=0)) / xdelta
                X_val_scld = (X_val - np.min(X_train, axis=0)) / xdelta
                X_test_scld = (X_test - np.min(X_train, axis=0)) / xdelta
                Y_train_scld = (Y_train - np.min(Y_train, axis=0)) / ydelta
                Y_val_scld = (Y_val - np.min(Y_train, axis=0)) / ydelta
                LOGGER.info("        ... done!")
                # Finding ARIMA orders through grid search
                LOGGER.info("        GridSearch:")
                mdl = mlp_gs.objct(grid_points=5, seed=1985)
                mdl.fit(X_train_scld, Y_train_scld, X_val_scld, Y_val_scld)
                LOGGER.info("        ... done!")
                LOGGER.info("        Prediction:")
                Y_train_pred_scld = mdl.predict(X_train_scld)
                Y_test_pred_scld = mdl.predict(X_test_scld)
                # Unscale prediction
                Y_train_pred = Y_train_pred_scld * ydelta + np.min(Y_train,
                                                                   axis=0)
                Y_test_pred = Y_test_pred_scld * ydelta + np.min(Y_train,
                                                                 axis=0)
                # Measuring accuracy
                results['MSE train'].append(
                    mean_squared_error(Y_train, Y_train_pred))
                results['MSE test'].append(
                    mean_squared_error(Y_test, Y_test_pred))
                results['R2 train'].append(r2_score(Y_train, Y_train_pred))
                results['R2 test'].append(r2_score(Y_test, Y_test_pred))

                # Save list of dictionaries with results
                dump(results, f'./results/mlpgrid_{datasets[experiment]}'
                              f'_h{H}.joblib')
                LOGGER.info("    ... done!")
            idx_matrix.rotate(1)
            LOGGER.info("... done!")
        if fit_esn:
            results = {'R2 train': [], 'R2 test': [],
                       'MSE train': [], 'MSE test': []}

            # Run model selection
            LOGGER.info(f"Starting the experiment with the horizon {H}...")
            for experiment in range(len(datasets)):
                LOGGER.info(f"    Fold {experiment}: ")

                # Experiment indexes
                trn = ts2super(data[idx_matrix[0]], 0, H)
                val = ts2super(data[idx_matrix[1]], 0, H)
                tst = ts2super(data[idx_matrix[2]], 0, H)
                LOGGER.info("        Feature Extraction:")
                X_train = trn.iloc[:, 0].values.reshape(-1, 1)
                y_train = trn.iloc[:, -1].values.reshape(-1, 1)
                X_val = val.iloc[:, 0].values.reshape(-1, 1)
                y_val = val.iloc[:, -1].values.reshape(-1, 1)
                X_test = tst.iloc[:, 0].values.reshape(-1, 1)
                y_test = tst.iloc[:, -1].values.reshape(-1, 1)
                LOGGER.info("        MinMaxScaling:")
                # Data scaling to [0, 1]
                x_scaler = MinMaxScaler().fit(X_train)
                y_scaler = MinMaxScaler().fit(y_train)
                X_train_scaled = x_scaler.transform(X_train)
                X_val_scaled = x_scaler.transform(X_val)
                X_test_scaled = x_scaler.transform(X_test)
                y_train_scaled = y_scaler.transform(y_train)
                y_val_scaled = y_scaler.transform(y_val)
                LOGGER.info("        ... done!")
                # Finding ARIMA orders through grid search
                LOGGER.info("        GridSearch:")
                mdl = mlp_gs.objct(grid_points=5, seed=1985)
                mdl.fit(X_train_scld, Y_train_scld, X_val_scld, Y_val_scld)
                LOGGER.info("        ... done!")
                LOGGER.info("        Prediction:")
                Y_train_pred_scld = mdl.predict(X_train_scld)
                Y_test_pred_scld = mdl.predict(X_test_scld)
                # Unscale prediction
                Y_train_pred = Y_train_pred_scld * ydelta + np.min(Y_train,
                                                                   axis=0)
                Y_test_pred = Y_test_pred_scld * ydelta + np.min(Y_train,
                                                                 axis=0)
                # Measuring accuracy
                results['MSE train'].append(
                    mean_squared_error(Y_train, Y_train_pred))
                results['MSE test'].append(
                    mean_squared_error(Y_test, Y_test_pred))
                results['R2 train'].append(r2_score(Y_train, Y_train_pred))
                results['R2 test'].append(r2_score(Y_test, Y_test_pred))

                # Save list of dictionaries with results
                dump(results, f'./results/mlpgrid_{datasets[experiment]}'
                              f'_h{H}.joblib')
                idx_matrix.rotate(1)
                LOGGER.info("... done!")
            LOGGER.info("... done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_arima", action="store_true")
    parser.add_argument("--fit_mlp", action="store_true")
    parser.add_argument("--fit_esn", action="store_true")
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    LOGGER.setLevel(logging.DEBUG)
    main(**args)
    exit(0)
