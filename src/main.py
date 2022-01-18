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
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load


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
    datasets = ["CAT", "EBAY", "MSFT"]
    data = [None] * 3

    LOGGER.info("Loading the training dataset...")
    for k, dataset in enumerate(datasets):
        data[k] = load_data(f"./data/{dataset}.csv")["t0"]
    LOGGER.info("... done!")

    # Index for model selection
    idx_matrix = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])

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
            for experiment in range(len(data)):
                LOGGER.info(f"    Fold {experiment}: ")

                # Experiment indexes
                ts_train = data[idx_matrix[experiment][0]]
                ts_val = data[idx_matrix[experiment][1]]
                ts_test = data[idx_matrix[experiment][2]]
                LOGGER.info("        MinMaxScaling:")
                # Data scaling to [0, 1]
                train_delta = np.max(ts_train) - np.min(ts_train)
                ts_train_scld = (ts_train - np.min(ts_train)) / train_delta
                ts_val_scld = (ts_val - np.min(ts_train)) / train_delta
                ts_test_scld = (ts_test - np.min(ts_train)) / train_delta
                LOGGER.info("        ... done!")
                # Finding ARIMA orders through grid search
                LOGGER.info("        GridSearch:")
                order = arima.grid_search(
                    ts_train_scld, ts_val_scld, H, p_grid, d_grid, q_grid)
                LOGGER.info("        ... done!")
                LOGGER.info("        Prediction:")
                try:
                    pred_train, pred_test = arima.fit_predict(
                        ts_train_scld, ts_test_scld, order, H)
                    LOGGER.info("        ... done!")
                except:
                    pred_train = 'convergence failed'
                    pred_test = 'convergence failed'
                    LOGGER.info("        ... failed!")
                # Unscale prediction
                if type(pred_train) != str:
                    pred_train_unscld = pred_train * train_delta \
                                        + np.min(ts_train)
                    pred_test_unscld = pred_test * train_delta \
                                       + np.min(ts_train)

                # Measuring accuracy
                if type(pred_train) != str:
                    mse_test = mean_squared_error(
                        ts_test[H:], pred_test_unscld)
                    r2_test = r2_score(
                        ts_test[H:], pred_test_unscld)
                    mse_train = mean_squared_error(
                        ts_train[:len(pred_train_unscld)], pred_train_unscld)
                    r2_train = r2_score(
                        ts_train[:len(pred_train_unscld)], pred_train_unscld)
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
                dump(results, f'./results/arimagrid_{datasets[experiment]}'
                              f'_h{H}.joblib')
                LOGGER.info("    ... done!")
            LOGGER.info("... done!")
        if fit_mlp:
            results = {'R2 train': [], 'R2 test': [],
                       'MSE train': [], 'MSE test': []}

            # Run model selection
            LOGGER.info(f"Starting the experiment with the horizon {H}...")
            for experiment in range(len(data)):
                LOGGER.info(f"    Fold {experiment}: ")

                # Experiment indexes
                trn = data[idx_matrix[experiment][0]]
                val = data[idx_matrix[experiment][1]]
                tst = data[idx_matrix[experiment][2]]
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
            LOGGER.info("... done!")
        if fit_esn:
            results = {'R2 train': [], 'R2 test': [],
                       'MSE train': [], 'MSE test': []}

            # Run model selection
            LOGGER.info(f"Starting the experiment with the horizon {H}...")
            for experiment in range(len(data)):
                LOGGER.info(f"    Fold {experiment}: ")

                # Experiment indexes
                trn = data[idx_matrix[experiment][0]]
                val = data[idx_matrix[experiment][1]]
                tst = data[idx_matrix[experiment][2]]
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
