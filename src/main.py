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
from src.preprocessing import (
    ts2super, split_datasets, compute_average_volatility)
from src.file_handling import export_results
from src.model_selection import PredefinedTrainValidationTestSplit
from src.adapter import PyESN, ReservoirPyESN
import pandas as pd
import itertools
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump, load
from collections import OrderedDict, deque
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

from pyrcn.extreme_learning_machine import ELMRegressor
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.model_selection import SequentialSearchCV
from skelm import ELMRegressor as SkELMRegressor


LOGGER = logging.getLogger(__name__)


def main(fit_arima: bool = False, fit_mlp: bool = False, fit_har: bool = False,
         fit_pyrcn_esn: bool = False, fit_har_pyrcn_esn: bool = False,
         fit_pyrcn_elm: bool = False, fit_har_pyrcn_elm: bool = False,
         fit_skelm_elm: bool = False, fit_har_skelm_elm: bool = False,
         fit_pyesn_esn: bool = False, fit_har_pyesn_esn: bool = False,
         fit_reservoirpy_esn: bool = False,
         fit_har_reservoirpy_esn: bool = False) -> None:
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
    fit_har: default=False
        Fit the HAR model.
    fit_pyrcn_esn:
        Fit ESN models with PyRCN.
    fit_har_pyrcn_esn:
        Fit ESN models with PyRCN.
    fit_pyrcn_elm:
        Fit ESN models with PyRCN.
    fit_har_pyrcn_elm:
        Fit ESN models with PyRCN.
    fit_skelm_elm:
        Fit ESN models with scikit-ELM.
    fit_har_skelm_elm:
        Fit ESN models with scikit-ELM.
    fit_pyesn_esn:
        Fit ESN models with PyESN.
    fit_har_pyesn_esn:
        Fit ESN models with PyESN.
    fit_reservoirpy_esn:
        Fit ESN models with ReservoirPy.
    fit_har_reservoirpy_esn:
        Fit ESN models with ReservoirPy.
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
                ts_train, ts_val, ts_test = split_datasets(data, idx_matrix)
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
                LOGGER.info("    ... done!")
                idx_matrix.rotate(1)
            export_results(results, f'./results/arima_grid_{experiment}_h{H}.csv')
            LOGGER.info("... done!")
        if fit_mlp:
            results = {'R2 train': [], 'R2 test': [],
                       'MSE train': [], 'MSE test': []}

            # Run model selection
            LOGGER.info(f"Starting the experiment with the horizon {H}...")
            for experiment in range(len(datasets)):
                LOGGER.info(f"    Fold {experiment}: ")

                # Experiment indexes
                trn, val, tst = split_datasets(data, idx_matrix)
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
                mdl = GridSearchCV(
                    estimator=MLPRegressor(
                        activation="tanh", solver="lbfgs", max_iter=1000,
                        max_fun=1000),
                    param_grid={"hidden_layer_sizes": [(10, ), (10, 10),
                                                       (20, ), (20, 20),
                                                       (30, ), (30, 30),
                                                       (300, ), (300, 300)],
                                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0]})
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
        if fit_har:
            LOGGER.info(f"Starting the HAR experiment with the horizon {H}...")
            LOGGER.info(f"    Creating feature extraction and HAR pipeline...")
            day_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 1})
            week_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 5})
            month_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 22})
            har_features = FeatureUnion(
                transformer_list=[("day", day_volatility_transformer),
                                  ("week", week_volatility_transformer),
                                  ("month", month_volatility_transformer)])
            har_pipeline = Pipeline(
                steps=[("har_features", har_features),
                       ("scaler", MinMaxScaler()),
                       ("lstsq", TransformedTargetRegressor(
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            search = GridSearchCV(
                estimator=har_pipeline, param_grid={}, cv=ps,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search, f'./results/har_grid_h{H}.joblib')
            LOGGER.info("    ... done!")
        if fit_pyrcn_esn:
            LOGGER.info(f"Starting the ESN experiment with the horizon {H}...")
            LOGGER.info(f"    Creating feature extraction and ESN pipeline...")
            initial_esn_params = {
                'hidden_layer_size': 50, 'k_in': 1, 'input_scaling': 0.4,
                'input_activation': 'identity', 'bias_scaling': 0.0,
                'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
                'reservoir_activation': 'tanh', 'bidirectional': False,
                'alpha': 1e-5, 'random_state': 42}
            esn_pipeline = Pipeline(
                steps=[("scaler", MinMaxScaler()),
                       ("esn", TransformedTargetRegressor(
                           regressor=ESNRegressor(**initial_esn_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'esn__regressor__input_scaling': uniform(loc=1e-2, scale=1),
                'esn__regressor__spectral_radius': uniform(loc=0, scale=2)}
            step2_params = {'esn__regressor__leakage': uniform(1e-5, 1e0)}
            step3_params = {
                'esn__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step4_params = {
                'esn__regressor__hidden_layer_size':
                    [50, 100, 200, 400, 800, 1600, 3200, 6400],
                'esn__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step4 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                ('step3', RandomizedSearchCV, step3_params, kwargs_step3),
                ('step4', RandomizedSearchCV, step4_params, kwargs_step4)]
            search = SequentialSearchCV(
                esn_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyrcn_seq_esn_h{H}.joblib')

            ps_test = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_test = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={}, cv=ps_test,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_test, f'./results/pyrcn_seq_esn_h{H}_test.joblib')
            LOGGER.info("    ... done!")
        if fit_har_pyrcn_esn:
            LOGGER.info(f"Starting the HAR-ESN experiment with the horizon "
                        f"{H}...")
            LOGGER.info(f"    Creating HAR and ESN pipeline...")
            day_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 1})
            week_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 5})
            month_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 22})
            har_features = FeatureUnion(
                transformer_list=[("day", day_volatility_transformer),
                                  ("week", week_volatility_transformer),
                                  ("month", month_volatility_transformer)])

            initial_esn_params = {
                'hidden_layer_size': 50, 'k_in': 2, 'input_scaling': 0.4,
                'input_activation': 'identity', 'bias_scaling': 0.0,
                'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
                'reservoir_activation': 'tanh', 'bidirectional': False,
                'alpha': 1e-5, 'random_state': 42}
            esn_pipeline = Pipeline(
                steps=[("har_features", har_features),
                       ("scaler", MinMaxScaler()),
                       ("esn", TransformedTargetRegressor(
                           regressor=ESNRegressor(**initial_esn_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'esn__regressor__input_scaling': uniform(loc=1e-2, scale=1),
                'esn__regressor__spectral_radius': uniform(loc=0, scale=2)}
            step2_params = {'esn__regressor__leakage': uniform(1e-5, 1e0)}
            step3_params = {
                'esn__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step4_params = {
                'esn__regressor__hidden_layer_size':
                    [50, 100, 200, 400, 800, 1600, 3200, 6400],
                'esn__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step4 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                ('step3', RandomizedSearchCV, step3_params, kwargs_step3),
                ('step4', RandomizedSearchCV, step4_params, kwargs_step4)]
            search = SequentialSearchCV(
                esn_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyrcn_har_seq_esn_h{H}.joblib')

            ps_test = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_test = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={}, cv=ps_test,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_test, f'./results/pyrcn_har_seq_esn_h{H}_test.joblib')
            LOGGER.info("    ... done!")
        if fit_pyrcn_elm:
            LOGGER.info(f"Starting the ELM experiment with the horizon {H}...")
            LOGGER.info(f"    Creating feature extraction and ELM pipeline...")
            initial_elm_params = {
                'hidden_layer_size': 50, 'k_in': 1, 'input_scaling': 0.4,
                'input_activation': 'tanh', 'bias_scaling': 0.0, 'alpha': 1e-5,
                'random_state': 42}
            elm_pipeline = Pipeline(
                steps=[("scaler", MinMaxScaler()),
                       ("elm", TransformedTargetRegressor(
                           regressor=ELMRegressor(**initial_elm_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'elm__regressor__input_scaling': uniform(loc=1e-2, scale=1)}
            step2_params = {
                'elm__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step3_params = {
                'elm__regressor__hidden_layer_size':
                    [50, 100, 200, 400, 800, 1600, 3200, 6400],
                'elm__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                ('step3', RandomizedSearchCV, step3_params, kwargs_step3)]
            search = SequentialSearchCV(
                elm_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyrcn_seq_elm_h{H}.joblib')

            ps_test = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_test = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={}, cv=ps_test,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_test, f'./results/pyrcn_seq_elm_h{H}_test.joblib')
            LOGGER.info("    ... done!")
        if fit_har_pyrcn_elm:
            LOGGER.info(f"Starting the HAR-ELM experiment with the horizon "
                        f"{H}...")
            LOGGER.info(f"    Creating HAR and ELM pipeline...")
            day_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 1})
            week_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 5})
            month_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 22})
            har_features = FeatureUnion(
                transformer_list=[("day", day_volatility_transformer),
                                  ("week", week_volatility_transformer),
                                  ("month", month_volatility_transformer)])

            initial_elm_params = {
                'hidden_layer_size': 50, 'k_in': 2, 'input_scaling': 0.4,
                'input_activation': 'tanh', 'bias_scaling': 0.0, 'alpha': 1e-5,
                'random_state': 42}
            elm_pipeline = Pipeline(
                steps=[("har_features", har_features),
                       ("scaler", MinMaxScaler()),
                       ("elm", TransformedTargetRegressor(
                           regressor=ELMRegressor(**initial_elm_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'elm__regressor__input_scaling': uniform(loc=1e-2, scale=1)}
            step2_params = {
                'elm__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step3_params = {
                'elm__regressor__hidden_layer_size':
                    [50, 100, 200, 400, 800, 1600, 3200, 6400],
                'elm__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                ('step3', RandomizedSearchCV, step3_params, kwargs_step3)]
            search = SequentialSearchCV(
                elm_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyrcn_har_seq_elm_h{H}.joblib')

            ps_test = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_test = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={}, cv=ps_test,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_test, f'./results/pyrcn_har_seq_elm_h{H}_test.joblib')
            LOGGER.info("    ... done!")
        if fit_skelm_elm:
            LOGGER.info(f"Starting the ELM experiment with the horizon {H}...")
            LOGGER.info(f"    Creating feature extraction and ELM pipeline...")
            initial_elm_params = {
                'n_neurons': 50, 'density': 0.1, 'ufunc': 'tanh',
                'alpha': 1e-5, 'random_state': 42}
            elm_pipeline = Pipeline(
                steps=[("scaler", MinMaxScaler()),
                       ("elm", TransformedTargetRegressor(
                           regressor=SkELMRegressor(**initial_elm_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'elm__regressor__n_neurons':
                    [50, 100, 200, 400, 800, 1600, 3200, 6400],
                'elm__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1)]
            search = SequentialSearchCV(
                elm_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/skelm_seq_elm_h{H}.joblib')

            ps_test = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_test = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={}, cv=ps_test,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_test, f'./results/skelm_seq_elm_h{H}_test.joblib')
            LOGGER.info("    ... done!")
        if fit_har_skelm_elm:
            LOGGER.info(f"Starting the HAR-ELM experiment with the horizon "
                        f"{H}...")
            LOGGER.info(f"    Creating HAR and ELM pipeline...")
            day_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 1})
            week_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 5})
            month_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 22})
            har_features = FeatureUnion(
                transformer_list=[("day", day_volatility_transformer),
                                  ("week", week_volatility_transformer),
                                  ("month", month_volatility_transformer)])

            initial_elm_params = {
                'n_neurons': 50, 'density': 0.1, 'ufunc': 'tanh',
                'alpha': 1e-5, 'random_state': 42}
            elm_pipeline = Pipeline(
                steps=[("har_features", har_features),
                       ("scaler", MinMaxScaler()),
                       ("elm", TransformedTargetRegressor(
                           regressor=SkELMRegressor(**initial_elm_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'elm__regressor__n_neurons':
                    [50, 100, 200, 400, 800, 1600, 3200, 6400],
                'elm__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1)]
            search = SequentialSearchCV(
                elm_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/skelm_har_seq_elm_h{H}.joblib')

            ps_test = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_test = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={}, cv=ps_test,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_test, f'./results/skelm_har_seq_elm_h{H}_test.joblib')
            LOGGER.info("    ... done!")
        if fit_reservoirpy_esn:
            LOGGER.info(f"Starting the ESN experiment with the horizon {H}...")
            LOGGER.info(f"    Creating feature extraction and ESN pipeline...")
            initial_esn_params = {
                'units': 50, 'lr': 1.0, 'sr': 0.0, 'noise_rc': 1e-5,
                'input_scaling': 0.4, 'fb_scaling': 1.0,
                'input_connectivity': 0.1, 'rc_connectivity': 0.1,
                'seed': 42, 'ridge': 1e-5}
            esn_pipeline = Pipeline(
                steps=[("scaler", MinMaxScaler()),
                       ("esn", TransformedTargetRegressor(
                           regressor=ReservoirPyESN(**initial_esn_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'esn__regressor__input_scaling': uniform(loc=1e-2, scale=1),
                'esn__regressor__spectral_radius': uniform(loc=0, scale=2)}
            step2_params = {'esn__regressor__leakage': uniform(1e-5, 1e0)}
            step3_params = {
                'esn__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step4_params = {'esn__regressor__ridge': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': 1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                ('step3', RandomizedSearchCV, step3_params, kwargs_step3),
                ('step4', RandomizedSearchCV, step4_params, kwargs_step4)]
            search = SequentialSearchCV(
                esn_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyrcn_seq_h{H}.joblib')

            ps_final = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_final = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={"esn__regressor__hidden_layer_size": [
                    50, 100, 200, 400, 800, 1600]}, cv=ps_final,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_final, f'./results/pyrcn_seq_h{H}_final.joblib')
            LOGGER.info("    ... done!")
        if fit_har_reservoirpy_esn:
            LOGGER.info(f"Starting the HAR-ESN experiment with the horizon "
                        f"{H}...")
            LOGGER.info(f"    Creating HAR and ESN pipeline...")
            day_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 1})
            week_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 5})
            month_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 22})
            har_features = FeatureUnion(
                transformer_list=[("day", day_volatility_transformer),
                                  ("week", week_volatility_transformer),
                                  ("month", month_volatility_transformer)])

            initial_esn_params = {
                'hidden_layer_size': 50, 'k_in': 2, 'input_scaling': 0.4,
                'input_activation': 'identity', 'bias_scaling': 0.0,
                'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
                'reservoir_activation': 'tanh', 'bidirectional': False,
                'alpha': 1e-5, 'random_state': 42}
            esn_pipeline = Pipeline(
                steps=[("har_features", har_features),
                       ("scaler", MinMaxScaler()),
                       ("esn", TransformedTargetRegressor(
                           regressor=ESNRegressor(**initial_esn_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'esn__regressor__input_scaling': uniform(loc=1e-2, scale=1),
                'esn__regressor__spectral_radius': uniform(loc=0, scale=2)}
            step2_params = {'esn__regressor__leakage': uniform(1e-5, 1e0)}
            step3_params = {
                'esn__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step4_params = {'esn__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                ('step3', RandomizedSearchCV, step3_params, kwargs_step3),
                ('step4', RandomizedSearchCV, step4_params, kwargs_step4)]
            search = SequentialSearchCV(
                esn_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyrcn_har_seq_h{H}.joblib')

            ps_final = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_final = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={"esn__regressor__hidden_layer_size": [
                    50, 100, 200, 400, 800, 1600]}, cv=ps_final,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_final, f'./results/pyrcn_har_seq_h{H}_final.joblib')
            LOGGER.info("    ... done!")
        if fit_pyesn_esn:
            LOGGER.info(f"Starting the ESN experiment with the horizon {H}...")
            LOGGER.info(f"    Creating feature extraction and ESN pipeline...")
            initial_esn_params = {
                'n_inputs': 1, 'n_outputs': 1, 'n_reservoir': 50,
                'spectral_radius': 0.0, 'sparsity': 0.5, 'noise': 1e-3,
                'input_shift': 0, 'input_scaling': 0.4,
                'teacher_forcing': True, 'teacher_scaling': 1.,
                'teacher_shift': 0, 'random_state': 42}
            esn_pipeline = Pipeline(
                steps=[("scaler", MinMaxScaler()),
                       ("esn", TransformedTargetRegressor(
                           regressor=PyESN(**initial_esn_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'esn__regressor__input_scaling': uniform(loc=1e-2, scale=1),
                'esn__regressor__spectral_radius': uniform(loc=0, scale=2)}
            # step2_params = {'esn__regressor__leakage': uniform(1e-5, 1e0)}
            # step3_params = {
            #     'esn__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step4_params = {'esn__regressor__noise': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': 1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            """
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            """
            kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                # ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                # ('step3', RandomizedSearchCV, step3_params, kwargs_step3),
                ('step4', RandomizedSearchCV, step4_params, kwargs_step4)]
            search = SequentialSearchCV(
                esn_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyesn_seq_h{H}.joblib')

            ps_final = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_final = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={"esn__regressor__hidden_layer_size": [
                    50, 100, 200, 400, 800, 1600]}, cv=ps_final,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_final, f'./results/pyesn_seq_h{H}_final.joblib')
            LOGGER.info("    ... done!")
        if fit_har_pyesn_esn:
            LOGGER.info(f"Starting the HAR-ESN experiment with the horizon "
                        f"{H}...")
            LOGGER.info(f"    Creating HAR and ESN pipeline...")
            day_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 1})
            week_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 5})
            month_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 22})
            har_features = FeatureUnion(
                transformer_list=[("day", day_volatility_transformer),
                                  ("week", week_volatility_transformer),
                                  ("month", month_volatility_transformer)])

            initial_esn_params = {
                'n_inputs': 3, 'k_in': 2, 'input_scaling': 0.4,
                'input_activation': 'identity', 'bias_scaling': 0.0,
                'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
                'reservoir_activation': 'tanh', 'bidirectional': False,
                'alpha': 1e-5, 'random_state': 42}
            esn_pipeline = Pipeline(
                steps=[("har_features", har_features),
                       ("scaler", MinMaxScaler()),
                       ("esn", TransformedTargetRegressor(
                           regressor=PyESN(**initial_esn_params),
                           transformer=MinMaxScaler()))])
            LOGGER.info("    ... done!")

            # Prepare data
            LOGGER.info(f"    Preparing the dataset...")
            df = pd.concat([ts2super(d, 0, H) for d in data])
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, -1].values.reshape(-1, 1)
            test_fold = [
                [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
            test_fold = list(itertools.chain.from_iterable(test_fold))

            ps = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=True)
            LOGGER.info("    ... done!")

            # Run model selection
            LOGGER.info(f"    Performing the grid search...")
            step1_params = {
                'esn__regressor__input_scaling': uniform(loc=1e-2, scale=1),
                'esn__regressor__spectral_radius': uniform(loc=0, scale=2)}
            step2_params = {'esn__regressor__leakage': uniform(1e-5, 1e0)}
            step3_params = {
                'esn__regressor__bias_scaling': uniform(loc=0, scale=3)}
            step4_params = {'esn__regressor__alpha': loguniform(1e-5, 1e1)}
            scoring = {
                "MSE": "neg_mean_squared_error",
                "RMSE": "neg_root_mean_squared_error", "R2": "r2"
            }
            kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}
            kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1,
                            'n_jobs': -1, 'scoring': scoring, "refit": "R2",
                            "cv": ps, "return_train_score": True}

            searches = [
                ('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                ('step3', RandomizedSearchCV, step3_params, kwargs_step3),
                ('step4', RandomizedSearchCV, step4_params, kwargs_step4)]
            search = SequentialSearchCV(
                esn_pipeline, searches=searches).fit(X, y)
            dump(search, f'./results/pyesn_har_seq_h{H}.joblib')

            ps_final = PredefinedTrainValidationTestSplit(
                test_fold=test_fold, validation=False)
            LOGGER.info("    ... done!")
            LOGGER.info("    Final evaluation...")
            search_final = GridSearchCV(
                estimator=clone(search.best_estimator_),
                param_grid={"esn__regressor__hidden_layer_size": [
                    50, 100, 200, 400, 800, 1600]}, cv=ps_final,
                scoring={"MSE": "neg_mean_squared_error",
                         "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
                refit="R2", return_train_score=True).fit(X, y)
            dump(search_final, f'./results/pyesn_har_seq_h{H}_final.joblib')
            LOGGER.info("    ... done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_arima", action="store_true")
    parser.add_argument("--fit_mlp", action="store_true")
    parser.add_argument("--fit_har", action="store_true")
    parser.add_argument("--fit_pyrcn_esn", action="store_true")
    parser.add_argument("--fit_har_pyrcn_esn", action="store_true")
    parser.add_argument("--fit_pyrcn_elm", action="store_true")
    parser.add_argument("--fit_har_pyrcn_elm", action="store_true")
    parser.add_argument("--fit_skelm_elm", action="store_true")
    parser.add_argument("--fit_har_skelm_elm", action="store_true")
    parser.add_argument("--fit_reservoirpy_esn", action="store_true")
    parser.add_argument("--fit_har_reservoirpy_esn", action="store_true")
    parser.add_argument("--fit_pyesn_esn", action="store_true")
    parser.add_argument("--fit_har_pyesn_esn", action="store_true")
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    LOGGER.setLevel(logging.DEBUG)
    main(**args)
    exit(0)
