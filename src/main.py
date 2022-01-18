"""
Main Code to reproduce the results in the paper
'Template Repository for Research Papers with Python Code'.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

import logging

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from file_handling import (
    load_data, export_results, serialize_model, deserialize_model)
from preprocessing import select_features

from pyrcn.extreme_learning_machine import ELMRegressor


LOGGER = logging.getLogger(__name__)


def main(plot=False, export=False, serialize=False):
    """
    This is the main function to reproduce all visualizations and models for
    the paper "Template Repository for Research Papers with Python Code".

    It is controlled via command line arguments:

    Params
    ------
    plot : bool, default=False
        Create all plots in the publication.
    export: default=False
        Store the results in ``data/results.dat``
    serialize:
        Store the fitted model in ``data/model.joblib``

    Returns
    -------
    results : dict
        Results that are stored in data/results.dat
    """

    LOGGER.info("Loading the training dataset...")
    training_data = load_data("./data/train.csv")
    LOGGER.info("... done!")

    if plot:
        fig, axs = plt.subplots()
        sns.pairplot(data=training_data, vars=["LotArea", "OverallQual",
                                               "OverallCond", "YearBuilt",
                                               "YearRemodAdd", "GrLivArea",
                                               "SalePrice"])
        plt.title("Training data")
        plt.tight_layout()

    LOGGER.info("Selecting input feature set...")
    X, y, feature_trf = select_features(
        df=training_data, input_features=["GrLivArea"], target="SalePrice")
    LOGGER.info("... done!")

    LOGGER.info("Scaling the dataset to have zero mean and a variance of 1...")
    scaler = StandardScaler().fit(X)
    X_train = scaler.transform(X)
    y_train = y
    LOGGER.info("... done!")

    try:
        LOGGER.info("Attempting to load a pre-trained model...")
        model = deserialize_model("./results/model.joblib")
    except FileNotFoundError:
        LOGGER.info("... No model serialized yet.")
        LOGGER.info("Fitting a new model...")
        model = RandomizedSearchCV(
            estimator=ELMRegressor(input_activation="relu", random_state=42,
                                   hidden_layer_size=50),
            param_distributions={"input_scaling": uniform(loc=0, scale=2),
                                 "bias_scaling": uniform(loc=0, scale=2),
                                 "alpha": loguniform(1e-5, 1e1)},
            random_state=42, n_iter=200, refit=True).fit(X_train, y_train)

    LOGGER.info("... done!")
    if serialize:
        serialize_model(model, "./results/model.joblib")

    if plot:
        y_pred = model.predict(X)
        sns.scatterplot(x=X.ravel(), y=y_pred.ravel(), ax=axs)

    LOGGER.info("Loading the test dataset...")
    test_data = load_data("./data/test.csv")
    LOGGER.info("... done!")

    LOGGER.info("Selecting input feature set...")
    X = feature_trf.transform(test_data)
    LOGGER.info("... done!")
    LOGGER.info("Scaling the dataset with the fitted training scaler...")
    X_test = scaler.transform(X)
    LOGGER.info("... done!")
    LOGGER.info("Predicting prices on the test set...")
    y_pred = model.predict(X_test)
    LOGGER.info("... done!")
    if plot:
        fig, axs = plt.subplots()
        sns.scatterplot(x=X.ravel(), y=y_pred.ravel(), ax=axs)
        plt.ylabel("Predicted SalePrice")
        plt.title("Test data")
        plt.tight_layout()

    results = {
        "GrLivArea": test_data["GrLivArea"], "PredictedSalePrice":
            y_pred.ravel()}

    if export:
        LOGGER.info("Storing results...")
        export_results(results, "./results/results.csv")
        LOGGER.info("... done!")
    if plot:
        plt.show()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # TODO: Specify command line arguments to add runtime options for the code.
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--serialize", action="store_true")
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    LOGGER.setLevel(logging.DEBUG)
    main(**args)
    exit(0)
