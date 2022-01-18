"""
Preprocessing utilities required to reproduce the results in the paper
'Template Repository for Research Papers with Python Code'.
"""
# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

from typing import List, Tuple
from pandas import DataFrame
import numpy as np
from sklearn.compose import ColumnTransformer


def select_features(
        df: DataFrame, input_features: List, target: str = "SalePrice")\
        -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    input_trf = ColumnTransformer(
        transformers=[("selector", "passthrough", input_features)],
        remainder="drop").fit(df)
    X = input_trf.fit_transform(df)
    y = ColumnTransformer(
        transformers=[("selector", "passthrough", [target])],
        remainder="drop").fit_transform(df)
    return X, y, input_trf
