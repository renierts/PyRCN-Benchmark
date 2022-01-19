"""Custom model selection tools."""
from sklearn.model_selection import PredefinedSplit
import numpy as np


class PredefinedTrainValidationTestSplit(PredefinedSplit):

    def __init__(self, test_fold, validation=True):
        super().__init__(test_fold=test_fold)
        self.validation = validation

    def split(self, X=None, y=None, groups=None):
        """
        Generate indices to split data into training, validation and test set.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if self.validation:
            ind = np.arange(len(self.test_fold))
            for (vali_index, test_index) in self._iter_vali_test_masks():
                train_index = ind[np.logical_not(vali_index + test_index)]
                vali_index = ind[vali_index]
                yield train_index, vali_index
        else:
            yield super().split(X=X, y=y, groups=groups)

    def _iter_vali_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            test_mask = np.zeros(len(self.test_fold), dtype=bool)
            test_mask[test_index] = True
            yield test_mask


if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    test_fold = [0, 1, 2]
    ps = PredefinedTrainValidationTestSplit(test_fold, validation=True)
    ps.get_n_splits()

    print(ps)

    for train_index, test_index in ps.split():
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
