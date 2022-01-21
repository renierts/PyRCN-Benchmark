"""Custom model selection tools."""
from sklearn.model_selection import PredefinedSplit
import numpy as np


class PredefinedTrainValidationTestSplit(PredefinedSplit):
    """Predefined split cross-validator
    Provides train/test indices to split data into training/validation/test
    sets using a predefined scheme specified by the user with the ``test_fold``
    parameter.

    Parameters
    ----------
    test_fold : array-like of shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set (i.e. include sample ``i`` in every training set) by
        setting ``test_fold[i]`` equal to -1.
    validation : bool, default=True
        Whether to split the dataset ino training/validation or in
        training/test.

    Examples
    --------
    >>> import numpy as np
    >>> from src.model_selection import PredefinedTrainValidationTestSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> test_fold = [0, 1, 2, -1]
    >>> ps = PredefinedTrainValidationTestSplit(test_fold, validation=True)
    >>> ps.get_n_splits()
    3
    >>> print(ps)
    PredefinedTrainValidationTestSplit(test_fold=array([ 0,  1,  2, -1]),
                  validation=True)
    >>> for train_index, vali_index in ps.split():
    ...     print("TRAIN:", train_index, "VAL:", vali_index)
    ...     X_train, X_vali = X[train_index], X[vali_index]
    TRAIN: [1 3] VAL: [2]
    TRAIN: [2 3] VAL: [0]
    TRAIN: [0 3] VAL: [1]
    >>> ps = PredefinedTrainValidationTestSplit(test_fold, validation=False)
    >>> ps.get_n_splits()
    3
    >>> print(ps)
    PredefinedTrainValidationTestSplit(test_fold=array([ 0,  1,  2, -1]),
                  validation=False)
    >>> for train_index, test_index in ps.split():
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    TRAIN: [1 3] TEST: [0]
    TRAIN: [2 3] TEST: [1]
    TRAIN: [0 3] TEST: [2]
    """

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
        vali : ndarray
            The validation set indices for that split. Only with ``validation``
        test : ndarray
            The testing set indices for that split.
        """
        ind = np.arange(len(self.test_fold))
        if self.validation:
            for (vali_index, test_index) in self._iter_vali_test_masks():
                train_index = ind[np.logical_not(vali_index + test_index)]
                vali_index = ind[vali_index]
                yield train_index, vali_index
        else:
            for test_index in self._iter_test_masks():
                train_index = ind[np.logical_not(test_index)]
                test_index = ind[test_index]
                yield train_index, test_index

    def _iter_vali_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            if f == 0:
                vali_index = np.where(
                    self.test_fold == max(self.unique_folds))[0]
            else:
                vali_index = np.where(self.test_fold == f - 1)[0]
            vali_mask = np.zeros(len(self.test_fold), dtype=bool)
            vali_mask[vali_index] = True
            test_mask = np.zeros(len(self.test_fold), dtype=bool)
            test_mask[test_index] = True
            yield vali_mask, test_mask
