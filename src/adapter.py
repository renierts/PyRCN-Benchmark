"""This module contains various adapter classes for toolboxes to sklearn."""
from src.pyESN.pyESN import ESN
from src.pyESN.pyESN import identity
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import _deprecate_positional_args
from reservoirpy.nodes import Reservoir, Ridge
import numpy as np


class PyESN(BaseEstimator, RegressorMixin):

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001,
                 input_shift=None, input_scaling=None, teacher_forcing=True,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, silent=True):
        self.input_shift = input_shift
        self.input_scaling = input_scaling
        self._model = ESN(
            n_inputs=n_inputs, n_outputs=n_outputs, n_reservoir=n_reservoir,
            spectral_radius=spectral_radius, sparsity=sparsity, noise=noise,
            input_shift=self.input_shift, input_scaling=self.input_scaling,
            teacher_forcing=teacher_forcing, teacher_scaling=teacher_scaling,
            teacher_shift=teacher_shift, out_activation=out_activation,
            inverse_out_activation=inverse_out_activation,
            random_state=random_state, silent=silent)

    def get_params(self, deep=True):
        if self.input_scaling != np.unique(self._model.input_scaling):
            raise ValueError("Input scaling different")
        if self.input_shift != self._model.input_shift:
            raise ValueError("Input shift different")
        return {"input_scaling": self.input_scaling,
                "input_shift": self.input_shift,
                "n_inputs": self._model.n_inputs,
                "n_outputs": self._model.n_outputs,
                "n_reservoir": self._model.n_reservoir,
                "spectral_radius": self._model.spectral_radius,
                "sparsity": self._model.sparsity, "noise": self._model.noise,
                "teacher_forcing": self._model.teacher_forcing,
                "teacher_scaling": self._model.teacher_scaling,
                "teacher_shift": self._model.teacher_shift,
                "out_activation": self._model.out_activation,
                "inverse_out_activation": self._model.inverse_out_activation,
                "random_state": self._model.random_state,
                "silent": self._model.silent}

    def set_params(self, **params):
        if "input_scaling" in params.keys():
            self.input_scaling = params.pop("input_scaling")
            self._model.__setattr__("input_scaling", self.input_scaling)
        if "input_shift" in params.keys():
            self.input_shift = params.pop("input_shift")
            self._model.__setattr__("input_shift", self.input_shift)
        for key, value in params.items():
            self._model.__setattr__(key, value)
        return self

    def fit(self, X, y):
        pred_train = self._model.fit(inputs=X, outputs=y, inspect=False)
        return self

    def predict(self, X, continuation=True):
        y_pred = self._model.predict(inputs=X, continuation=continuation)
        return y_pred


class ReservoirPyESN(BaseEstimator, RegressorMixin):

    @_deprecate_positional_args
    def __init__(self, units=50, lr=1.0, sr=None, noise_rc=0.0,
                 input_scaling=1.0, fb_scaling=1.0, input_connectivity=0.1,
                 rc_connectivity=0.1, seed=None, ridge=1e-5, reservoir=None,
                 readout=None):
        if reservoir is not None:
            self.reservoir = reservoir
        else:
            self.reservoir = Reservoir(
                units=units, lr=lr, sr=sr, noise_rc=noise_rc,
                input_scaling=input_scaling, fb_scaling=fb_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity, seed=seed)
        if readout is not None:
            self.readout = readout
        else:
            self.readout = Ridge(1, input_bias=True, ridge=ridge)
        self.esn = self.reservoir >> self.readout

    def get_params(self, deep=True):
        """Get all parameters."""
        if deep:
            return {**self.reservoir.hypers, **self.readout.hypers}
        else:
            return {"reservoir": self.reservoir, "readout": self.readout}

    def set_params(self, **parameters: dict):
        reservoir_params = self.reservoir.hypers.keys()
        for key in parameters.keys():
            if key in reservoir_params:
                self.reservoir.hypers[key] = parameters[key]
        ridge_params = self.readout.hypers.keys()
        for key in parameters.keys():
            if key in ridge_params:
                self.readout.hypers[key] = parameters[key]
        return self

    def fit(self, X, y):
        self.esn.fit(X=X, Y=y)
        return self

    def predict(self, X):
        return self.esn.run(X=X)
