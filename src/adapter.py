"""This module contains various adapter classes for toolboxes to sklearn."""
from src.pyESN.pyESN import ESN
from src.pyESN.pyESN import identity
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import _deprecate_positional_args
from reservoirpy.nodes import Reservoir, Ridge
import numpy as np


class PyESN(ESN, RegressorMixin):

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001,
                 input_shift=None, input_scaling=None, teacher_forcing=True,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, silent=True):
        super().__init__(
            n_inputs=n_inputs, n_outputs=n_outputs, n_reservoir=n_reservoir,
            spectral_radius=spectral_radius, sparsity=sparsity, noise=noise,
            input_shift=input_shift, input_scaling=input_scaling,
            teacher_forcing=teacher_forcing, teacher_scaling=teacher_scaling,
            teacher_shift=teacher_shift, out_activation=out_activation,
            inverse_out_activation=inverse_out_activation,
            random_state=random_state, silent=silent)

    def get_params(self, deep=True):
        """Get all parameters."""
        return {
            "input_scaling": self.input_scaling[0],
            "input_shift": self.input_shift[0],
            "n_inputs": self.n_inputs, "n_outputs": self.n_outputs,
            "n_reservoir": self.n_reservoir, "noise": self.noise,
            "random_state": self.random_state, "sparsity": self.sparsity,
            "spectral_radius": self.spectral_radius,
            "teacher_scaling": self.teacher_scaling,
            "teacher_shift": self.teacher_shift}

    def set_params(self, **parameters: dict):
        for key in parameters.keys():
            if key in self.get_params():
                super().__setattr__(key, parameters[key])
        return self

    def fit(self, X, y):
        pred_train = super(ESN).fit(inputs=X, outputs=y, inspect=False)
        return self

    def predict(self, X, continuation=True):
        return super(ESN).predict(inputs=X, continuation=continuation)


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
