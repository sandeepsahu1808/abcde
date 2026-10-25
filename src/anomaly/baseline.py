"""Healthy baseline: 'what should this value be, given the operating context?'

Each feature is regressed on the context columns (e.g. crane load) using
healthy data only. residual / healthy-std = a z-score that is already
load-adjusted. With no context columns it is a plain mean/std baseline.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Baseline:
    def fit(self, X, C):
        self.cols, self.ctx_cols = list(X.columns), list(C.columns)
        self.models, self.std = {}, {}
        for c in self.cols:
            if self.ctx_cols:
                m = LinearRegression().fit(C.values, X[c].values)
                resid = X[c].values - m.predict(C.values)
            else:
                m = float(X[c].mean())
                resid = X[c].values - m
            self.models[c] = m
            self.std[c] = max(float(np.std(resid)), 0.1)
        return self

    def residuals(self, X, C):
        Z = {}
        for c in self.cols:
            pred = self.models[c].predict(C.values) if self.ctx_cols else self.models[c]
            Z[c] = np.clip((X[c].values - pred) / self.std[c], -20.0, 20.0)
        return pd.DataFrame(Z, index=X.index)
