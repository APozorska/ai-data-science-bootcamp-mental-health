import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class UniformBinner(BaseEstimator, TransformerMixin):

    def __init__(self, col, n_bins=3, labels=None):
        self.col = col
        self.n_bins = n_bins
        self.labels = labels if labels is not None else {i: f'bin_{i}' for i in range(n_bins)}
        self.est = None

    def fit(self, X: pd.DataFrame, y=None):
        mask = X[self.col].notna()
        self.est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        self.est.fit(X.loc[mask, [self.col]])
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        mask_valid = X[self.col].notna()
        if mask_valid.any():
            X.loc[mask_valid, self.col] = (self.est.transform(X.loc[mask_valid, [self.col]]).astype(int).flatten())
            X.loc[mask_valid, self.col] = X.loc[mask_valid, self.col].map(self.labels)
        return X
