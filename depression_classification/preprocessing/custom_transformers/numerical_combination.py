import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Combines two numerical columns using a specified aggregation strategy.
    Args
    col1: First column name.
    col2: Second column name.
    strategy: Aggregation method, {'max', 'min', 'sum', 'mean'}, default='max'
    new_col_name: Name of the new combined column.
    """
    def __init__(self, col1: str, col2: str, strategy='max', new_col_name=None):
        self.col1 = col1
        self.col2 = col2
        self.strategy = strategy
        self.new_col_name = new_col_name or f"{col1}_{strategy}_{col2}"

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if self.strategy == 'max':
            X[self.new_col_name] = X[[self.col1, self.col2]].max(axis=1)
        elif self.strategy == 'min':
            X[self.new_col_name] = X[[self.col1, self.col2]].min(axis=1)
        elif self.strategy == 'sum':
            X[self.new_col_name] = X[self.col1] + X[self.col2]
        elif self.strategy == 'mean':
            X[self.new_col_name] = (X[self.col1] + X[self.col2]) / 2
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return X

    def set_output(self, transform="pandas"):
        return self
