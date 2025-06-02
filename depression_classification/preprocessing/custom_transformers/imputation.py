import pandas as pd
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin


class RelationalImputer(BaseEstimator, TransformerMixin):
    """
    Conditionally imputes missing values in specified columns based on another column's value.
    Args:
        cols_to_impute (list of str): Columns to impute missing values in.
        condition_col (str): Column used to define the imputation condition.
        condition_value (str): Value in `condition_col` that triggers imputation
        strategy (str, default='constant'): Imputation strategy ('constant', 'median', or 'most_frequent').
        fill_value (any, optional): Value to use for 'constant' strategy.
    Raises:
        ValueError: If strategy is invalid or required columns are missing.
    """

    def __init__(self,
                 cols_to_impute: list["str"],
                 condition_col: str,
                 condition_value: str,
                 strategy: str = 'constant',
                 fill_value: Any = None
                 ):
        self.cols_to_impute = cols_to_impute
        self.condition_col = condition_col
        self.condition_value = condition_value
        self.strategy = strategy
        self.fill_value = fill_value
        self.medians_ = {}
        self.modes_ = {}
        if strategy not in ['constant', 'median', 'most_frequent']:
            raise ValueError("Unknown strategy. Choose: 'constant', 'median' or 'most_frequent'")
        if strategy == 'constant' and fill_value is None:
            raise ValueError("fill_value must be set for 'constant' strategy.")

    def fit(self, X: pd.DataFrame, y=None) -> "RelationalImputer":
        """Learn imputation values from data where condition is met."""
        required_cols = set(self.cols_to_impute) | {self.condition_col}
        missing_cols = required_cols - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        mask = X[self.condition_col] == self.condition_value
        if self.strategy == 'median':
            for col in self.cols_to_impute:
                self.medians_[col] = X.loc[mask, col].median()
        elif self.strategy == 'most_frequent':
            for col in self.cols_to_impute:
                mode_series = X.loc[mask, col].mode(dropna=True)
                self.modes_[col] = mode_series.iloc[0] if not mode_series.empty else self.fill_value
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values where condition is met."""
        X = X.copy()
        for col in self.cols_to_impute:
            mask = (X[self.condition_col] == self.condition_value) & (X[col].isna())
            if self.strategy == 'median':
                X.loc[mask, col] = self.medians_[col]
            elif self.strategy == 'constant':
                X.loc[mask, col] = self.fill_value
            elif self.strategy == 'most_frequent':
                X.loc[mask, col] = self.modes_[col]
        return X

    def set_output(self, transform="pandas"):
        return self
