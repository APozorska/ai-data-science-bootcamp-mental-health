import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCategoryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mappings, other_value="other", normalize_func=None):
        """
        mappings: dict, np. {'dietary_habits': dietary_mapping, 'sleep_duration': sleep_mapping}
        """
        self.mappings = mappings
        self.other_value = other_value
        self.normalize_func = normalize_func
        self.value_to_main_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.value_to_main_ = {}
        for col, mapping in self.mappings.items():
            value_to_main_col = {}
            for main_cat, variants in mapping.items():
                for v in variants:
                    key = self.normalize_func(v) if self.normalize_func else v
                    value_to_main_col[key] = main_cat
            self.value_to_main_[col] = value_to_main_col
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.mappings.keys():
            def map_func(x):
                if pd.isna(x):
                    return x
                key = self.normalize_func(x) if self.normalize_func else x
                return self.value_to_main_[col].get(key, self.other_value)

            X[col] = X[col].apply(map_func)
        return X

    def set_output(self, transform="pandas"):
        return self


class RareCategoryCombiner(BaseEstimator, TransformerMixin):

    def __init__(self, columns, min_count=10, other_label='other'):
        self.columns = columns
        self.min_count = min_count
        self.other_label = other_label
        self.rare_cats_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.columns:
            counts = X[col].value_counts(dropna=True)
            self.rare_cats_[col] = counts[counts < self.min_count].index.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.columns:
            mask = X[col].isin(self.rare_cats_[col])
            X.loc[mask, col] = self.other_label
        return X

    def set_output(self, transform="pandas"):
        return self
