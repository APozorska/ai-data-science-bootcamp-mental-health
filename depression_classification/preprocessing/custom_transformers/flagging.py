import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InconsistencyFlagger(BaseEstimator, TransformerMixin):
    """
    Adds an 'inconsistency_flag' column to the DataFrame, marking rows with logical inconsistencies
    based on pressure, satisfaction, and occupation/profession fields.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['inconsistency_flag'] = (
            ((X['work_pressure'] > 0) & (X['academic_pressure'] > 0)).astype(int) +
            ((X['study_satisfaction'] > 0) & (X['job_satisfaction'] > 0)).astype(int) +
            (
                ((X['occupation_status'] == 'Working Professional') & (X['profession'] == 'Student')) |
                ((X['occupation_status'] == 'Student') & (X['profession'] != 'Student') & (X['profession'].notna()))
            ).astype(int)
        )
        return X


class ConditionalFlagger(BaseEstimator, TransformerMixin):
    """
    Adds binary flag columns for missing values in specified columns, using user-defined conditional rules.
    Flags are set as 'not applicable' or 'imputed' depending on another column's value, as defined in the flagging_map dictionary.
    """
    def __init__(self, flagging_map, suffix_map=None):
        self.flagging_map = flagging_map
        self.suffix_map = suffix_map or {'not_applicable': '_not_applicable', 'imputed': '_imputed'}

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, conditions in self.flagging_map.items():
            for flag_type, cond in conditions.items():
                cond_col, cond_val = list(cond.items())[0]
                mask = (X[cond_col] == cond_val) & (X[col].isna())
                flag_col = f"{col}{self.suffix_map[flag_type]}"
                X[flag_col] = 0
                X.loc[mask, flag_col] = 1
        return X

