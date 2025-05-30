from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_cat_multiclass_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),

    ])


def get_cat_highcard_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
        ("encoder", TargetEncoder()),
    ])


def get_cat_binary_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ])


def get_cat_custom_pipe():
    return Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),])


def get_cat_preprocessor(cat_multiclass_features, cat_highcard_features, cat_binary_features):
    return ColumnTransformer([
        ("cat_multiclass", get_cat_multiclass_pipe(), cat_multiclass_features),
        ("cat_highcard", get_cat_highcard_pipe(), cat_highcard_features),
        ("cat_binary", get_cat_binary_pipe(), cat_binary_features),
    ])


def get_num_standard_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def get_num_custom_pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
    ])


def get_num_preprocessor(num_standard_features, num_custom_features):
    return ColumnTransformer([
        ("num_standard", get_num_standard_pipe(), num_standard_features),
        ("num_custom", get_num_custom_pipe(), num_custom_features),
    ])


def get_standard_preprocessor(num_standard_features, num_custom_features,
                              cat_multiclass_features, cat_highcard_features, cat_binary_features):
    transformer = ColumnTransformer([
        ("numerical", get_num_preprocessor(num_standard_features,num_custom_features),
         num_standard_features+num_custom_features),
        ("categorical", get_cat_preprocessor(cat_multiclass_features, cat_highcard_features, cat_binary_features),
         cat_binary_features+cat_multiclass_features+cat_highcard_features),
    ], remainder='passthrough')

    return "standard_preprocessor", transformer
