from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel
from depression_classification.models.models import get_model


def train_and_optimize(X_train, y_train, estimator, param_grid, scoring, cv):
    optimizer = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    optimizer.fit(X_train, y_train)
    return optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_


def build_param_grid(config, model_name):
    base_params = config["models"][model_name]["params_grid"]
    fs_estimator_name = config["feature_selection"]["estimator"]
    fs_estimator_params = config["feature_selection"]["estimator_params"]
    _, fs_estimator = get_model(fs_estimator_name, fs_estimator_params)
    fs_params_grid = config["feature_selection"]["params_grid"]
    fs_params = config["feature_selection"]["params"]
    feature_selector_options = [
        {
            "feature_selector": ["passthrough"]
        },
        {
            "feature_selector": [SelectFromModel(fs_estimator, **fs_params)],
            **fs_params_grid,
        }
    ]
    param_grid = [{**base_params, **selector} for selector in feature_selector_options]
    return param_grid


def get_cv(config):
    """
    Return a StratifiedKFold cross-validation splitter based on config.
    Args:
        config (dict): cross_validation section from config.
            Should contain n_splits, shuffle, random_state.
    Returns:
        StratifiedKFold object
    """
    return StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=config["shuffle"],
        random_state=config["random_state"]
    )

