from sklearn.feature_selection import SelectFromModel
from depression_classification.models.models import get_model


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
