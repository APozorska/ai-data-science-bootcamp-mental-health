import yaml
from pathlib import Path


def load_task_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return config


def get_target_col(config: dict) -> str:
    return config["data"]["target_col"]


def get_feature_groups(config: dict, logger) -> dict[str, list[str]]:
    """
    Extract all feature groups from the config and return as a dictionary.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        dict: Dictionary with keys for each feature group.
    """
    feature_groups = {
        "cat_binary": config["data"]["features"]["cat_binary_features"],
        "cat_multiclass": config["data"]["features"]["cat_multiclass_features"],
        "cat_highcard": config["data"]["features"]["cat_highcard_features"],
        "num_standard": config["data"]["features"]["num_standard_features"],
        "num_custom": config["data"]["features"]["num_custom_features"],
    }
    logger.info(
        f"Feature groups loaded. "
        f"cat_binary: {len(feature_groups['cat_binary'])}, "
        f"cat_multiclass: {len(feature_groups['cat_multiclass'])}, "
        f"cat_highcard: {len(feature_groups['cat_highcard'])}, "
        f"num_standard: {len(feature_groups['num_standard'])}, "
        f"num_custom: {len(feature_groups['num_custom'])}"
    )
    logger.debug(
        "Categorical features:\n"
        f"  cat_binary_features: {feature_groups['cat_binary']}\n"
        f"  cat_multiclass_features: {feature_groups['cat_multiclass']}\n"
        f"  cat_highcard_features: {feature_groups['cat_highcard']}\n"
        "Numerical features:\n"
        f"  num_standard_features: {feature_groups['num_standard']}\n"
        f"  num_custom_features: {feature_groups['num_custom']}\n"
    )
    return feature_groups
