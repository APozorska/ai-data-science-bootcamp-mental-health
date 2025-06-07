import json
import joblib
import pandas as pd

from depression_classification.utils.config import load_task_config
from depression_classification.utils.logger import get_logger
from depression_classification.data.loader import load_csv

from depression_classification.preprocessing.data_checks import remove_unnecessary_columns

logger = get_logger("PREDICT EXTERNAL DATA", log_level="DEBUG")


def main(config_path):

    logger.info("*" * 80)

    # Load configuration
    logger.info("=== LOAD CONFIGURATION ===")
    try:
        config = load_task_config(config_path)
        logger.info("Configuration loaded.")
        logger.debug(f"Config: {config}")
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

    # Load data
    logger.info("=== LOAD EXTERNAL DATA FROM CONFIGURATION ===")
    columns = config["data"]["columns"][:-1]
    external_data_path = config["data"]["external_data_path"]
    try:
        df_external = load_csv(external_data_path, columns)
        logger.info(f"Data loaded:{df_external.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {external_data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    logger.info("=== REMOVE UNNECESSARY COLUMNS ===")
    ids = df_external['id'].copy()
    df_external = remove_unnecessary_columns(df_external, logger, ["name", "id"])
    logger.debug(f"First rows of the data:\n{df_external.head(3).to_string()}")

    # Load best model
    logger.info("=== LOAD BEST MODEL ===")
    metadata_path = config["outputs"]["final_metadata_output_path"]
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    model_path = metadata["model_path"]

    best_threshold = metadata["best_threshold"]
    final_model = joblib.load(model_path)
    logger.info("Best model loaded.")

    logger.info("=== PREDICT ===")
    try:
        probs = final_model.predict_proba(df_external)[:, 1]
        predictions = (probs >= best_threshold).astype(int)
        output = pd.DataFrame({'id': ids, 'Depression': predictions})
        logger.info(f"Predictions completed.")
        logger.debug(f"Few first predictions: \n{output.head(3).to_string()}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

    logger.info("=== SAVE PREDICTIONS ===")
    output_path = config["outputs"]["predictions_output_path"]
    try:
        output.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {str(e)}")
        raise

    logger.info("*" * 80)


if __name__ == '__main__':
    from depression_classification.utils.settings import CFG_PATH

    main(CFG_PATH)
