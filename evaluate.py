import json
import joblib

from sklearn.metrics import classification_report

from depression_classification.utils.config import load_task_config, get_target_col
from depression_classification.utils.logger import get_logger
from depression_classification.data.loader import load_csv
from depression_classification.models.evaluation import evaluate_model


logger = get_logger("EVALUATE ON TEST SET", log_level="INFO")


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

    logger.info("=== LOAD TARGET FROM CONFIGURATION ===")
    target_col = get_target_col(config)
    logger.info(f"Target loaded: {target_col}")

    logger.info("=== LOAD TEST DATA | DATA SPLITTED WHILE TRAINING ===")
    data_path = config["data"]["data_splitted_test_path"]
    try:
        test_data = load_csv(data_path)
        logger.info(f"Data loaded:{test_data.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    # Split to X_test, y_test
    X_test = test_data.drop([target_col], axis=1)
    y_test = test_data[target_col]
    logger.info("X_test, y_test loaded.")

    # Load best model
    logger.info("=== LOAD BEST MODEL ===")
    metadata_path = config["outputs"]["final_metadata_output_path"]
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    model_path = metadata["model_path"]
    best_threshold = metadata["best_threshold"]

    final_model = joblib.load(model_path)
    logger.info("Best model loaded.")

    # FINAL evaluation on test set
    logger.info("=== EVALUATE PREDICTIONS ===")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else None
    y_pred_threshold = (y_proba >= best_threshold).astype(int)

    report_default = classification_report(y_test, y_pred)
    logger.debug(f"Classification report with default threshold:\n{report_default}")

    report_tuned = classification_report(y_test, y_pred_threshold)
    logger.debug(f"Classification report with tuned threshold (={best_threshold}):\n{report_tuned}")

    metrics_default = evaluate_model(y_test, y_pred, y_proba)
    metrics_tuned = evaluate_model(y_test, y_pred_threshold, y_proba)

    logger.info(f"Default threshold metrics (0.5): {metrics_default}")
    logger.info(f"Tuned threshold metrics ({best_threshold:.3f}): {metrics_tuned}")

    # Save metrics
    final_metadata = {
        "model_name": metadata["model_name"],
        "score": metadata['score'],
        "selection_metric": metadata["selection_metric"],
        "params": metadata["params"],
        "metrics_default": metrics_default,
        "metrics_tuned": metrics_tuned,
        "model_path": metadata["model_path"]
    }
    final_evaluation_output_path = config["outputs"]["final_evaluation_output_path"]
    with open(final_evaluation_output_path, "w") as f:
        json.dump(final_metadata, f, indent=4)
    logger.info(f"Metrics saved to: {final_evaluation_output_path}")

    logger.info("*" * 80)


if __name__ == '__main__':
    from depression_classification.utils.settings import CFG_PATH

    main(CFG_PATH)
