import joblib
import pandas as pd
import numpy as np
import time
import json

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_curve
from sklearn.base import clone

from depression_classification.data.loader import load_csv
from depression_classification.utils.config import load_task_config, get_feature_groups, get_target_col
from depression_classification.utils.logger import get_logger
from depression_classification.utils.json_utils import make_json_serializable

from depression_classification.preprocessing.data_checks import handle_duplicates, remove_unnecessary_columns
from depression_classification.preprocessing.preprocessor_flagger import get_conditional_flagger, get_inconsistency_flagger
from depression_classification.preprocessing.preprocessor_custom import get_custom_preprocessor
from depression_classification.preprocessing.preprocessor_standard import get_standard_preprocessor

from depression_classification.pipeline.pipeline import get_pipeline
from depression_classification.models.models import get_model
from depression_classification.models.param_grid import build_param_grid
from depression_classification.models.evaluation import evaluate_model

logger = get_logger("TRAIN AND TUNE", log_level="DEBUG")


def main(config_path: str):
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
    logger.info("=== LOAD DATA FROM CONFIGURATION ===")
    columns = config["data"]["columns"]
    data_path = config["data"]["raw_data_path"]
    try:
        df = load_csv(data_path, columns)
        logger.info(f"Data loaded:{df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Shape: {df.shape}")

    logger.debug(f"First rows of the data:\n{df.head(3).to_string()}")
    logger.debug(f"Missing values:\n{df.isnull().sum()}")
    logger.debug(f"Dtypes: \n{df.dtypes}")
    logger.debug(f"Describe:\n{df.describe().to_string()}")

    logger.info("=== LOAD FEATURE GROUPS AND TARGET FROM CONFIGURATION ===")
    feature_groups = get_feature_groups(config, logger)
    cat_binary_features = feature_groups["cat_binary"]
    cat_multiclass_features = feature_groups["cat_multiclass"]
    cat_highcard_features = feature_groups["cat_highcard"]
    num_standard_features = feature_groups["num_standard"]
    num_custom_features = feature_groups["num_custom"]
    target_col = get_target_col(config)
    logger.info(f"Target loaded: {target_col}")

    # DATA CHECKS: Checking duplicates and remove unnecessary columns
    logger.info("=== DATA CHECKS: CHECK FOR DUPLICATES ===")
    df = handle_duplicates(df, logger)

    logger.info("=== DATA CHECKS: REMOVE UNNECESSARY COLUMNS ===")
    df = remove_unnecessary_columns(df, logger, ["name", "id"])

    # Split to X, y
    logger.info("=== SPLITTING THE DATA ===")
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    logger.info("Data splitted to X,y correctly.")
    logger.debug(f"Class distribution:\n {y.value_counts()}")

    # Split train and test
    X_full_train, X_test, y_full_train, y_test = train_test_split(X, y,
                                                                  test_size=config["data_split"]["test_size"],
                                                                  random_state=config["data_split"]["random_state"],
                                                                  stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train,
                                                      test_size=config["data_split"]["test_size"],
                                                      random_state=config["data_split"]["random_state"],
                                                      stratify=y_full_train)
    logger.info(f"Data splitted to X_train, y_train, X_val, y_val, X_test, y_test correctly.\n"
                f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}\n"
                f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}\n"
                f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.debug(f"Class distribution \nin training set: {y_train.value_counts()}"
                 f" and in validation set: {y_val.value_counts()}"
                 f"\nand in test set: {y_test.value_counts()}")

    # Save train and test files
    train = pd.concat([X_train, y_train], axis=1)
    train_path = config["data"]["data_splitted_train_path"]
    train.to_csv(train_path, index=False)

    val = pd.concat([X_val, y_val], axis=1)
    val_path = config["data"]["data_splitted_val_path"]
    val.to_csv(val_path, index=False)

    test = pd.concat([X_test, y_test], axis=1)
    test_path = config["data"]["data_splitted_test_path"]
    test.to_csv(test_path, index=False)
    logger.info(f"Files of splitted data to train, val and test sets saved to {train_path}, {val_path} and {test_path}")

    # Load preprocessors and feature_selector
    logger.info("=== LOAD PREPROCESSORS AND FEATURE SELECTOR ===")
    inconsistency_flagger_preprocessor = get_inconsistency_flagger()
    flagging_map = config["flagging_map"]
    conditional_flagger_preprocessor = get_conditional_flagger(flagging_map)
    custom_preprocessor = get_custom_preprocessor()
    standard_preprocessor = get_standard_preprocessor(num_standard_features, num_custom_features,
                                                      cat_multiclass_features, cat_highcard_features, cat_binary_features)
    feature_selector = "feature_selector", "passthrough"
    logger.info(f"Preprocessing and feature selection steps loaded.")

    # Create pipeline
    models_to_test = config["models"]["to_test"]
    cv = StratifiedKFold(
        n_splits=config["cross_validation"]["n_splits"],
        shuffle=config["cross_validation"]["shuffle"],
        random_state=config["cross_validation"]["random_state"])
    primary_scoring = config["models"]["scoring"]["primary"]
    cv_metrics = {}

    for model_name in models_to_test:
        logger.info(f"=== HYPERPARAMETER TUNING OF {model_name} ===")

        params = config["models"][model_name]["params"]
        model = get_model(model_name, params)

        pipeline = get_pipeline([
            inconsistency_flagger_preprocessor,
            conditional_flagger_preprocessor,
            custom_preprocessor,
            standard_preprocessor,
            feature_selector,
            model
        ])
        logger.info(f"Pipeline for {model_name} created successfully")
        logger.debug(f"Steps: {({name: type(step).__name__ for name, step in pipeline.named_steps.items()})}")

        param_grid = build_param_grid(config, model_name=model_name)
        logger.info(f"Param_grid specified.")
        logger.debug(f"Param_grid details:\n{param_grid}")

        logger.info(f"Train and optimize {model_name}...")
        start = time.time()
        try:
            optimizer = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=primary_scoring,
                cv=cv,
                verbose=2)
            optimizer.fit(X_train, y_train)
            best_estimator = optimizer.best_estimator_
            best_params = optimizer.best_params_
            best_score = optimizer.best_score_
            logger.info(f"GridSearchCV for {model_name} completed in {time.time() - start:.2f} seconds.")
            logger.info(f"Best model includes feature selection: {isinstance(best_estimator.named_steps['feature_selector'], SelectFromModel)}")
            logger.info(f"Best params: {best_params}\n"
                        f"Best score: {best_score}")
        except Exception as e:
            logger.error(f"Error during model training/optimization: {str(e)}")
            raise

        # Save model
        model_output_path = config["models"][model_name]["model_output_path"]
        joblib.dump(best_estimator, model_output_path)
        logger.info(f"Model {model_name} saved to: {model_output_path}")

        logger.info(f"=== EVALUATION {model_name} ON VALIDATION DATA ===")
        # Default threshold
        y_val_pred_default = best_estimator.predict(X_val)
        y_val_proba = best_estimator.predict_proba(X_val)[:, 1]

        default_report = classification_report(y_val, y_val_pred_default)
        logger.debug(f"Classification report with default threshold: \n{default_report}")

        metrics_default = evaluate_model(y_val, y_val_pred_default, y_proba=y_val_proba)
        logger.info(f"Evaluation metrics for {model_name} with default threshold:")
        logger.info(f"F1-score (weighted): {metrics_default['f1_weighted']:.4f}")
        logger.info(f"Recall: {metrics_default['recall']:.4f}")
        if metrics_default['roc_auc'] is not None:
            logger.info(f"ROC-AUC score: {metrics_default['roc_auc']:.4f}")
        logger.info(f"Confusion matrix: {metrics_default['confusion_matrix']}")

        # Threshold tuning (e.g., maximize recall or F1)
        fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        # Predict with tuned threshold
        y_val_pred_tuned = (y_val_proba >= best_threshold).astype(int)
        tuned_report = classification_report(y_val, y_val_pred_tuned)
        logger.debug(f"Classification report after threshold tuning:\n{tuned_report}")

        metrics_tuned = evaluate_model(y_val, y_val_pred_tuned, y_proba=y_val_proba)
        logger.info(f"Evaluation metrics for {model_name} with tuned threshold (={best_threshold:.2f}):")
        logger.info(f"F1-score (weighted): {metrics_tuned['f1_weighted']:.4f}")
        logger.info(f"Recall: {metrics_tuned['recall']:.4f}")
        if metrics_tuned['roc_auc'] is not None:
            logger.info(f"ROC-AUC score: {metrics_tuned['roc_auc']:.4f}")
        logger.info(f"Confusion matrix: {metrics_tuned['confusion_matrix']}")

        # Save model summary
        best_params_serializable = make_json_serializable(best_params)
        metadata = {
            "model_name": model_name,
            "model_path": model_output_path,
            "best_score": best_score,
            "best_params": best_params_serializable,
            "metrics_default": metrics_default,
            "metrics_tuned": metrics_tuned,
            "best_threshold": float(best_threshold),
        }
        metadata_path = config["models"][model_name]["metadata_output_path"]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata {model_name} saved to: {metadata_path}")

        cv_metrics[model_name] = {
            # GridSearchCV results
            "best_score": best_score,
            "best_params": best_params,
            # Default threshold metrics
            "val_recall_default": metrics_default["recall"],
            "val_precision_default": metrics_default["precision"],
            "val_f1_weighted_default": metrics_default["f1_weighted"],
            "val_roc_auc_default": metrics_default["roc_auc"],
            # Tuned threshold metrics
            "val_recall_tuned": metrics_tuned["recall"],
            "val_precision_tuned": metrics_tuned["precision"],
            "val_f1_weighted_tuned": metrics_tuned["f1_weighted"],
            "val_roc_auc_tuned": metrics_tuned["roc_auc"],
            # Threshold info
            "best_threshold": best_threshold
        }

    # Save evaluation metrics
    cv_metrics_df = pd.DataFrame(cv_metrics).T
    cv_metrics_path = config["outputs"]["cv_metrics_output_path"]
    cv_metrics_df.to_csv(cv_metrics_path)
    logger.info(f"Metrics for all models saved to: {cv_metrics_path}")

    # Select best model
    logger.info("=== FINAL MODEL SELECTION ===")
    final_selection_metric = "val_f1_weighted_tuned"
    best_model_name = cv_metrics_df[final_selection_metric].idxmax()
    best_row = cv_metrics_df.loc[best_model_name]
    logger.info(f"Selected best model: {best_model_name}\n Summary: \n{best_row}")

    # Retrain on full train+val data
    best_model_path = config["models"][best_model_name]["model_output_path"]
    best_estimator = joblib.load(best_model_path)
    final_model = clone(best_estimator)
    final_model.fit(X_full_train, y_full_train)
    logger.info("Final model trained on X_full_train, y_full_train.")

    # Save retrained final model
    final_model_path = config["outputs"]["final_model_output_path"]
    joblib.dump(final_model, final_model_path)
    logger.info(f"Trained final model saved to: {final_model_path}")

    # Save model selection summary (name, score, params, output_path)
    final_metadata_output_path = config["outputs"]["final_metadata_output_path"]
    final_params_serializable = make_json_serializable(best_row["best_params"])
    final_metadata = {
        "model_name": best_model_name,
        "model_path": final_model_path,
        "score": best_row["best_score"],
        "selection_metric": primary_scoring,
        "params": final_params_serializable
    }
    with open(final_metadata_output_path, "w") as f:
        json.dump(final_metadata, f, indent=4)
    logger.info(f"Metrics saved to: {final_metadata_output_path}")

    logger.info("*" * 80)


if __name__ == '__main__':
    from depression_classification.utils.settings import CFG_PATH

    main(CFG_PATH)
