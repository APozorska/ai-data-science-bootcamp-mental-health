from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             classification_report,
                             roc_auc_score,
                             confusion_matrix)


def evaluate_model(model, X_test, y_test, logger):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None

    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Classification report:\n{report}")
    logger.info(f"F1-score (macro): {f1_macro:.4f}")
    logger.info(f"F1-score (weighted): {f1_weighted:.4f}")
    if roc_auc is not None:
        logger.info(f"ROC-AUC score: {roc_auc:.4f}")
    logger.info(f"Confusion matrix:\n{cm}")

    metrics = {
        "classification_report": report,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }
    return metrics
