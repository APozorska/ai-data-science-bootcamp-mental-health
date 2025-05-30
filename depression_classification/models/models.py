from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_model(model_name: str, params: dict):
    match model_name:
        case "logistic_regression":
            model = LogisticRegression(**params)
        case "random_forest":
            model = RandomForestClassifier(**params)
        case _:
            raise ValueError(f"Incorrect model name: {model_name}")
    return "model", model
