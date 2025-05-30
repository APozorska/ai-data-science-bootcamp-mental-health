from typing import Any
from sklearn.pipeline import Pipeline


def get_pipeline(steps: list[tuple[str, Any]]):
    return Pipeline(steps)

