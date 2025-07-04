import pandas as pd
from pathlib import Path


def load_csv(path: str | Path, columns: list[str] = None) -> pd.DataFrame:
    """
    Load CSV file with specified columns as names into Pandas Frame.
    Args:
        path (str | Path): The path to the CSV file.
        columns: list of columns names of the CSV file.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    if columns is not None:
        data = pd.read_csv(path, names=columns, header=0)
    else:
        data = pd.read_csv(path)
    return data
