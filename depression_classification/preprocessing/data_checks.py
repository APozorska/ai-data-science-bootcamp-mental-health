import pandas as pd


def handle_duplicates(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame and log the process.
    Args:
        df (pd.DataFrame): Input DataFrame.
        logger (logging.Logger): Logger object for logging info.
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    duplicates = df.duplicated().sum()
    logger.info(f"Number of duplicated rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info(f"Duplicates removed. New shape: {df.shape}")
    else:
        logger.info("No duplicates found.")
    return df


def remove_unnecessary_columns(df: pd.DataFrame, logger, columns_to_remove=None) -> pd.DataFrame:
    """
    Remove unnecessary columns from the DataFrame and log the process.
    Args:
        df (pd.DataFrame): Input DataFrame.
        logger (logging.Logger): Logger object for logging info.
        columns_to_remove (list, optional): List of columns to remove.
    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    df = df.drop(columns=columns_to_remove, errors='ignore')
    logger.info(f"Columns {columns_to_remove} removed.")
    return df
