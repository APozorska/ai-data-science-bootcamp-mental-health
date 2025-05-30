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


def add_inconsistency_flags(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Add single inconsistency flags based on logical conditions and log the process.
    Args:
        df (pd.DataFrame): Input DataFrame.
        logger (logging.Logger): Logger object for logging info.
    Returns:
        pd.DataFrame: DataFrame with an 'inconsistency_flag' column.
    """
    df['inconsistency_flag'] = (
            ((df['work_pressure'] > 0) & (df['academic_pressure'] > 0)).astype(int) +
            ((df['study_satisfaction'] > 0) & (df['job_satisfaction'] > 0)).astype(int) +
            (
                    ((df['occupation_status'] == 'Working Professional') & (df['profession'] == 'Student')) |
                    ((df['occupation_status'] == 'Student') & (df['profession'] != 'Student') & (df['profession'].notna()))
            ).astype(int)
    )

    n_inconsistent = df['inconsistency_flag'].sum()
    logger.info(f"Inconsistency flags count: {n_inconsistent}.")
    logger.debug(
        "Observations with inconsistency flag:\n"
        f"{df[df.inconsistency_flag > 0].to_string()}"
    )
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
