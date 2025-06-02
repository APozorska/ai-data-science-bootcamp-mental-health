from sklearn.model_selection import train_test_split
from depression_classification.utils.logger import get_logger
from depression_classification.data.loader import load_csv
from depression_classification.preprocessing.data_checks import handle_duplicates, remove_unnecessary_columns
from depression_classification.utils.config import load_task_config, get_target_col

logger = get_logger("PREPARE AND SPLIT DATA", log_level="DEBUG")


def main(config_path: str):

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

    logger.info("=== LOAD TARGET FROM CONFIGURATION ===")
    target_col = get_target_col(config)
    logger.info(f"Target loaded: {target_col}")

    # Checking duplicates
    logger.info("=== DATA CHECKS: CHECK FOR DUPLICATES ===")
    df = handle_duplicates(df, logger)

    # Removing unnecessary columns
    logger.info("=== DATA CHECKS: REMOVE UNNECESSARY COLUMNS ===")
    df = remove_unnecessary_columns(df, logger, ["name", "id"])
    logger.debug(f"First rows of the data: \n{df.head(3).to_string()}")

    # Split to X, y
    logger.info("=== SPLITTING THE DATA TO X, y ===")
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    logger.info("Data splitted to X,y correctly.")
    logger.debug(f"Class distribution:\n {y.value_counts()}")

    # Split train and test
    logger.info("=== SPLITTING THE DATA TO X_train, X_test, y_train, y_test ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=config["data_split"]["test_size"],
                                                        random_state=config["data_split"]["random_state"],
                                                        stratify=y)
    logger.info(f"Data splitted to X_train, y_train, X_test, y_test correctly.\n"
                f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}\n"
                f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.debug(f"\nClass distribution in training set: {y_train.value_counts()}\n"
                 f"Class distribution in validation set: {y_test.value_counts()}")

    # Save train and test files
    train = X_train.copy()
    train[target_col] = y_train
    train_path = config["data"]["train_path"]
    train.to_csv(train_path, index=False)

    test = X_test.copy()
    test[target_col] = y_test
    test_path = config["data"]["test_path"]
    test.to_csv(test_path, index=False)

    logger.info(f"Data prepared and splitted. Files saved to {train_path} and {test_path}")


if __name__ == '__main__':
    from depression_classification.utils.settings import CFG_PATH

    main(CFG_PATH)
