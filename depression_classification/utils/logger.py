import logging


def get_logger(
    logger_name: str,
    log_level: str = "INFO",
    logger_format: str | None = None
) -> logging.Logger:
    if logger_format is None:
        logger_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=log_level,
        format=logger_format,
    )
    logger = logging.getLogger(logger_name)
    return logger
