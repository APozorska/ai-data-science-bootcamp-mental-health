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


if __name__ == '__main__':
    logger = get_logger(__name__, log_level="CRITICAL")
    logger.info("Hello.")
    logger.error("Wrong name.")
    logger.debug("Debugging the size...")
    logger.critical("Critical! The program is not working as expected.")
