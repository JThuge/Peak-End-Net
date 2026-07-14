"""
Unified logging configuration module.
All scripts use setup_logger() to get a logger that outputs to both console and log file.
Log files are stored in logs/ directory, named by script name and timestamp.
"""
import os
import logging
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(script_name, log_dir=None, level=logging.INFO):
    """
    Create and configure logger, outputting to both console and log file.

    Args:
        script_name: script name, used for log file naming and logger naming
        log_dir: log file directory, defaults to logs/
        level: log level, defaults to INFO

    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(script_name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers when setup_logger is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Log file: {log_filepath}")

    return logger
