import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def configure_root_logger(level: int = logging.INFO) -> None:
    log_dir = os.environ.get("LOG_DIR", ".")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.environ.get("LOG_FILE", os.path.join(log_dir, "runpod_forecast_logs.log"))
    root_logger = logging.getLogger()
    # Remove all existing handlers so we start fresh each run.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Rotating file handler
    file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=10_000_000, backupCount=3)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Create a console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    root_logger.setLevel(level)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # If this logger doesn't have its own handlers, it will use the root logger's.
    if not logger.handlers:
        logger.setLevel(logging.INFO)
    logger.propagate = True
    return logger
