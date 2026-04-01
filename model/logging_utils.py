import logging
import os
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> logging.Logger:
    """
    Configure the root logger with console and file handlers.
    """
    desired_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, desired_level, logging.INFO)
    root_logger = logging.getLogger()

    if getattr(configure_logging, "_configured", False):
        root_logger.setLevel(numeric_level)
        return root_logger

    root_logger.setLevel(numeric_level)
    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    configure_logging._configured = True  # type: ignore[attr-defined]

    root_logger.debug("Logging configured. Level=%s File=%s", desired_level, LOG_FILE)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


__all__ = ["LOG_FILE", "configure_logging", "get_logger"]
