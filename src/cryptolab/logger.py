from __future__ import annotations

import logging
from pathlib import Path

from cryptolab.config import (
    get_config_value,
    load_config,
    resolve_project_path,
)


_LOGGING_CONFIGURED = False


def setup_logging() -> None:
    """
    Configure CryptoLab application logging.

    Logging configuration is loaded from config/config.yaml.
    """

    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    config = load_config()

    level_name = str(
        get_config_value(
            config,
            "logging.level",
            "INFO",
        )
    ).upper()

    level = getattr(
        logging,
        level_name,
        logging.INFO,
    )

    log_format = get_config_value(
        config,
        "logging.format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    handlers: list[logging.Handler] = []

    if get_config_value(
        config,
        "logging.console",
        True,
    ):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(log_format)
        )
        handlers.append(console_handler)

    if get_config_value(
        config,
        "logging.file",
        True,
    ):
        log_file_value = get_config_value(
            config,
            "logging.log_file",
            "logs/cryptolab.log",
        )

        log_file = resolve_project_path(
            log_file_value
        )

        Path(log_file).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )

        file_handler.setFormatter(
            logging.Formatter(log_format)
        )

        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.
    """

    setup_logging()

    return logging.getLogger(name)
