from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the CryptoLab configuration is invalid."""


def load_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load CryptoLab configuration from YAML.

    Parameters
    ----------
    config_path:
        Optional path to a YAML configuration file.
        If omitted, config/config.yaml is used.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    ConfigError
        If the configuration file is missing, empty,
        or cannot be parsed.
    """

    path = (
        Path(config_path).expanduser().resolve()
        if config_path
        else DEFAULT_CONFIG_PATH
    )

    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML configuration: {path}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigError(
            f"Configuration must be a YAML mapping: {path}"
        )

    return config


def get_config_value(
    config: dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a nested config value using dot notation.

    Example
    -------
    get_config_value(config, "binance.spot.base_url")
    """

    value: Any = config

    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Resolve a configured project-relative path.

    Absolute paths are returned unchanged.
    Relative paths are resolved against PROJECT_ROOT.
    """

    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_directories(config: dict[str, Any]) -> None:
    """
    Create directories required by CryptoLab.
    """

    path_keys = [
        "raw",
        "curated",
        "features",
        "outputs",
        "logs",
    ]

    paths = config.get("paths", {})

    for key in path_keys:
        value = paths.get(key)

        if value:
            resolve_project_path(value).mkdir(
                parents=True,
                exist_ok=True,
            )
