from cryptolab.config import (
    get_config_value,
    load_config,
)


def test_load_config():
    config = load_config()

    assert config["project"]["name"] == "CryptoLab_DataHub"


def test_default_symbol():
    config = load_config()

    symbol = get_config_value(
        config,
        "price.default_symbol",
    )

    assert symbol == "BTCUSDT"


def test_binance_base_url():
    config = load_config()

    base_url = get_config_value(
        config,
        "binance.spot.base_url",
    )

    assert base_url == "https://api.binance.com"


def test_missing_config_value():
    config = load_config()

    value = get_config_value(
        config,
        "does.not.exist",
        "fallback",
    )

    assert value == "fallback"
