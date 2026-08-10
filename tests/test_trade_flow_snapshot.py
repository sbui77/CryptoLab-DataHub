from __future__ import annotations
import pandas as pd
from cryptolab.features.trade_flow_snapshot import (
    build_trade_flow_snapshot,
    trade_flow_snapshot_to_dataframe,
)
def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
                "binance",
            ],
            "symbol": [
                "BTCUSDT",
                "BTCUSDT",
            ],
            "timeframe": [
                "1m",
                "1m",
            ],
            "open_time": pd.to_datetime(
                [
                    "2026-08-10T00:00:00Z",
                    "2026-08-10T00:01:00Z",
                ],
                utc=True,
            ),
            "close": [
                60000.0,
                60100.0,
            ],
            "trade_count": [
                100,
                120,
            ],
            "buy_trade_ratio": [
                0.55,
                0.60,
            ],
            "sell_trade_ratio": [
                0.45,
                0.40,
            ],
            "trade_count_imbalance": [
                0.10,
                0.20,
            ],
            "buy_quote_ratio": [
                0.56,
                0.62,
            ],
            "sell_quote_ratio": [
                0.44,
                0.38,
            ],
            "quote_delta": [
                10000.0,
                20000.0,
            ],
            "quote_delta_pct": [
                0.12,
                0.24,
            ],
            "base_delta": [
                0.2,
                0.4,
            ],
            "base_delta_pct": [
                0.10,
                0.20,
            ],
            "rolling_quote_imbalance_60": [
                0.05,
                0.08,
            ],
            "base_cvd": [
                1.0,
                1.4,
            ],
            "quote_cvd": [
                50000.0,
                70000.0,
            ],
            "daily_base_cvd": [
                1.0,
                1.4,
            ],
            "daily_quote_cvd": [
                50000.0,
                70000.0,
            ],
            "rolling_base_cvd_60": [
                1.0,
                1.4,
            ],
            "rolling_quote_cvd_60": [
                50000.0,
                70000.0,
            ],
            "rolling_base_cvd_1440": [
                1.0,
                1.4,
            ],
            "rolling_quote_cvd_1440": [
                50000.0,
                70000.0,
            ],
            "large_trade_quote_share": [
                0.20,
                0.30,
            ],
            "large_quote_delta": [
                5000.0,
                10000.0,
            ],
            "large_quote_delta_pct": [
                0.25,
                0.40,
            ],
            "trade_flow_regime": [
                "buy_dominant",
                "large_buy_pressure",
            ],
            "flow_regime_confidence": [
                0.75,
                1.0,
            ],
            "trade_flow_regime_valid": [
                True,
                True,
            ],
        }
    )
def test_snapshot_builds():
    snapshot = build_trade_flow_snapshot(
        make_df()
    )
    assert (
        snapshot.symbol
        == "BTCUSDT"
    )
    assert (
        snapshot.close
        == 60100.0
    )
def test_snapshot_latest_regime():
    snapshot = build_trade_flow_snapshot(
        make_df()
    )
    assert (
        snapshot.trade_flow_regime
        == "large_buy_pressure"
    )
    assert (
        snapshot.trade_flow_regime_valid
        is True
    )
def test_snapshot_latest_cvd():
    snapshot = build_trade_flow_snapshot(
        make_df()
    )
    assert (
        snapshot.quote_cvd
        == 70000.0
    )
    assert (
        snapshot.daily_quote_cvd
        == 70000.0
    )
def test_snapshot_dataframe():
    snapshot = build_trade_flow_snapshot(
        make_df()
    )
    result = (
        trade_flow_snapshot_to_dataframe(
            snapshot
        )
    )
    assert len(result) == 1
    assert (
        "trade_flow_regime"
        in result.columns
    )
    assert (
        result.loc[
            0,
            "trade_flow_regime",
        ]
        == "large_buy_pressure"
    )
