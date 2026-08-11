from __future__ import annotations

import pandas as pd

import cryptolab.pipelines.aggtrades_storage as agg_storage
import cryptolab.pipelines.open_interest_storage as oi_storage
import cryptolab.pipelines.funding_rate_storage as funding_storage
import cryptolab.pipelines.basis_storage as basis_storage
import cryptolab.pipelines.taker_flow_storage as taker_storage


RUNTIME_INGESTED_AT = pd.Timestamp(
    "2026-08-11T12:00:00Z"
)


def _runtime_metadata(
    event_time: object,
) -> dict[str, object]:
    timestamp = pd.Timestamp(
        event_time
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return {
        "available_at": timestamp,
        "available_at_quality": "derived",
        "ingested_at": RUNTIME_INGESTED_AT,
        "ingested_at_quality": "exact",
    }


# ============================================================
# aggTrades
# ============================================================


def _legacy_aggtrade() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "agg_trade_id": 100,
                "price": 120000.0,
                "quantity": 0.10,
                "first_trade_id": 200,
                "last_trade_id": 200,
                "trade_time": pd.Timestamp(
                    "2026-08-11T00:00:00Z"
                ),
                "buyer_is_maker": False,
            }
        ]
    )


def _runtime_aggtrade() -> pd.DataFrame:
    event_time = pd.Timestamp(
        "2026-08-11T00:00:00Z"
    )

    row = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "agg_trade_id": 100,
        "price": 120001.0,
        "quantity": 0.11,
        "first_trade_id": 200,
        "last_trade_id": 200,
        "trade_time": event_time,
        "buyer_is_maker": False,
    }

    row.update(
        _runtime_metadata(
            event_time
        )
    )

    return pd.DataFrame(
        [row]
    )


def test_aggtrades_legacy_runtime_merge(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        agg_storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    agg_storage.save_aggtrades(
        _legacy_aggtrade()
    )

    legacy = agg_storage.read_aggtrades(
        "binance",
        "BTCUSDT",
    )

    assert len(legacy) == 1

    assert (
        legacy.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )

    assert pd.isna(
        legacy.iloc[0][
            "ingested_at"
        ]
    )

    agg_storage.save_aggtrades(
        _runtime_aggtrade()
    )

    merged = agg_storage.read_aggtrades(
        "binance",
        "BTCUSDT",
    )

    assert len(merged) == 1

    assert (
        merged.iloc[0][
            "agg_trade_id"
        ]
        == 100
    )

    assert (
        merged.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        merged.iloc[0][
            "ingested_at"
        ]
        == RUNTIME_INGESTED_AT
    )

    assert (
        merged.iloc[0][
            "price"
        ]
        == 120001.0
    )


# ============================================================
# Open Interest
# ============================================================


def _legacy_oi() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": "BTCUSDT",
                "period": "5m",
                "timestamp": pd.Timestamp(
                    "2026-08-11T00:00:00Z"
                ),
                "open_interest_base": 80000.0,
                "open_interest_quote": 9600000000.0,
            }
        ]
    )


def _runtime_oi() -> pd.DataFrame:
    event_time = pd.Timestamp(
        "2026-08-11T00:00:00Z"
    )

    row = {
        "exchange": "binance",
        "market": "usdm_perpetual",
        "symbol": "BTCUSDT",
        "period": "5m",
        "timestamp": event_time,
        "open_interest_base": 80010.0,
        "open_interest_quote": 9601200000.0,
    }

    row.update(
        _runtime_metadata(
            event_time
        )
    )

    return pd.DataFrame(
        [row]
    )


def test_open_interest_legacy_runtime_merge(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        oi_storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    oi_storage.save_open_interest(
        _legacy_oi()
    )

    legacy = oi_storage.read_open_interest(
        "binance",
        "BTCUSDT",
        "5m",
    )

    assert len(legacy) == 1

    assert (
        legacy.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )

    oi_storage.save_open_interest(
        _runtime_oi()
    )

    merged = oi_storage.read_open_interest(
        "binance",
        "BTCUSDT",
        "5m",
    )

    assert len(merged) == 1

    assert (
        merged.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        merged.iloc[0][
            "ingested_at"
        ]
        == RUNTIME_INGESTED_AT
    )

    assert (
        merged.iloc[0][
            "open_interest_base"
        ]
        == 80010.0
    )


# ============================================================
# Funding
# ============================================================


def _legacy_funding() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": "BTCUSDT",
                "funding_time": pd.Timestamp(
                    "2026-08-11T00:00:00Z"
                ),
                "funding_rate": 0.0001,
                "mark_price": 120000.0,
                "rate_type": "Regular",
            }
        ]
    )


def _runtime_funding() -> pd.DataFrame:
    event_time = pd.Timestamp(
        "2026-08-11T00:00:00Z"
    )

    row = {
        "exchange": "binance",
        "market": "usdm_perpetual",
        "symbol": "BTCUSDT",
        "funding_time": event_time,
        "funding_rate": 0.0002,
        "mark_price": 120100.0,
        "rate_type": "Regular",
    }

    row.update(
        _runtime_metadata(
            event_time
        )
    )

    return pd.DataFrame(
        [row]
    )


def test_funding_legacy_runtime_merge(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        funding_storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    funding_storage.save_funding_rate(
        _legacy_funding()
    )

    legacy = funding_storage.read_funding_rate(
        "binance",
        "BTCUSDT",
    )

    assert len(legacy) == 1

    assert (
        legacy.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )

    funding_storage.save_funding_rate(
        _runtime_funding()
    )

    merged = funding_storage.read_funding_rate(
        "binance",
        "BTCUSDT",
    )

    assert len(merged) == 1

    assert (
        merged.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        merged.iloc[0][
            "ingested_at"
        ]
        == RUNTIME_INGESTED_AT
    )

    assert (
        merged.iloc[0][
            "funding_rate"
        ]
        == 0.0002
    )


# ============================================================
# Basis
# ============================================================


def _legacy_basis() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": "BTCUSDT",
                "contract_type": "PERPETUAL",
                "period": "5m",
                "timestamp": pd.Timestamp(
                    "2026-08-11T00:00:00Z"
                ),
                "index_price": 120000.0,
                "futures_price": 120010.0,
                "basis": 10.0,
                "basis_rate": 0.000083,
                "annualized_basis_rate": 0.0075,
            }
        ]
    )


def _runtime_basis() -> pd.DataFrame:
    event_time = pd.Timestamp(
        "2026-08-11T00:00:00Z"
    )

    row = {
        "exchange": "binance",
        "market": "usdm_perpetual",
        "symbol": "BTCUSDT",
        "contract_type": "PERPETUAL",
        "period": "5m",
        "timestamp": event_time,
        "index_price": 120001.0,
        "futures_price": 120012.0,
        "basis": 11.0,
        "basis_rate": 0.000091,
        "annualized_basis_rate": 0.0080,
    }

    row.update(
        _runtime_metadata(
            event_time
        )
    )

    return pd.DataFrame(
        [row]
    )


def test_basis_legacy_runtime_merge(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        basis_storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    basis_storage.save_basis(
        _legacy_basis()
    )

    legacy = basis_storage.read_basis(
        "binance",
        "BTCUSDT",
        "5m",
    )

    assert len(legacy) == 1

    assert (
        legacy.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )

    basis_storage.save_basis(
        _runtime_basis()
    )

    merged = basis_storage.read_basis(
        "binance",
        "BTCUSDT",
        "5m",
    )

    assert len(merged) == 1

    assert (
        merged.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        merged.iloc[0][
            "ingested_at"
        ]
        == RUNTIME_INGESTED_AT
    )

    assert (
        merged.iloc[0][
            "basis"
        ]
        == 11.0
    )


# ============================================================
# Taker Flow
# ============================================================


def _legacy_taker_flow() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange": "binance",
                "market": "usdm_perpetual",
                "symbol": "BTCUSDT",
                "period": "5m",
                "timestamp": pd.Timestamp(
                    "2026-08-11T00:00:00Z"
                ),
                "buy_volume": 1000.0,
                "sell_volume": 900.0,
                "buy_sell_ratio": 1.111111,
            }
        ]
    )


def _runtime_taker_flow() -> pd.DataFrame:
    event_time = pd.Timestamp(
        "2026-08-11T00:00:00Z"
    )

    row = {
        "exchange": "binance",
        "market": "usdm_perpetual",
        "symbol": "BTCUSDT",
        "period": "5m",
        "timestamp": event_time,
        "buy_volume": 1200.0,
        "sell_volume": 1000.0,
        "buy_sell_ratio": 1.2,
    }

    row.update(
        _runtime_metadata(
            event_time
        )
    )

    return pd.DataFrame(
        [row]
    )


def test_taker_flow_legacy_runtime_merge(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        taker_storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    taker_storage.save_taker_flow(
        _legacy_taker_flow()
    )

    legacy = taker_storage.read_taker_flow(
        "binance",
        "BTCUSDT",
        "5m",
    )

    assert len(legacy) == 1

    assert (
        legacy.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )

    taker_storage.save_taker_flow(
        _runtime_taker_flow()
    )

    merged = taker_storage.read_taker_flow(
        "binance",
        "BTCUSDT",
        "5m",
    )

    assert len(merged) == 1

    assert (
        merged.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        merged.iloc[0][
            "ingested_at"
        ]
        == RUNTIME_INGESTED_AT
    )

    assert (
        merged.iloc[0][
            "buy_volume"
        ]
        == 1200.0
    )
