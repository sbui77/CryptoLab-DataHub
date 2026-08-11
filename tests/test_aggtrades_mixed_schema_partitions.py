from __future__ import annotations

import pandas as pd

import cryptolab.pipelines.aggtrades_storage as storage


def _legacy_trade(
    agg_trade_id: int,
    trade_time: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange": [
                "binance",
            ],
            "symbol": [
                "BTCUSDT",
            ],
            "agg_trade_id": [
                agg_trade_id,
            ],
            "price": [
                120000.0,
            ],
            "quantity": [
                0.10,
            ],
            "first_trade_id": [
                agg_trade_id + 1000,
            ],
            "last_trade_id": [
                agg_trade_id + 1000,
            ],
            "trade_time": [
                pd.Timestamp(
                    trade_time
                ),
            ],
            "buyer_is_maker": [
                False,
            ],
        }
    )


def _runtime_trade(
    agg_trade_id: int,
    trade_time: str,
) -> pd.DataFrame:
    event_time = pd.Timestamp(
        trade_time
    )

    return pd.DataFrame(
        {
            "exchange": [
                "binance",
            ],
            "symbol": [
                "BTCUSDT",
            ],
            "agg_trade_id": [
                agg_trade_id,
            ],
            "price": [
                120100.0,
            ],
            "quantity": [
                0.20,
            ],
            "first_trade_id": [
                agg_trade_id + 1000,
            ],
            "last_trade_id": [
                agg_trade_id + 1000,
            ],
            "trade_time": [
                event_time,
            ],
            "buyer_is_maker": [
                True,
            ],
            "available_at": [
                event_time,
            ],
            "available_at_quality": [
                "derived",
            ],
            "ingested_at": [
                pd.Timestamp(
                    "2026-08-11T12:00:00Z"
                ),
            ],
            "ingested_at_quality": [
                "exact",
            ],
        }
    )


def test_read_mixed_schema_partitions(
    tmp_path,
    monkeypatch,
):
    """
    A legacy daily partition and a PIT-aware daily
    partition must be readable together.
    """

    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    legacy = _legacy_trade(
        agg_trade_id=100,
        trade_time="2026-08-10T23:59:59Z",
    )

    legacy_path = (
        tmp_path
        / "trade_flow"
        / "aggtrades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
        / "day=10"
        / "data.parquet"
    )

    legacy_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    legacy.to_parquet(
        legacy_path,
        index=False,
    )

    runtime = _runtime_trade(
        agg_trade_id=101,
        trade_time="2026-08-11T00:00:01Z",
    )

    runtime_path = (
        tmp_path
        / "trade_flow"
        / "aggtrades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
        / "day=11"
        / "data.parquet"
    )

    runtime_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime.to_parquet(
        runtime_path,
        index=False,
    )

    result = storage.read_aggtrades(
        "binance",
        "BTCUSDT",
    )

    assert len(result) == 2

    assert (
        result[
            "available_at"
        ]
        == result[
            "trade_time"
        ]
    ).all()

    legacy_result = result[
        result[
            "agg_trade_id"
        ]
        == 100
    ].iloc[0]

    assert (
        legacy_result[
            "available_at_quality"
        ]
        == "derived"
    )

    assert (
        legacy_result[
            "ingested_at_quality"
        ]
        == "unknown"
    )

    assert pd.isna(
        legacy_result[
            "ingested_at"
        ]
    )

    runtime_result = result[
        result[
            "agg_trade_id"
        ]
        == 101
    ].iloc[0]

    assert (
        runtime_result[
            "available_at_quality"
        ]
        == "derived"
    )

    assert (
        runtime_result[
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        runtime_result[
            "ingested_at"
        ]
        == pd.Timestamp(
            "2026-08-11T12:00:00Z"
        )
    )


def test_save_runtime_into_legacy_partition(
    tmp_path,
    monkeypatch,
):
    """
    Saving a runtime trade into an existing legacy daily
    partition must upgrade the partition safely.
    """

    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    legacy = _legacy_trade(
        agg_trade_id=200,
        trade_time="2026-08-11T00:00:00Z",
    )

    legacy_path = (
        tmp_path
        / "trade_flow"
        / "aggtrades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
        / "day=11"
        / "data.parquet"
    )

    legacy_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    legacy.to_parquet(
        legacy_path,
        index=False,
    )

    runtime = _runtime_trade(
        agg_trade_id=201,
        trade_time="2026-08-11T00:00:01Z",
    )

    storage.save_aggtrades(
        runtime
    )

    result = storage.read_aggtrades(
        "binance",
        "BTCUSDT",
    )

    assert len(result) == 2

    assert (
        result[
            "agg_trade_id"
        ]
        .tolist()
        == [
            200,
            201,
        ]
    )

    assert (
        result[
            "ingested_at_quality"
        ]
        .tolist()
        == [
            "unknown",
            "exact",
        ]
    )

    assert (
        result[
            "available_at"
        ]
        == result[
            "trade_time"
        ]
    ).all()


def test_runtime_duplicate_replaces_legacy_same_agg_trade_id(
    tmp_path,
    monkeypatch,
):
    """
    When the same agg_trade_id is refetched at runtime,
    the runtime observation must replace the legacy row.

    This establishes metadata precedence:

        runtime exact > legacy unknown
    """

    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    legacy = _legacy_trade(
        agg_trade_id=300,
        trade_time="2026-08-11T00:00:00Z",
    )

    storage.save_aggtrades(
        legacy
    )

    before = storage.read_aggtrades(
        "binance",
        "BTCUSDT",
    )

    assert len(before) == 1

    assert (
        before.iloc[0][
            "ingested_at_quality"
        ]
        == "unknown"
    )

    runtime = _runtime_trade(
        agg_trade_id=300,
        trade_time="2026-08-11T00:00:00Z",
    )

    storage.save_aggtrades(
        runtime
    )

    after = storage.read_aggtrades(
        "binance",
        "BTCUSDT",
    )

    assert len(after) == 1

    row = after.iloc[0]

    assert (
        row[
            "agg_trade_id"
        ]
        == 300
    )

    assert (
        row[
            "ingested_at_quality"
        ]
        == "exact"
    )

    assert (
        row[
            "ingested_at"
        ]
        == pd.Timestamp(
            "2026-08-11T12:00:00Z"
        )
    )

    assert (
        row[
            "price"
        ]
        == 120100.0
    )

    assert bool(
        row[
            "buyer_is_maker"
        ]
    ) is True


def test_cross_partition_global_agg_trade_id_dedup(
    tmp_path,
    monkeypatch,
):
    """
    Defensive test:

    Even if the same agg_trade_id somehow appears in
    different daily partitions, read_aggtrades must still
    return exactly one logical trade.
    """

    monkeypatch.setattr(
        storage,
        "_get_raw_root",
        lambda: tmp_path,
    )

    first = _legacy_trade(
        agg_trade_id=400,
        trade_time="2026-08-10T23:59:59Z",
    )

    first_path = (
        tmp_path
        / "trade_flow"
        / "aggtrades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
        / "day=10"
        / "data.parquet"
    )

    first_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first.to_parquet(
        first_path,
        index=False,
    )

    second = _runtime_trade(
        agg_trade_id=400,
        trade_time="2026-08-11T00:00:01Z",
    )

    second_path = (
        tmp_path
        / "trade_flow"
        / "aggtrades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "year=2026"
        / "month=08"
        / "day=11"
        / "data.parquet"
    )

    second_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    second.to_parquet(
        second_path,
        index=False,
    )

    result = storage.read_aggtrades(
        "binance",
        "BTCUSDT",
    )

    assert len(result) == 1

    assert (
        result.iloc[0][
            "agg_trade_id"
        ]
        == 400
    )

    assert (
        result.iloc[0][
            "ingested_at_quality"
        ]
        == "exact"
    )
