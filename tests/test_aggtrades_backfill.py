from __future__ import annotations

import pandas as pd

import cryptolab.pipelines.aggtrades_backfill as backfill


def make_page(
    first_id: int,
    count: int,
    start_time: str,
) -> pd.DataFrame:
    timestamp = pd.date_range(
        start_time,
        periods=count,
        freq="1s",
    )

    ids = list(
        range(
            first_id,
            first_id + count,
        )
    )

    return pd.DataFrame(
        {
            "exchange": [
                "binance"
            ] * count,

            "symbol": [
                "BTCUSDT"
            ] * count,

            "agg_trade_id": ids,

            "price": [
                60_000.0
            ] * count,

            "quantity": [
                0.01
            ] * count,

            "quote_quantity": [
                600.0
            ] * count,

            "first_trade_id": ids,

            "last_trade_id": ids,

            "trade_time": timestamp,

            "buyer_is_maker": [
                False
            ] * count,

            "taker_side": [
                "buy"
            ] * count,
        }
    )


def test_backfill_forward_pagination(
    monkeypatch,
):
    seed = make_page(
        first_id=100,
        count=2,
        start_time=(
            "2026-08-08T17:15:00Z"
        ),
    )

    page_1 = make_page(
        first_id=100,
        count=3,
        start_time=(
            "2026-08-08T17:15:00Z"
        ),
    )

    page_2 = make_page(
        first_id=103,
        count=3,
        start_time=(
            "2026-08-08T17:15:03Z"
        ),
    )

    saved = []

    monkeypatch.setattr(
        backfill,
        "fetch_aggtrades_by_time",
        lambda **kwargs: seed,
    )

    def fake_from_id(
        symbol,
        from_id,
        limit,
    ):
        if from_id == 100:
            return page_1

        if from_id == 103:
            return page_2

        raise AssertionError(
            f"Unexpected from_id={from_id}"
        )

    monkeypatch.setattr(
        backfill,
        "fetch_aggtrades_from_id",
        fake_from_id,
    )

    monkeypatch.setattr(
        backfill,
        "save_aggtrades",
        lambda df: saved.append(
            df.copy()
        ),
    )

    result = backfill.backfill_aggtrades(
        symbol="BTCUSDT",
        start_time=(
            "2026-08-08T17:15:00Z"
        ),
        end_time=(
            "2026-08-08T17:15:04Z"
        ),
        page_limit=3,
        pause_seconds=0,
    )

    assert result.seed_id == 100

    assert (
        result.crossed_end_time
        is True
    )

    assert (
        result.complete
        is True
    )

    combined = pd.concat(
        saved,
        ignore_index=True,
    )

    assert (
        combined[
            "agg_trade_id"
        ].tolist()
        == [
            100,
            101,
            102,
            103,
            104,
        ]
    )


def test_invalid_time_range():
    try:
        backfill.backfill_aggtrades(
            symbol="BTCUSDT",
            start_time=(
                "2026-08-10T10:00:00Z"
            ),
            end_time=(
                "2026-08-10T09:00:00Z"
            ),
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
