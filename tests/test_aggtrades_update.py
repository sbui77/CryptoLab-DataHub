from __future__ import annotations

import pandas as pd
import pytest

import cryptolab.pipelines.aggtrades_update as updater


def make_page(
    first_id: int,
    count: int,
) -> pd.DataFrame:
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
                60000.0
            ] * count,

            "quantity": [
                0.01
            ] * count,

            "quote_quantity": [
                600.0
            ] * count,

            "first_trade_id": [
                100000 + value
                for value in ids
            ],

            "last_trade_id": [
                100000 + value
                for value in ids
            ],

            "trade_time": pd.date_range(
                "2026-08-10T00:00:00Z",
                periods=count,
                freq="1s",
            ),

            "buyer_is_maker": [
                False
            ] * count,

            "taker_side": [
                "buy"
            ] * count,
        }
    )


def test_incremental_single_page(
    monkeypatch,
):
    saved_pages = []

    monkeypatch.setattr(
        updater,
        "get_latest_agg_trade_id",
        lambda exchange, symbol: 99,
    )

    monkeypatch.setattr(
        updater,
        "_get_remote_latest_id",
        lambda symbol: 109,
    )

    monkeypatch.setattr(
        updater,
        "fetch_aggtrades_from_id",
        lambda symbol, from_id, limit:
        make_page(
            from_id,
            limit,
        ),
    )

    monkeypatch.setattr(
        updater,
        "save_aggtrades",
        lambda df:
        saved_pages.append(
            df.copy()
        ),
    )

    result = updater.update_aggtrades(
        symbol="BTCUSDT",
        page_limit=1000,
        pause_seconds=0,
    )

    assert result.start_id == 100
    assert result.target_id == 109

    assert result.rows_fetched == 10
    assert result.pages_fetched == 1

    assert result.first_fetched_id == 100
    assert result.last_fetched_id == 109

    assert result.up_to_date is True

    assert len(saved_pages) == 1


def test_incremental_multiple_pages(
    monkeypatch,
):
    saved_pages = []

    monkeypatch.setattr(
        updater,
        "get_latest_agg_trade_id",
        lambda exchange, symbol: 99,
    )

    monkeypatch.setattr(
        updater,
        "_get_remote_latest_id",
        lambda symbol: 2599,
    )

    monkeypatch.setattr(
        updater,
        "fetch_aggtrades_from_id",
        lambda symbol, from_id, limit:
        make_page(
            from_id,
            limit,
        ),
    )

    monkeypatch.setattr(
        updater,
        "save_aggtrades",
        lambda df:
        saved_pages.append(
            df.copy()
        ),
    )

    result = updater.update_aggtrades(
        symbol="BTCUSDT",
        page_limit=1000,
        pause_seconds=0,
    )

    assert result.rows_fetched == 2500
    assert result.pages_fetched == 3

    assert result.first_fetched_id == 100
    assert result.last_fetched_id == 2599

    assert result.up_to_date is True

    assert [
        len(page)
        for page in saved_pages
    ] == [
        1000,
        1000,
        500,
    ]


def test_already_up_to_date(
    monkeypatch,
):
    monkeypatch.setattr(
        updater,
        "get_latest_agg_trade_id",
        lambda exchange, symbol: 100,
    )

    monkeypatch.setattr(
        updater,
        "_get_remote_latest_id",
        lambda symbol: 100,
    )

    result = updater.update_aggtrades(
        symbol="BTCUSDT",
        pause_seconds=0,
    )

    assert result.rows_fetched == 0
    assert result.pages_fetched == 0
    assert result.up_to_date is True


def test_empty_local_storage(
    monkeypatch,
):
    monkeypatch.setattr(
        updater,
        "get_latest_agg_trade_id",
        lambda exchange, symbol: None,
    )

    with pytest.raises(
        updater.AggTradesUpdateError
    ):
        updater.update_aggtrades(
            symbol="BTCUSDT",
            pause_seconds=0,
        )


def test_bad_pagination_start(
    monkeypatch,
):
    monkeypatch.setattr(
        updater,
        "get_latest_agg_trade_id",
        lambda exchange, symbol: 99,
    )

    monkeypatch.setattr(
        updater,
        "_get_remote_latest_id",
        lambda symbol: 109,
    )

    monkeypatch.setattr(
        updater,
        "fetch_aggtrades_from_id",
        lambda symbol, from_id, limit:
        make_page(
            from_id + 1,
            limit,
        ),
    )

    monkeypatch.setattr(
        updater,
        "save_aggtrades",
        lambda df: None,
    )

    with pytest.raises(
        updater.AggTradesUpdateError
    ):
        updater.update_aggtrades(
            symbol="BTCUSDT",
            pause_seconds=0,
        )


def test_max_pages_safety(
    monkeypatch,
):
    monkeypatch.setattr(
        updater,
        "get_latest_agg_trade_id",
        lambda exchange, symbol: 99,
    )

    monkeypatch.setattr(
        updater,
        "_get_remote_latest_id",
        lambda symbol: 5000,
    )

    monkeypatch.setattr(
        updater,
        "fetch_aggtrades_from_id",
        lambda symbol, from_id, limit:
        make_page(
            from_id,
            limit,
        ),
    )

    monkeypatch.setattr(
        updater,
        "save_aggtrades",
        lambda df: None,
    )

    with pytest.raises(
        updater.AggTradesUpdateError
    ):
        updater.update_aggtrades(
            symbol="BTCUSDT",
            page_limit=1000,
            max_pages=1,
            pause_seconds=0,
        )
