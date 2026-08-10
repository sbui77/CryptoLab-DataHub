from cryptolab.pipelines.aggtrades_storage import (
    get_latest_agg_trade_id,
)
from cryptolab.pipelines.aggtrades_update import (
    update_aggtrades,
)


def main() -> None:
    exchange = "binance"
    symbol = "BTCUSDT"

    before_id = (
        get_latest_agg_trade_id(
            exchange=exchange,
            symbol=symbol,
        )
    )

    print()
    print("=" * 80)
    print("BINANCE aggTrades INCREMENTAL UPDATE")
    print("=" * 80)

    print(
        "Latest local ID before:",
        before_id,
    )

    result = update_aggtrades(
        exchange=exchange,
        symbol=symbol,
        page_limit=1000,
        pause_seconds=0.05,
    )

    after_id = (
        get_latest_agg_trade_id(
            exchange=exchange,
            symbol=symbol,
        )
    )

    print()
    print("UPDATE RESULT")
    print("-" * 80)

    print(
        "Start ID        :",
        result.start_id,
    )

    print(
        "Target ID       :",
        result.target_id,
    )

    print(
        "Rows fetched    :",
        result.rows_fetched,
    )

    print(
        "Pages fetched   :",
        result.pages_fetched,
    )

    print(
        "First fetched ID:",
        result.first_fetched_id,
    )

    print(
        "Last fetched ID :",
        result.last_fetched_id,
    )

    print(
        "Latest local ID :",
        after_id,
    )

    print(
        "Up to date      :",
        result.up_to_date,
    )

    print()
    print("=" * 80)

    if (
        after_id is not None
        and after_id >= result.target_id
        and result.up_to_date
    ):
        print(
            "STEP 7.4 INCREMENTAL UPDATE: PASSED"
        )
    else:
        print(
            "STEP 7.4 INCREMENTAL UPDATE: REVIEW REQUIRED"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
