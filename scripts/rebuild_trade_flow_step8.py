from __future__ import annotations

import pandas as pd

from cryptolab.pipelines.trade_flow_service import (
    run_trade_flow_pipeline,
)
from cryptolab.pipelines.trade_flow_storage import (
    read_trade_flow_features,
)


def main() -> None:
    exchange = "binance"
    symbol = "BTCUSDT"
    timeframe = "1m"

    print()
    print("=" * 100)
    print("STEP 8.10A.4.3 — FULL SPOT TRADE FLOW REBUILD")
    print("=" * 100)

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Raw aggTrades were already backfilled in 8.10A.4.2.
    # Do NOT hit Binance again here.
    # --------------------------------------------------------

    result = run_trade_flow_pipeline(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        update_raw=False,
    )

    print()
    print("RAW DATA")
    print("-" * 100)

    print(
        "Raw rows            :",
        result.raw_rows,
    )

    print(
        "Structurally valid  :",
        result.structurally_valid,
    )

    print(
        "ID continuous       :",
        result.continuous,
    )

    print()

    print("CURATED BUILD")
    print("-" * 100)

    print(
        "Built 1m rows       :",
        result.flow_rows,
    )

    # ========================================================
    # READ BACK FROM STORAGE
    # ========================================================

    df = read_trade_flow_features(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )

    if df.empty:
        raise RuntimeError(
            "Stored Trade Flow dataset is empty "
            "after rebuild"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    df = (
        df
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    first = df["open_time"].min()
    last = df["open_time"].max()

    duplicates = int(
        df["open_time"]
        .duplicated()
        .sum()
    )

    diff = (
        df["open_time"]
        .diff()
    )

    gap_count = int(
        (
            diff
            > pd.Timedelta(minutes=1)
        ).sum()
    )

    expected_rows = int(
        (
            last
            - first
        )
        / pd.Timedelta(minutes=1)
    ) + 1

    coverage = (
        len(df)
        / expected_rows
        if expected_rows > 0
        else 0.0
    )

    print()
    print("STORED 1m COVERAGE")
    print("-" * 100)

    print(
        "Rows                :",
        len(df),
    )

    print(
        "First               :",
        first,
    )

    print(
        "Last                :",
        last,
    )

    print(
        "Expected rows       :",
        expected_rows,
    )

    print(
        "Coverage            :",
        f"{coverage:.2%}",
    )

    print(
        "Duplicates          :",
        duplicates,
    )

    print(
        "Gaps > 1m           :",
        gap_count,
    )

    # ========================================================
    # REQUIRED STEP-8 WINDOW
    # ========================================================

    required_start = pd.Timestamp(
        "2026-08-08T17:15:00Z"
    )

    required_end = pd.Timestamp(
        "2026-08-10T10:59:00Z"
    )

    covers_start = (
        first
        <= required_start
    )

    covers_end = (
        last
        >= required_end
    )

    target = df[
        (
            df["open_time"]
            >= required_start
        )
        & (
            df["open_time"]
            <= required_end
        )
    ].copy()

    target_expected = int(
        (
            required_end
            - required_start
        )
        / pd.Timedelta(minutes=1)
    ) + 1

    target_coverage = (
        len(target)
        / target_expected
        if target_expected > 0
        else 0.0
    )

    target_diff = (
        target["open_time"]
        .diff()
    )

    target_gaps = int(
        (
            target_diff
            > pd.Timedelta(minutes=1)
        ).sum()
    )

    print()
    print("STEP-8 TARGET WINDOW")
    print("-" * 100)

    print(
        "Required start      :",
        required_start,
    )

    print(
        "Required end        :",
        required_end,
    )

    print(
        "Rows                :",
        len(target),
    )

    print(
        "Expected            :",
        target_expected,
    )

    print(
        "Coverage            :",
        f"{target_coverage:.2%}",
    )

    print(
        "Gaps > 1m           :",
        target_gaps,
    )

    print(
        "Covers start        :",
        covers_start,
    )

    print(
        "Covers end          :",
        covers_end,
    )

    # ========================================================
    # CORE FLOW IDENTITIES
    # ========================================================

    identity_columns = {
        "quote_volume",
        "quote_delta",
    }

    identity_available = (
        identity_columns
        <= set(df.columns)
    )

    delta_bound_ok = False

    if identity_available:
        delta_bound_ok = bool(
            (
                df["quote_delta"].abs()
                <= (
                    df["quote_volume"]
                    + 1e-8
                )
            )
            .all()
        )

    print()
    print("FLOW VALIDATION")
    print("-" * 100)

    print(
        "quote columns exist :",
        identity_available,
    )

    print(
        "|delta| <= volume   :",
        delta_bound_ok,
    )

    # ========================================================
    # FINAL
    # ========================================================

    passed = all(
        [
            result.valid,
            result.structurally_valid,
            result.continuous,
            len(df) > 0,
            duplicates == 0,
            target_gaps == 0,
            covers_start,
            covers_end,
            target_coverage >= 0.99,
            identity_available,
            delta_bound_ok,
        ]
    )

    print()
    print("=" * 100)

    if passed:
        print(
            "STEP 8.10A.4.3 SPOT TRADE FLOW REBUILD: PASSED"
        )
    else:
        print(
            "STEP 8.10A.4.3 SPOT TRADE FLOW REBUILD: "
            "REVIEW REQUIRED"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
