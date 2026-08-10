from __future__ import annotations

import pandas as pd

from cryptolab.pipelines.aggtrades_backfill import (
    backfill_aggtrades,
)
from cryptolab.pipelines.aggtrades_storage import (
    read_aggtrades,
)


START_TIME = pd.Timestamp(
    "2026-08-08T17:15:00Z"
)

# Use 11:00 rather than 10:50 so the 10:50–10:54:59
# Spot 5m bucket can be completed.
END_TIME = pd.Timestamp(
    "2026-08-10T11:00:00Z"
)


def main() -> None:
    print()
    print("=" * 100)
    print("STEP 8.10A.4.2 — aggTrades HISTORICAL BACKFILL")
    print("=" * 100)

    print(
        "Requested start:",
        START_TIME,
    )

    print(
        "Requested end  :",
        END_TIME,
    )

    print()

    result = backfill_aggtrades(
        exchange="binance",
        symbol="BTCUSDT",
        start_time=START_TIME,
        end_time=END_TIME,
        page_limit=1000,
        pause_seconds=0.20,
    )

    print()
    print("BACKFILL RESULT")
    print("-" * 100)

    print(
        "Seed ID          :",
        result.seed_id,
    )

    print(
        "Rows fetched     :",
        result.rows_fetched,
    )

    print(
        "Rows saved       :",
        result.rows_saved,
    )

    print(
        "Pages fetched    :",
        result.pages_fetched,
    )

    print(
        "First saved ID   :",
        result.first_saved_id,
    )

    print(
        "Last saved ID    :",
        result.last_saved_id,
    )

    print(
        "First saved time :",
        result.first_saved_time,
    )

    print(
        "Last saved time  :",
        result.last_saved_time,
    )

    print(
        "Crossed end time :",
        result.crossed_end_time,
    )

    print(
        "Complete         :",
        result.complete,
    )

    # ========================================================
    # STORAGE AUDIT
    # ========================================================

    df = read_aggtrades(
        exchange="binance",
        symbol="BTCUSDT",
    )

    target = df[
        (
            df["trade_time"]
            >= START_TIME
        )
        & (
            df["trade_time"]
            <= END_TIME
        )
    ].copy()

    duplicates = int(
        target[
            "agg_trade_id"
        ]
        .duplicated()
        .sum()
    )

    ids = (
        target[
            "agg_trade_id"
        ]
        .sort_values()
    )

    id_diff = (
        ids.diff()
    )

    id_gap_events = int(
        (
            id_diff > 1
        ).sum()
    )

    missing_ids = int(
        (
            id_diff[
                id_diff > 1
            ]
            - 1
        ).sum()
    )

    covers_start = (
        not target.empty
        and (
            target[
                "trade_time"
            ].min()
            <= (
                START_TIME
                + pd.Timedelta(
                    seconds=10
                )
            )
        )
    )

    # We only store <= END_TIME, so the final saved trade
    # will normally be shortly BEFORE END_TIME.
    covers_end = (
        not target.empty
        and (
            target[
                "trade_time"
            ].max()
            >= (
                END_TIME
                - pd.Timedelta(
                    seconds=10
                )
            )
        )
    )

    print()
    print("STORAGE AUDIT")
    print("-" * 100)

    print(
        "Rows in target  :",
        len(target),
    )

    print(
        "First           :",
        (
            target["trade_time"].min()
            if not target.empty
            else None
        ),
    )

    print(
        "Last            :",
        (
            target["trade_time"].max()
            if not target.empty
            else None
        ),
    )

    print(
        "Duplicates      :",
        duplicates,
    )

    print(
        "ID gap events   :",
        id_gap_events,
    )

    print(
        "Missing IDs     :",
        missing_ids,
    )

    print(
        "Covers start    :",
        covers_start,
    )

    print(
        "Covers end      :",
        covers_end,
    )

    passed = (
        result.complete
        and result.crossed_end_time
        and not target.empty
        and duplicates == 0
        and id_gap_events == 0
        and missing_ids == 0
        and covers_start
        and covers_end
    )

    print()
    print("=" * 100)

    if passed:
        print(
            "STEP 8.10A.4.2 aggTrades BACKFILL: PASSED"
        )
    else:
        print(
            "STEP 8.10A.4.2 aggTrades BACKFILL: "
            "REVIEW REQUIRED"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
