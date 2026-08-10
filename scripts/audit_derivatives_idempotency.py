from __future__ import annotations

import hashlib

import pandas as pd

from cryptolab.pipelines.basis_storage import (
    read_basis,
)
from cryptolab.pipelines.derivatives_quality import (
    build_derivatives_quality_dataset,
)
from cryptolab.pipelines.derivatives_quality_covered import (
    build_covered_derivatives_quality_dataset,
)
from cryptolab.pipelines.derivatives_snapshot import (
    build_derivatives_snapshot_dataset,
)
from cryptolab.pipelines.funding_rate_storage import (
    read_funding_rate,
)
from cryptolab.pipelines.liquidation_heartbeat import (
    read_liquidation_heartbeat,
)
from cryptolab.pipelines.liquidation_storage import (
    read_liquidations,
)
from cryptolab.pipelines.open_interest_storage import (
    read_open_interest,
)
from cryptolab.pipelines.taker_flow_storage import (
    read_taker_flow,
)


EXCHANGE = "binance"
SYMBOL = "BTCUSDT"
PERIOD = "5m"


def dataframe_hash(
    df: pd.DataFrame,
) -> str:
    """
    Produce deterministic hash for dataframe content.

    Row order and column order are normalized first.
    """

    if df.empty:
        return "EMPTY"

    work = df.copy()

    work = work.reindex(
        sorted(work.columns),
        axis=1,
    )

    sort_candidates = [
        "timestamp",
        "open_time",
        "funding_time",
        "event_time",
        "heartbeat_time",
    ]

    sort_columns = [
        column
        for column in sort_candidates
        if column in work.columns
    ]

    if sort_columns:
        work = (
            work
            .sort_values(
                sort_columns
            )
            .reset_index(drop=True)
        )

    hashed = pd.util.hash_pandas_object(
        work,
        index=False,
    ).values.tobytes()

    return hashlib.sha256(
        hashed
    ).hexdigest()


def audit_duplicates(
    name: str,
    df: pd.DataFrame,
    key: str,
) -> tuple[bool, str]:
    if df.empty:
        return (
            True,
            f"{name}: empty",
        )

    if key not in df.columns:
        return (
            False,
            f"{name}: missing key={key}",
        )

    duplicates = int(
        df[key]
        .duplicated()
        .sum()
    )

    return (
        duplicates == 0,
        (
            f"{name}: rows={len(df)}, "
            f"duplicates={duplicates}"
        ),
    )


def main() -> None:
    print()
    print("=" * 110)
    print(
        "STEP 8.12.7 — DERIVATIVES IDEMPOTENCY AUDIT"
    )
    print("=" * 110)

    # ========================================================
    # RAW / CURATED STORAGE
    # ========================================================

    datasets = {
        "open_interest": (
            read_open_interest(
                exchange=EXCHANGE,
                symbol=SYMBOL,
                period=PERIOD,
            ),
            "timestamp",
        ),

        "funding": (
            read_funding_rate(
                exchange=EXCHANGE,
                symbol=SYMBOL,
            ),
            "funding_time",
        ),

        "basis": (
            read_basis(
                exchange=EXCHANGE,
                symbol=SYMBOL,
                period=PERIOD,
            ),
            "timestamp",
        ),

        "taker_flow": (
            read_taker_flow(
                exchange=EXCHANGE,
                symbol=SYMBOL,
                period=PERIOD,
            ),
            "timestamp",
        ),

        "liquidations": (
            read_liquidations(
                exchange=EXCHANGE,
                symbol=SYMBOL,
            ),
            "event_time",
        ),

        "heartbeat": (
            read_liquidation_heartbeat(
                exchange=EXCHANGE,
                symbol=SYMBOL,
            ),
            "heartbeat_time",
        ),
    }

    print()
    print("1. STORAGE DUPLICATE AUDIT")
    print("-" * 110)

    storage_results = []

    for name, (
        df,
        key,
    ) in datasets.items():

        passed, detail = (
            audit_duplicates(
                name,
                df,
                key,
            )
        )

        storage_results.append(
            passed
        )

        print(
            f"{name:18s}: "
            f"{'PASS' if passed else 'FAIL'} "
            f"{detail}"
        )

    # ========================================================
    # QUALITY BUILD REPEATABILITY
    # ========================================================

    print()
    print("2. QUALITY REBUILD REPEATABILITY")
    print("-" * 110)

    quality_1 = (
        build_derivatives_quality_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe="1h",
            derivatives_period=PERIOD,
        )
    )

    quality_2 = (
        build_derivatives_quality_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe="1h",
            derivatives_period=PERIOD,
        )
    )

    quality_hash_1 = (
        dataframe_hash(
            quality_1
        )
    )

    quality_hash_2 = (
        dataframe_hash(
            quality_2
        )
    )

    quality_repeatable = (
        quality_hash_1
        == quality_hash_2
    )

    print(
        "Rows run 1      :",
        len(quality_1),
    )

    print(
        "Rows run 2      :",
        len(quality_2),
    )

    print(
        "Hash run 1      :",
        quality_hash_1,
    )

    print(
        "Hash run 2      :",
        quality_hash_2,
    )

    print(
        "Repeatable      :",
        quality_repeatable,
    )

    # ========================================================
    # COVERAGE-AWARE QUALITY REPEATABILITY
    # ========================================================

    print()
    print("3. COVERED QUALITY REPEATABILITY")
    print("-" * 110)

    covered_1 = (
        build_covered_derivatives_quality_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe="1h",
            derivatives_period=PERIOD,
            minimum_liquidation_coverage_ratio=0.80,
            heartbeat_interval_seconds=30.0,
        )
    )

    covered_2 = (
        build_covered_derivatives_quality_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe="1h",
            derivatives_period=PERIOD,
            minimum_liquidation_coverage_ratio=0.80,
            heartbeat_interval_seconds=30.0,
        )
    )

    covered_hash_1 = (
        dataframe_hash(
            covered_1
        )
    )

    covered_hash_2 = (
        dataframe_hash(
            covered_2
        )
    )

    covered_repeatable = (
        covered_hash_1
        == covered_hash_2
    )

    print(
        "Rows run 1      :",
        len(covered_1),
    )

    print(
        "Rows run 2      :",
        len(covered_2),
    )

    print(
        "Hash run 1      :",
        covered_hash_1,
    )

    print(
        "Hash run 2      :",
        covered_hash_2,
    )

    print(
        "Repeatable      :",
        covered_repeatable,
    )

    # ========================================================
    # SNAPSHOT DETERMINISM
    # ========================================================

    print()
    print("4. SNAPSHOT DETERMINISM")
    print("-" * 110)

    snapshot_1, frame_1 = (
        build_derivatives_snapshot_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe="1h",
            derivatives_period=PERIOD,
        )
    )

    snapshot_2, frame_2 = (
        build_derivatives_snapshot_dataset(
            exchange=EXCHANGE,
            symbol=SYMBOL,
            price_timeframe="1h",
            derivatives_period=PERIOD,
        )
    )

    snapshot_hash_1 = (
        dataframe_hash(
            frame_1
        )
    )

    snapshot_hash_2 = (
        dataframe_hash(
            frame_2
        )
    )

    snapshot_repeatable = (
        snapshot_hash_1
        == snapshot_hash_2
    )

    snapshot_identity = (
        snapshot_1.open_time
        == snapshot_2.open_time
        and snapshot_1.as_of_time
        == snapshot_2.as_of_time
        and snapshot_1.derivatives_regime
        == snapshot_2.derivatives_regime
        and snapshot_1.derivatives_quality_score
        == snapshot_2.derivatives_quality_score
    )

    print(
        "Snapshot 1 time :",
        snapshot_1.open_time,
    )

    print(
        "Snapshot 2 time :",
        snapshot_2.open_time,
    )

    print(
        "Hash run 1      :",
        snapshot_hash_1,
    )

    print(
        "Hash run 2      :",
        snapshot_hash_2,
    )

    print(
        "Repeatable      :",
        snapshot_repeatable,
    )

    print(
        "Identity match  :",
        snapshot_identity,
    )

    # ========================================================
    # FINAL
    # ========================================================

    passed = all(
        storage_results
        + [
            quality_repeatable,
            covered_repeatable,
            snapshot_repeatable,
            snapshot_identity,
        ]
    )

    print()
    print("=" * 110)

    if passed:
        print(
            "STEP 8.12.7 DERIVATIVES IDEMPOTENCY: PASSED"
        )
    else:
        print(
            "STEP 8.12.7 DERIVATIVES IDEMPOTENCY: "
            "REVIEW REQUIRED"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()
