from __future__ import annotations

import sys

from cryptolab.pipelines.derivatives_snapshot import (
    build_and_save_derivatives_snapshot,
)
from cryptolab.pipelines.derivatives_update import (
    update_derivatives,
)


def main() -> None:
    print()
    print("=" * 100)
    print("CRYPTOLAB DERIVATIVES PRODUCTION RUN")
    print("=" * 100)

    update = update_derivatives(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
        continue_on_error=True,
    )

    print()
    print("REST UPDATE")
    print("-" * 100)

    streams = [
        update.open_interest,
        update.funding_rate,
        update.basis,
        update.taker_flow,
    ]

    for stream in streams:
        status = (
            "PASS"
            if stream.success
            else "FAIL"
        )

        print(
            f"{stream.name:20s}: "
            f"{status:4s} "
            f"rows={stream.rows_fetched} "
            f"up_to_date={stream.up_to_date}"
        )

        if not stream.success:
            print(
                " " * 22
                + f"{stream.error_type}: "
                + f"{stream.error_message}"
            )

    # --------------------------------------------------------
    # At least the core high-frequency streams must survive.
    #
    # Basis may occasionally be externally rate-limited.
    # Existing locally stored basis data remains available,
    # and downstream quality masks determine whether evidence
    # is still usable.
    # --------------------------------------------------------

    core_streams_ok = all(
        [
            update.open_interest.success,
            update.funding_rate.success,
            update.taker_flow.success,
        ]
    )

    if not core_streams_ok:
        print()
        print("=" * 100)
        print(
            "DERIVATIVES PRODUCTION RUN: FAILED"
        )
        print(
            "One or more core REST streams failed."
        )
        print("=" * 100)

        sys.exit(1)

    print()
    print("DOWNSTREAM SNAPSHOT")
    print("-" * 100)

    try:
        snapshot, output = (
            build_and_save_derivatives_snapshot(
                exchange="binance",
                symbol="BTCUSDT",
                price_timeframe="1h",
                derivatives_period="5m",
            )
        )

    except Exception as exc:
        print(
            "Snapshot rebuild failed:",
            f"{type(exc).__name__}: {exc}",
        )

        print()
        print("=" * 100)
        print(
            "DERIVATIVES PRODUCTION RUN: FAILED"
        )
        print("=" * 100)

        sys.exit(1)

    print(
        "Open time   :",
        snapshot.open_time,
    )

    print(
        "As-of time  :",
        snapshot.as_of_time,
    )

    print(
        "Regime      :",
        snapshot.derivatives_regime,
    )

    print(
        "Quality     :",
        snapshot.derivatives_quality_tier,
    )

    print(
        "Output      :",
        output,
    )

    print()
    print("=" * 100)

    if update.basis.success:
        print(
            "STEP 8.12.9 PRODUCTION RUN: PASSED"
        )
    else:
        print(
            "STEP 8.12.9 PRODUCTION RUN: "
            "PASSED WITH BASIS DEGRADED"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
