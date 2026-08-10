from __future__ import annotations

import sys

from cryptolab.pipelines.derivatives_update import (
    DerivativesStreamUpdateResult,
    update_derivatives,
)


def print_stream(
    result: DerivativesStreamUpdateResult,
) -> None:
    status = (
        "PASS"
        if result.success
        else "FAIL"
    )

    print(
        f"{result.name:20s}: "
        f"{status}"
    )

    print(
        f"{'rows_fetched':20s}: "
        f"{result.rows_fetched}"
    )

    print(
        f"{'up_to_date':20s}: "
        f"{result.up_to_date}"
    )

    print(
        f"{'first_timestamp':20s}: "
        f"{result.first_timestamp}"
    )

    print(
        f"{'last_timestamp':20s}: "
        f"{result.last_timestamp}"
    )

    if not result.success:
        print(
            f"{'error_type':20s}: "
            f"{result.error_type}"
        )

        print(
            f"{'error_message':20s}: "
            f"{result.error_message}"
        )


def main() -> None:
    result = update_derivatives(
        exchange="binance",
        symbol="BTCUSDT",
        period="5m",
        continue_on_error=True,
    )

    print()
    print("=" * 100)
    print(
        "CRYPTOLAB INCREMENTAL DERIVATIVES UPDATE"
    )
    print("=" * 100)

    print(
        "Exchange   :",
        result.exchange,
    )

    print(
        "Symbol     :",
        result.symbol,
    )

    print(
        "Period     :",
        result.period,
    )

    print(
        "Started at :",
        result.started_at,
    )

    print(
        "Finished at:",
        result.finished_at,
    )

    for stream in [
        result.open_interest,
        result.funding_rate,
        result.basis,
        result.taker_flow,
    ]:
        print()
        print("-" * 100)

        print_stream(
            stream
        )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        "Success       :",
        result.success,
    )

    print(
        "Failure count :",
        result.failure_count,
    )

    print(
        "Failures      :",
        (
            ", ".join(
                result.failures
            )
            if result.failures
            else "None"
        ),
    )

    print()
    print("=" * 100)

    if result.success:
        print(
            "STEP 8.12.3 INCREMENTAL "
            "DERIVATIVES UPDATE: PASSED"
        )

        print("=" * 100)

        return

    print(
        "STEP 8.12.3 INCREMENTAL "
        "DERIVATIVES UPDATE: REVIEW REQUIRED"
    )

    print("=" * 100)

    sys.exit(
        1
    )


if __name__ == "__main__":
    main()
