from __future__ import annotations

import argparse

from cryptolab.pipelines.liquidation_collector import (
    run_liquidation_collector,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CryptoLab Binance liquidation "
            "WebSocket collector."
        )
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    print()
    print("=" * 100)
    print(
        "CRYPTOLAB BINANCE LIQUIDATION COLLECTOR"
    )
    print("=" * 100)

    print(
        "Symbol              :",
        args.symbol.upper(),
    )

    print(
        "Max messages        :",
        args.max_messages,
    )

    print(
        "Max runtime seconds :",
        args.max_runtime_seconds,
    )

    print(
        "Heartbeat seconds   :",
        args.heartbeat_seconds,
    )

    print()
    print(
        "Collector running..."
    )

    try:
        result = run_liquidation_collector(
            exchange="binance",
            symbol=args.symbol,
            max_messages=args.max_messages,
            max_runtime_seconds=(
                args.max_runtime_seconds
            ),
            heartbeat_interval_seconds=(
                args.heartbeat_seconds
            ),
        )

    except KeyboardInterrupt:
        print()
        print(
            "Collector interrupted by user."
        )
        return

    print()
    print("=" * 100)
    print("COLLECTOR RESULT")
    print("=" * 100)

    print(
        "Started at          :",
        result.started_at,
    )

    print(
        "Finished at         :",
        result.finished_at,
    )

    print(
        "Messages received   :",
        result.messages_received,
    )

    print(
        "Liquidation rows    :",
        result.liquidation_rows,
    )

    print(
        "Rows saved          :",
        result.rows_saved,
    )

    print(
        "Heartbeat rows saved:",
        result.heartbeat_rows_saved,
    )

    print(
        "Reconnects          :",
        result.reconnect_count,
    )

    print(
        "Parse errors        :",
        result.parse_error_count,
    )

    print(
        "Storage errors      :",
        result.storage_error_count,
    )

    print(
        "Stopped by limit    :",
        result.stopped_by_limit,
    )

    print()
    print("=" * 100)

    if result.success:
        print(
            "STEP 8.12.5 LIQUIDATION "
            "HEARTBEAT COLLECTOR: PASSED"
        )
    else:
        print(
            "STEP 8.12.5 LIQUIDATION "
            "HEARTBEAT COLLECTOR: REVIEW REQUIRED"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
