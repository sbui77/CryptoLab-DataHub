from __future__ import annotations

import argparse

from cryptolab.pipelines.ohlcv_audit import (
    audit_ohlcv,
)
from cryptolab.pipelines.ohlcv_service import (
    backfill_ohlcv,
    repair_ohlcv_gaps,
    update_ohlcv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptolab",
        description="CryptoLab DataHub CLI",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ohlcv = subparsers.add_parser(
        "ohlcv",
        help="OHLCV data operations",
    )

    operations = ohlcv.add_subparsers(
        dest="operation",
        required=True,
    )

    backfill = operations.add_parser(
        "backfill"
    )

    backfill.add_argument(
        "--symbol",
        default="BTCUSDT",
    )

    backfill.add_argument(
        "--timeframe",
        default="1h",
    )

    backfill.add_argument(
        "--start",
        default=None,
    )

    backfill.add_argument(
        "--end",
        default=None,
    )

    update = operations.add_parser(
        "update"
    )

    update.add_argument(
        "--symbol",
        default="BTCUSDT",
    )

    update.add_argument(
        "--timeframe",
        default="1h",
    )

    audit = operations.add_parser(
        "audit"
    )

    audit.add_argument(
        "--symbol",
        default="BTCUSDT",
    )

    audit.add_argument(
        "--timeframe",
        default="1h",
    )

    repair = operations.add_parser(
        "repair"
    )

    repair.add_argument(
        "--symbol",
        default="BTCUSDT",
    )

    repair.add_argument(
        "--timeframe",
        default="1h",
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    if args.command != "ohlcv":
        parser.error(
            "Unsupported command"
        )

    if args.operation == "backfill":
        df = backfill_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
        )

        print(
            f"Backfill rows: {len(df)}"
        )

    elif args.operation == "update":
        df = update_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
        )

        print(
            f"New rows: {len(df)}"
        )

    elif args.operation == "audit":
        result = audit_ohlcv(
            exchange="binance",
            symbol=args.symbol,
            timeframe=args.timeframe,
        )

        print(result)

    elif args.operation == "repair":
        repaired = repair_ohlcv_gaps(
            symbol=args.symbol,
            timeframe=args.timeframe,
        )

        print(
            f"Repaired candles: {repaired}"
        )


if __name__ == "__main__":
    main()
