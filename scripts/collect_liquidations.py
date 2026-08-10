from __future__ import annotations

import signal
import sys
import time

import pandas as pd
import websocket

from cryptolab.pipelines.liquidation_storage import (
    save_liquidations,
)
from cryptolab.sources.derivatives_liquidations import (
    get_liquidation_stream_url,
    parse_liquidation_message,
)


SYMBOL = "BTCUSDT"

running = True


def stop_handler(
    signum,
    frame,
) -> None:
    """
    Graceful shutdown handler.
    """

    global running

    running = False

    print()
    print(
        "Stopping liquidation collector..."
    )


def main() -> None:
    """
    Collect Binance BTCUSDT forceOrder events.

    Run manually:

        PYTHONPATH=src python scripts/collect_liquidations.py

    Stop with:

        Ctrl+C
    """

    signal.signal(
        signal.SIGINT,
        stop_handler,
    )

    signal.signal(
        signal.SIGTERM,
        stop_handler,
    )

    url = (
        get_liquidation_stream_url(
            SYMBOL
        )
    )

    print()
    print("=" * 90)
    print("BINANCE LIQUIDATION COLLECTOR")
    print("=" * 90)

    print(
        "Symbol :",
        SYMBOL,
    )

    print(
        "URL    :",
        url,
    )

    print()
    print(
        "Waiting for liquidation events..."
    )

    reconnect_delay = 5.0

    while running:
        ws = None

        try:
            ws = websocket.create_connection(
                url,
                timeout=30,
            )

            print(
                "WebSocket connected."
            )

            while running:
                try:
                    message = ws.recv()

                except websocket.WebSocketTimeoutException:
                    continue

                if not message:
                    continue

                df = parse_liquidation_message(
                    message
                )

                save_liquidations(
                    df
                )

                row = df.iloc[0]

                print(
                    row["event_time"],
                    "|",
                    row["liquidation_side"],
                    "|",
                    f"{row['filled_quantity']:.6f}",
                    "BTC |",
                    f"${row['liquidation_notional']:,.2f}",
                )

        except (
            websocket.WebSocketException,
            OSError,
        ) as exc:
            if not running:
                break

            print(
                "WebSocket error:",
                exc,
            )

            print(
                f"Reconnect in "
                f"{reconnect_delay:.0f}s..."
            )

            time.sleep(
                reconnect_delay
            )

        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    print(
        "Liquidation collector stopped."
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
