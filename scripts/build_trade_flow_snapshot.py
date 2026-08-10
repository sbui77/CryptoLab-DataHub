from cryptolab.pipelines.trade_flow_snapshot import (
    build_and_save_trade_flow_snapshot,
)
def _format_pct(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"
def _format_number(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.4f}"
def main() -> None:
    snapshot = (
        build_and_save_trade_flow_snapshot(
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1m",
        )
    )
    print()
    print("=" * 100)
    print("CRYPTOLAB TRADE FLOW SNAPSHOT")
    print("=" * 100)
    print(
        "Timestamp               :",
        snapshot.open_time,
    )
    print(
        "Price                   :",
        f"{snapshot.close:,.2f}",
    )
    print(
        "Trade count             :",
        snapshot.trade_count,
    )
    print()
    print("AGGRESSOR FLOW")
    print("-" * 100)
    print(
        "Buy trade ratio         :",
        _format_pct(
            snapshot.buy_trade_ratio
        ),
    )
    print(
        "Sell trade ratio        :",
        _format_pct(
            snapshot.sell_trade_ratio
        ),
    )
    print(
        "Trade-count imbalance   :",
        _format_pct(
            snapshot.trade_count_imbalance
        ),
    )
    print(
        "Buy quote ratio         :",
        _format_pct(
            snapshot.buy_quote_ratio
        ),
    )
    print(
        "Sell quote ratio        :",
        _format_pct(
            snapshot.sell_quote_ratio
        ),
    )
    print(
        "Quote delta             :",
        _format_number(
            snapshot.quote_delta
        ),
    )
    print(
        "Quote delta %           :",
        _format_pct(
            snapshot.quote_delta_pct
        ),
    )
    print()
    print("CVD")
    print("-" * 100)
    print(
        "Base CVD                :",
        _format_number(
            snapshot.base_cvd
        ),
    )
    print(
        "Quote CVD               :",
        _format_number(
            snapshot.quote_cvd
        ),
    )
    print(
        "Daily Base CVD          :",
        _format_number(
            snapshot.daily_base_cvd
        ),
    )
    print(
        "Daily Quote CVD         :",
        _format_number(
            snapshot.daily_quote_cvd
        ),
    )
    print(
        "Rolling Quote CVD 60    :",
        _format_number(
            snapshot.rolling_quote_cvd_60
        ),
    )
    print(
        "Rolling Quote Imbal 60  :",
        _format_pct(
            snapshot.rolling_quote_imbalance_60
        ),
    )
    print()
    print("LARGE TRADES")
    print("-" * 100)
    print(
        "Large trade quote share :",
        _format_pct(
            snapshot.large_trade_quote_share
        ),
    )
    print(
        "Large quote delta       :",
        _format_number(
            snapshot.large_quote_delta
        ),
    )
    print(
        "Large quote delta %     :",
        _format_pct(
            snapshot.large_quote_delta_pct
        ),
    )
    print()
    print("FLOW REGIME")
    print("-" * 100)
    print(
        "Trade Flow Regime       :",
        snapshot.trade_flow_regime,
    )
    print(
        "Regime confidence       :",
        snapshot.flow_regime_confidence,
    )
    print(
        "Regime valid            :",
        snapshot.trade_flow_regime_valid,
    )
    print()
    print("=" * 100)
    print(
        "STEP 7.11 TRADE FLOW SNAPSHOT: COMPLETE"
    )
    print("=" * 100)
if __name__ == "__main__":
    main()
