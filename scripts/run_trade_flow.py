from cryptolab.pipelines.trade_flow_service import (
    run_trade_flow_pipeline,
)
def main() -> None:
    result = run_trade_flow_pipeline(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        update_raw=True,
    )
    update = result.update_result
    audit = result.audit_result
    snapshot = result.snapshot
    print()
    print("=" * 100)
    print("CRYPTOLAB TRADE FLOW PRODUCTION PIPELINE")
    print("=" * 100)
    print()
    print("RAW UPDATE")
    print("-" * 100)
    print(
        "Start ID            :",
        update.start_id,
    )
    print(
        "Target ID           :",
        update.target_id,
    )
    print(
        "Rows fetched        :",
        update.rows_fetched,
    )
    print(
        "Pages fetched       :",
        update.pages_fetched,
    )
    print(
        "Up to date          :",
        update.up_to_date,
    )
    print()
    print("RAW DATA AUDIT")
    print("-" * 100)
    print(
        "Rows                :",
        audit.rows,
    )
    print(
        "First trade         :",
        audit.first_trade_time,
    )
    print(
        "Last trade          :",
        audit.last_trade_time,
    )
    print(
        "Duplicates          :",
        audit.duplicate_count,
    )
    print(
        "ID gap groups       :",
        audit.id_gap_count,
    )
    print(
        "Missing IDs         :",
        audit.missing_id_count,
    )
    print(
        "Structurally valid  :",
        audit.structurally_valid,
    )
    print(
        "Continuous          :",
        audit.continuous,
    )
    print()
    print("CURATED FLOW")
    print("-" * 100)
    print(
        "Raw trades          :",
        result.raw_rows,
    )
    print(
        "Curated 1m bars     :",
        result.flow_rows,
    )
    print()
    print("LATEST SNAPSHOT")
    print("-" * 100)
    print(
        "Timestamp           :",
        snapshot.open_time,
    )
    print(
        "Price               :",
        snapshot.close,
    )
    print(
        "Quote delta %       :",
        snapshot.quote_delta_pct,
    )
    print(
        "Rolling imbalance   :",
        snapshot.rolling_quote_imbalance_60,
    )
    print(
        "Daily Quote CVD     :",
        snapshot.daily_quote_cvd,
    )
    print(
        "Large quote share   :",
        snapshot.large_trade_quote_share,
    )
    print(
        "Large quote delta % :",
        snapshot.large_quote_delta_pct,
    )
    print(
        "Flow regime         :",
        snapshot.trade_flow_regime,
    )
    print(
        "Regime confidence   :",
        snapshot.flow_regime_confidence,
    )
    print(
        "Regime valid        :",
        snapshot.trade_flow_regime_valid,
    )
    print()
    print("=" * 100)
    if result.valid:
        print(
            "STEP 7.12 TRADE FLOW PRODUCTION: PASSED"
        )
    else:
        print(
            "STEP 7.12 TRADE FLOW PRODUCTION: REVIEW REQUIRED"
        )
    print("=" * 100)
if __name__ == "__main__":
    main()
