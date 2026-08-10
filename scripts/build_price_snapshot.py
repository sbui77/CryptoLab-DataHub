from dataclasses import asdict
from cryptolab.pipelines.price_snapshot import (
    build_and_save_price_snapshot,
)
def _format_pct(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"
def _format_price(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}"
def main() -> None:
    snapshot = (
        build_and_save_price_snapshot(
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1h",
        )
    )
    print()
    print("=" * 100)
    print("CRYPTOLAB PRICE STATE SNAPSHOT")
    print("=" * 100)
    print(
        f"Timestamp          : "
        f"{snapshot.open_time}"
    )
    print(
        f"Price              : "
        f"{_format_price(snapshot.close)}"
    )
    print()
    print("RETURNS")
    print("-" * 100)
    print(
        f"24h                : "
        f"{_format_pct(snapshot.return_24h)}"
    )
    print(
        f"7d                 : "
        f"{_format_pct(snapshot.return_7d)}"
    )
    print(
        f"30d                : "
        f"{_format_pct(snapshot.return_30d)}"
    )
    print()
    print("VOLATILITY")
    print("-" * 100)
    print(
        f"ATR14 %            : "
        f"{_format_pct(snapshot.atr_14_pct)}"
    )
    print(
        f"RV 30d             : "
        f"{_format_pct(snapshot.rv_30d)}"
    )
    print(
        f"RV percentile 1y   : "
        f"{_format_pct(snapshot.rv_30d_percentile_1y)}"
    )
    print(
        f"Volatility regime  : "
        f"{snapshot.volatility_regime}"
    )
    print()
    print("VWAP")
    print("-" * 100)
    print(
        f"Daily VWAP         : "
        f"{_format_price(snapshot.daily_vwap)}"
    )
    print(
        f"Weekly VWAP        : "
        f"{_format_price(snapshot.weekly_vwap)}"
    )
    print(
        f"Monthly VWAP       : "
        f"{_format_price(snapshot.monthly_vwap)}"
    )
    print(
        f"vs Daily VWAP      : "
        f"{_format_pct(snapshot.distance_to_daily_vwap_pct)}"
    )
    print(
        f"vs Weekly VWAP     : "
        f"{_format_pct(snapshot.distance_to_weekly_vwap_pct)}"
    )
    print(
        f"vs Monthly VWAP    : "
        f"{_format_pct(snapshot.distance_to_monthly_vwap_pct)}"
    )
    print()
    print("MARKET STRUCTURE")
    print("-" * 100)
    print(
        f"Structure state    : "
        f"{snapshot.structure_state}"
    )
    print(
        f"Resistance         : "
        f"{_format_price(snapshot.active_resistance)}"
    )
    print(
        f"Support            : "
        f"{_format_price(snapshot.active_support)}"
    )
    print(
        f"Range state        : "
        f"{snapshot.range_state}"
    )
    if (
        snapshot.structural_range_position
        is None
    ):
        structural_position = "N/A"
    else:
        structural_position = (
            f"{snapshot.structural_range_position:.3f}"
        )
    print(
        f"Range position     : "
        f"{structural_position}"
    )
    print(
        f"Nearest structure  : "
        f"{snapshot.nearest_structure_side}"
    )
    print(
        f"Distance           : "
        f"{_format_pct(snapshot.nearest_structure_distance_pct)}"
    )
    print()
    print("STRUCTURAL EVENTS")
    print("-" * 100)
    print(
        f"Latest BOS         : "
        f"{snapshot.latest_bos_direction} "
        f"@ {snapshot.latest_bos_time}"
    )
    print(
        f"Latest CHOCH       : "
        f"{snapshot.latest_choch_direction} "
        f"@ {snapshot.latest_choch_time}"
    )
    print()
    print("REGIME")
    print("-" * 100)
    print(
        f"Price regime       : "
        f"{snapshot.price_regime}"
    )
    print(
        f"Location regime    : "
        f"{snapshot.location_regime}"
    )
    print(
        f"Confidence         : "
        f"{snapshot.regime_confidence}"
    )
    print(
        f"Regime valid       : "
        f"{snapshot.price_regime_valid}"
    )
    print()
    print("DATA QUALITY")
    print("-" * 100)
    print(
        f"24h                : "
        f"{snapshot.quality_24h}"
    )
    print(
        f"7d                 : "
        f"{snapshot.quality_7d}"
    )
    print(
        f"30d                : "
        f"{snapshot.quality_30d}"
    )
    print(
        f"90d                : "
        f"{snapshot.quality_90d}"
    )
    print()
    print("=" * 100)
    print(
        "STEP 6.10 PRICE STATE SNAPSHOT: COMPLETE"
    )
    print("=" * 100)
if __name__ == "__main__":
    main()
