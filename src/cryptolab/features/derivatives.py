from __future__ import annotations

import numpy as np
import pandas as pd


class DerivativesFeatureError(ValueError):
    """Raised when derivatives feature generation fails."""


CANONICAL_PERIOD = "5m"

BARS_PER_HOUR = 12
BARS_PER_4H = 48
BARS_PER_DAY = 288


def _rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """
    Causal rolling z-score.

    Current observation is included in the rolling
    distribution, but no future information is used.
    """

    mean = series.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    std = series.rolling(
        window=window,
        min_periods=min_periods,
    ).std(
        ddof=0
    )

    return np.where(
        std > 0,
        (
            series
            - mean
        )
        / std,
        np.nan,
    )


def aggregate_liquidations_5m(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate liquidation events into canonical 5m bars.

    Sign convention
    ---------------
    long liquidation:
        forced SELL

    short liquidation:
        forced BUY

    liquidation_delta:
        short_liquidation_notional
        - long_liquidation_notional

    Therefore:
        positive = forced-buy dominance
        negative = forced-sell dominance
    """

    columns = [
        "timestamp",
        "long_liquidation_count",
        "short_liquidation_count",
        "long_liquidation_notional",
        "short_liquidation_notional",
        "total_liquidation_notional",
        "liquidation_delta",
        "liquidation_imbalance",
    ]

    if df.empty:
        return pd.DataFrame(
            columns=columns
        )

    required = [
        "event_time",
        "liquidation_side",
        "liquidation_notional",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise DerivativesFeatureError(
            "Missing liquidation columns: "
            f"{missing}"
        )

    work = df.copy()

    work["event_time"] = pd.to_datetime(
        work["event_time"],
        utc=True,
        errors="coerce",
    )

    if work["event_time"].isna().any():
        raise DerivativesFeatureError(
            "Invalid liquidation event_time"
        )

    work[
        "liquidation_notional"
    ] = pd.to_numeric(
        work["liquidation_notional"],
        errors="raise",
    ).astype("float64")

    is_long = (
        work["liquidation_side"]
        .astype("string")
        .eq("long")
    )

    is_short = (
        work["liquidation_side"]
        .astype("string")
        .eq("short")
    )

    work[
        "long_liquidation_count_i"
    ] = is_long.astype("int64")

    work[
        "short_liquidation_count_i"
    ] = is_short.astype("int64")

    work[
        "long_liquidation_notional_i"
    ] = np.where(
        is_long,
        work["liquidation_notional"],
        0.0,
    )

    work[
        "short_liquidation_notional_i"
    ] = np.where(
        is_short,
        work["liquidation_notional"],
        0.0,
    )

    result = (
        work
        .set_index("event_time")
        .resample(
            "5min",
            label="left",
            closed="left",
            origin="epoch",
        )
        .agg(
            long_liquidation_count=(
                "long_liquidation_count_i",
                "sum",
            ),

            short_liquidation_count=(
                "short_liquidation_count_i",
                "sum",
            ),

            long_liquidation_notional=(
                "long_liquidation_notional_i",
                "sum",
            ),

            short_liquidation_notional=(
                "short_liquidation_notional_i",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "event_time": "timestamp",
            }
        )
    )

    result[
        "total_liquidation_notional"
    ] = (
        result[
            "long_liquidation_notional"
        ]
        + result[
            "short_liquidation_notional"
        ]
    )

    result[
        "liquidation_delta"
    ] = (
        result[
            "short_liquidation_notional"
        ]
        - result[
            "long_liquidation_notional"
        ]
    )

    result[
        "liquidation_imbalance"
    ] = np.where(
        result[
            "total_liquidation_notional"
        ] > 0,
        result[
            "liquidation_delta"
        ]
        / result[
            "total_liquidation_notional"
        ],
        0.0,
    )

    return result[
        columns
    ]


def build_derivatives_features(
    open_interest: pd.DataFrame,
    funding_rate: pd.DataFrame,
    basis: pd.DataFrame,
    taker_flow: pd.DataFrame,
    liquidations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build canonical 5m derivatives feature dataset.

    Main time spine
    ---------------
    Open Interest 5m is used as the canonical clock.

    Other datasets are joined onto this spine.

    Funding is event-based and is backward-asof joined:
        each 5m row sees only the latest funding event
        that occurred at or before that timestamp.

    Liquidations are first aggregated to 5m.
    """

    if open_interest.empty:
        raise DerivativesFeatureError(
            "Open Interest dataset is empty"
        )

    required_oi = [
        "exchange",
        "market",
        "symbol",
        "period",
        "timestamp",
        "open_interest_base",
        "open_interest_quote",
    ]

    missing_oi = [
        column
        for column in required_oi
        if column not in open_interest.columns
    ]

    if missing_oi:
        raise DerivativesFeatureError(
            "Missing Open Interest columns: "
            f"{missing_oi}"
        )

    # ========================================================
    # CANONICAL OI SPINE
    # ========================================================

    result = (
        open_interest[
            required_oi
        ]
        .copy()
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )

    if (
        result["period"]
        .astype(str)
        .ne(CANONICAL_PERIOD)
        .any()
    ):
        raise DerivativesFeatureError(
            "Derivatives features currently require "
            "Open Interest period=5m"
        )

    # ========================================================
    # OPEN INTEREST FEATURES
    # ========================================================

    result["oi_base_change"] = (
        result[
            "open_interest_base"
        ]
        .diff()
    )

    result["oi_quote_change"] = (
        result[
            "open_interest_quote"
        ]
        .diff()
    )

    result["oi_base_change_pct"] = (
        result[
            "open_interest_base"
        ]
        .pct_change(
            fill_method=None
        )
    )

    result["oi_quote_change_pct"] = (
        result[
            "open_interest_quote"
        ]
        .pct_change(
            fill_method=None
        )
    )

    for label, bars in [
        (
            "1h",
            BARS_PER_HOUR,
        ),
        (
            "4h",
            BARS_PER_4H,
        ),
        (
            "24h",
            BARS_PER_DAY,
        ),
    ]:
        result[
            f"oi_quote_change_pct_{label}"
        ] = (
            result[
                "open_interest_quote"
            ]
            .pct_change(
                periods=bars,
                fill_method=None,
            )
        )

    result[
        "oi_quote_zscore_24h"
    ] = _rolling_zscore(
        result[
            "open_interest_quote"
        ],
        window=BARS_PER_DAY,
        min_periods=BARS_PER_HOUR,
    )

    # ========================================================
    # BASIS JOIN
    # ========================================================

    if not basis.empty:
        basis_columns = [
            "timestamp",
            "basis",
            "basis_rate",
            "annualized_basis_rate",
        ]

        missing = [
            column
            for column in basis_columns
            if column not in basis.columns
        ]

        if missing:
            raise DerivativesFeatureError(
                "Missing Basis columns: "
                f"{missing}"
            )

        basis_work = (
            basis[
                basis_columns
            ]
            .copy()
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
        )

        basis_work[
            "timestamp"
        ] = pd.to_datetime(
            basis_work["timestamp"],
            utc=True,
        )

        result = result.merge(
            basis_work,
            on="timestamp",
            how="left",
        )

    else:
        result["basis"] = np.nan
        result["basis_rate"] = np.nan
        result[
            "annualized_basis_rate"
        ] = np.nan

    result["basis_bps"] = (
        result["basis_rate"]
        * 10_000.0
    )

    result[
        "basis_rate_change"
    ] = (
        result[
            "basis_rate"
        ]
        .diff()
    )

    result[
        "basis_zscore_24h"
    ] = _rolling_zscore(
        result["basis_rate"],
        window=BARS_PER_DAY,
        min_periods=BARS_PER_HOUR,
    )

    # ========================================================
    # FUTURES TAKER FLOW JOIN
    # ========================================================

    if not taker_flow.empty:
        taker_columns = [
            "timestamp",
            "buy_volume",
            "sell_volume",
            "buy_sell_ratio",
        ]

        missing = [
            column
            for column in taker_columns
            if column not in taker_flow.columns
        ]

        if missing:
            raise DerivativesFeatureError(
                "Missing Taker Flow columns: "
                f"{missing}"
            )

        taker_work = (
            taker_flow[
                taker_columns
            ]
            .copy()
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .rename(
                columns={
                    "buy_volume": (
                        "futures_buy_volume"
                    ),

                    "sell_volume": (
                        "futures_sell_volume"
                    ),

                    "buy_sell_ratio": (
                        "futures_buy_sell_ratio"
                    ),
                }
            )
        )

        taker_work[
            "timestamp"
        ] = pd.to_datetime(
            taker_work["timestamp"],
            utc=True,
        )

        result = result.merge(
            taker_work,
            on="timestamp",
            how="left",
        )

    else:
        result[
            "futures_buy_volume"
        ] = np.nan

        result[
            "futures_sell_volume"
        ] = np.nan

        result[
            "futures_buy_sell_ratio"
        ] = np.nan

    result[
        "futures_total_volume"
    ] = (
        result[
            "futures_buy_volume"
        ]
        + result[
            "futures_sell_volume"
        ]
    )

    result[
        "futures_taker_delta"
    ] = (
        result[
            "futures_buy_volume"
        ]
        - result[
            "futures_sell_volume"
        ]
    )

    result[
        "futures_taker_delta_pct"
    ] = np.where(
        result[
            "futures_total_volume"
        ] > 0,
        result[
            "futures_taker_delta"
        ]
        / result[
            "futures_total_volume"
        ],
        np.nan,
    )

    result[
        "futures_buy_ratio"
    ] = np.where(
        result[
            "futures_total_volume"
        ] > 0,
        result[
            "futures_buy_volume"
        ]
        / result[
            "futures_total_volume"
        ],
        np.nan,
    )

    result[
        "futures_sell_ratio"
    ] = np.where(
        result[
            "futures_total_volume"
        ] > 0,
        result[
            "futures_sell_volume"
        ]
        / result[
            "futures_total_volume"
        ],
        np.nan,
    )

    result[
        "futures_taker_delta_1h"
    ] = (
        result[
            "futures_taker_delta"
        ]
        .rolling(
            window=BARS_PER_HOUR,
            min_periods=1,
        )
        .sum()
    )

    rolling_futures_volume_1h = (
        result[
            "futures_total_volume"
        ]
        .rolling(
            window=BARS_PER_HOUR,
            min_periods=1,
        )
        .sum()
    )

    result[
        "futures_taker_delta_pct_1h"
    ] = np.where(
        rolling_futures_volume_1h > 0,
        result[
            "futures_taker_delta_1h"
        ]
        / rolling_futures_volume_1h,
        np.nan,
    )

    # ========================================================
    # FUNDING — BACKWARD ASOF JOIN
    # ========================================================

    if not funding_rate.empty:
        funding_columns = [
            "funding_time",
            "funding_rate",
            "mark_price",
            "rate_type",
        ]

        missing = [
            column
            for column in funding_columns
            if column not in funding_rate.columns
        ]

        if missing:
            raise DerivativesFeatureError(
                "Missing Funding columns: "
                f"{missing}"
            )

        funding_work = (
            funding_rate[
                funding_columns
            ]
            .copy()
            .sort_values("funding_time")
            .drop_duplicates(
                subset=["funding_time"],
                keep="last",
            )
        )

        funding_work[
            "funding_time"
        ] = pd.to_datetime(
            funding_work[
                "funding_time"
            ],
            utc=True,
        )

        result = pd.merge_asof(
            result.sort_values(
                "timestamp"
            ),
            funding_work.sort_values(
                "funding_time"
            ),
            left_on="timestamp",
            right_on="funding_time",
            direction="backward",
        )

    else:
        result["funding_time"] = pd.NaT
        result["funding_rate"] = np.nan
        result["mark_price"] = np.nan

        result["rate_type"] = pd.Series(
            [pd.NA] * len(result),
            dtype="string",
        )

    result[
        "funding_rate_bps"
    ] = (
        result["funding_rate"]
        * 10_000.0
    )

    result[
        "hours_since_funding"
    ] = (
        (
            result["timestamp"]
            - result["funding_time"]
        )
        .dt.total_seconds()
        / 3600.0
    )

    # ========================================================
    # LIQUIDATIONS
    # ========================================================

    liquidation_5m = (
        aggregate_liquidations_5m(
            liquidations
        )
    )

    if not liquidation_5m.empty:
        result = result.merge(
            liquidation_5m,
            on="timestamp",
            how="left",
        )

    liquidation_columns = [
        "long_liquidation_count",
        "short_liquidation_count",
        "long_liquidation_notional",
        "short_liquidation_notional",
        "total_liquidation_notional",
        "liquidation_delta",
        "liquidation_imbalance",
    ]

    for column in liquidation_columns:
        if column not in result.columns:
            result[column] = 0.0

        result[column] = (
            result[column]
            .fillna(0.0)
        )

    for column in [
        "long_liquidation_count",
        "short_liquidation_count",
    ]:
        result[column] = (
            result[column]
            .astype("int64")
        )

    result[
        "liquidation_notional_1h"
    ] = (
        result[
            "total_liquidation_notional"
        ]
        .rolling(
            window=BARS_PER_HOUR,
            min_periods=1,
        )
        .sum()
    )

    result[
        "liquidation_notional_4h"
    ] = (
        result[
            "total_liquidation_notional"
        ]
        .rolling(
            window=BARS_PER_4H,
            min_periods=1,
        )
        .sum()
    )

    # ========================================================
    # FINAL ORDER
    # ========================================================

    return (
        result
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )
