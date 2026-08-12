from __future__ import annotations

import numpy as np
import pandas as pd

from cryptolab.features.availability import (
    AVAILABLE_AT,
    combine_availability,
    extract_availability,
    has_availability,
    pit_asof_indices,
    rolling_dependency_availability,
)


class DerivativesFeatureError(ValueError):
    """Raised when derivatives feature generation fails."""


LIQUIDATION_AVAILABILITY = (
    "_liquidation_available_at"
)

LIQUIDATION_EVENT_COUNT = (
    "_liquidation_event_count"
)


# Longest trailing dependency window per raw input, expressed as
# the COUNT of observations an output feature consumes.
#
#   diff()                -> 2
#   pct_change(periods=N) -> N + 1
#   rolling(N)            -> N
#
# Open Interest
#   oi_quote_change_pct_24h  pct_change(288) -> 289  <- longest
#   oi_quote_zscore_24h      rolling(288)    -> 288
#
# Basis
#   basis_zscore_24h         rolling(288)    -> 288  <- longest
#   basis_rate_change        diff()          ->   2
#
# Taker flow
#   futures_taker_delta_1h       rolling(12) ->  12  <- longest
#   futures_taker_delta_pct_1h   rolling(12) ->  12
#
# Liquidations
#   liquidation_notional_4h  rolling(48)     ->  48  <- longest
#   liquidation_notional_1h  rolling(12)     ->  12

CANONICAL_PERIOD = "5m"

BARS_PER_HOUR = 12
BARS_PER_4H = 48
BARS_PER_DAY = 288


OI_DEPENDENCY_BARS = (
    BARS_PER_DAY + 1
)

BASIS_DEPENDENCY_BARS = BARS_PER_DAY

TAKER_DEPENDENCY_BARS = BARS_PER_HOUR

LIQUIDATION_DEPENDENCY_BARS = BARS_PER_4H


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


def _consumed_component(
    frame: pd.DataFrame,
    availability_column: str,
    matched_column: str,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """
    Build one availability component from a left-joined input.

    The consumed mask is taken from an explicit join marker
    rather than from value nullness, so a matched observation
    carrying a null measurement still contributes its
    availability, and an unmatched candidate contributes
    nothing.
    """

    consumed = (
        frame[matched_column]
        .fillna(False)
        .astype(bool)
    )

    return (
        frame[availability_column],
        consumed,
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

    propagate_availability = has_availability(
        df
    )

    if propagate_availability:
        columns = columns + [
            LIQUIDATION_AVAILABILITY,
            LIQUIDATION_EVENT_COUNT,
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

    aggregations: dict[
        str,
        tuple[str, str],
    ] = {
        "long_liquidation_count": (
            "long_liquidation_count_i",
            "sum",
        ),

        "short_liquidation_count": (
            "short_liquidation_count_i",
            "sum",
        ),

        "long_liquidation_notional": (
            "long_liquidation_notional_i",
            "sum",
        ),

        "short_liquidation_notional": (
            "short_liquidation_notional_i",
            "sum",
        ),
    }

    if propagate_availability:
        work[
            AVAILABLE_AT
        ] = pd.to_datetime(
            work[AVAILABLE_AT],
            utc=True,
            errors="coerce",
        )

        work[
            "_event_i"
        ] = 1

        work[
            "_availability_known_i"
        ] = (
            work[AVAILABLE_AT]
            .notna()
            .astype("int64")
        )

        aggregations[
            LIQUIDATION_EVENT_COUNT
        ] = (
            "_event_i",
            "sum",
        )

        aggregations[
            "_availability_known"
        ] = (
            "_availability_known_i",
            "sum",
        )

        aggregations[
            LIQUIDATION_AVAILABILITY
        ] = (
            AVAILABLE_AT,
            "max",
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
            **aggregations
        )
        .reset_index()
        .rename(
            columns={
                "event_time": "timestamp",
            }
        )
    )

    if propagate_availability:
        # A bar that consumed an event of unknown availability
        # has unknown availability. Taking the max over the
        # known subset would understate it.
        incomplete = (
            result[
                "_availability_known"
            ]
            < result[
                LIQUIDATION_EVENT_COUNT
            ]
        )

        result[
            LIQUIDATION_AVAILABILITY
        ] = result[
            LIQUIDATION_AVAILABILITY
        ].where(
            ~incomplete,
            pd.NaT,
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
    require_availability: bool | None = None,
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

    Point-in-time availability
    --------------------------
    `timestamp` is an EVENT timestamp. It is not an availability
    cutoff and is never used as one.

    The row point-in-time cutoff is the availability of the
    Open Interest observation that defines the row:

        row PIT cutoff = spine available_at

    The cutoff is taken from the spine alone so that it is
    independent of join order. Every joined observation must
    satisfy

        input available_at <= row PIT cutoff

    and the emitted row availability is

        available_at = max(available_at of the observations
                           actually consumed by that row)

    require_availability
        None  - propagate when the spine carries available_at
        True  - require it and fail otherwise
        False - skip availability propagation entirely

    Availability is never synthesized from event time here. The
    raw storage layer owns that decision and records it via
    available_at_quality.
    """

    if open_interest.empty:
        raise DerivativesFeatureError(
            "Open Interest dataset is empty"
        )

    if require_availability is None:
        propagate_availability = (
            has_availability(
                open_interest
            )
        )
    else:
        propagate_availability = bool(
            require_availability
        )

    if (
        propagate_availability
        and not has_availability(
            open_interest
        )
    ):
        raise DerivativesFeatureError(
            "Open Interest spine is missing "
            "available_at; point-in-time "
            "availability cannot be established"
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

    if propagate_availability:
        required_oi = required_oi + [
            AVAILABLE_AT,
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
    # EQUALITY-JOINED AVAILABILITY COMPONENTS
    # ========================================================
    #
    # Equality-joined observations constitute the row, so they
    # determine when the row becomes knowable. Each is recorded
    # with its raw per-observation availability here and widened
    # to its actual dependency window below.

    equality_components: list[
        tuple[
            pd.Series,
            pd.Series,
            int,
        ]
    ] = []

    if propagate_availability:
        spine_availability = (
            extract_availability(
                result,
                label="open_interest",
            )
        )

        result = result.drop(
            columns=[
                AVAILABLE_AT,
            ]
        )

        equality_components.append(
            (
                spine_availability,
                pd.Series(
                    True,
                    index=result.index,
                ),
                OI_DEPENDENCY_BARS,
            )
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

        if propagate_availability:
            basis_columns = (
                basis_columns
                + [
                    AVAILABLE_AT,
                ]
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

        if propagate_availability:
            basis_work = basis_work.rename(
                columns={
                    AVAILABLE_AT: (
                        "_basis_available_at"
                    ),
                }
            )

            basis_work[
                "_basis_matched"
            ] = True

        result = result.merge(
            basis_work,
            on="timestamp",
            how="left",
        )

        if propagate_availability:
            values, consumed = (
                _consumed_component(
                    result,
                    availability_column=(
                        "_basis_available_at"
                    ),
                    matched_column=(
                        "_basis_matched"
                    ),
                )
            )

            equality_components.append(
                (
                    values,
                    consumed,
                    BASIS_DEPENDENCY_BARS,
                )
            )

            result = result.drop(
                columns=[
                    "_basis_available_at",
                    "_basis_matched",
                ]
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

        if propagate_availability:
            taker_columns = (
                taker_columns
                + [
                    AVAILABLE_AT,
                ]
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

                    AVAILABLE_AT: (
                        "_taker_available_at"
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

        if propagate_availability:
            taker_work[
                "_taker_matched"
            ] = True

        result = result.merge(
            taker_work,
            on="timestamp",
            how="left",
        )

        if propagate_availability:
            values, consumed = (
                _consumed_component(
                    result,
                    availability_column=(
                        "_taker_available_at"
                    ),
                    matched_column=(
                        "_taker_matched"
                    ),
                )
            )

            equality_components.append(
                (
                    values,
                    consumed,
                    TAKER_DEPENDENCY_BARS,
                )
            )

            result = result.drop(
                columns=[
                    "_taker_available_at",
                    "_taker_matched",
                ]
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
    # LIQUIDATION AGGREGATION
    # ========================================================
    #
    # Aggregated before funding selection because liquidations
    # are equality-joined and therefore help determine the row's
    # knowability, which is the cutoff funding is selected
    # against.

    if (
        propagate_availability
        and not liquidations.empty
        and not has_availability(
            liquidations
        )
    ):
        raise DerivativesFeatureError(
            "Liquidations are missing available_at while "
            "the Open Interest spine carries it; partial "
            "availability metadata cannot be reconciled"
        )

    liquidation_5m = (
        aggregate_liquidations_5m(
            liquidations
        )
    )

    if (
        propagate_availability
        and not liquidation_5m.empty
    ):
        aligned = result[
            [
                "timestamp",
            ]
        ].merge(
            liquidation_5m[
                [
                    "timestamp",
                    LIQUIDATION_AVAILABILITY,
                    LIQUIDATION_EVENT_COUNT,
                ]
            ],
            on="timestamp",
            how="left",
        )

        aligned.index = result.index

        # Only bars that actually aggregated at least one
        # liquidation event consumed a liquidation observation.
        # Empty bars are a real zero, not a consumed input.
        equality_components.append(
            (
                aligned[
                    LIQUIDATION_AVAILABILITY
                ],
                aligned[
                    LIQUIDATION_EVENT_COUNT
                ]
                .fillna(0)
                .astype("int64")
                > 0,
                LIQUIDATION_DEPENDENCY_BARS,
            )
        )

        liquidation_5m = (
            liquidation_5m.drop(
                columns=[
                    LIQUIDATION_AVAILABILITY,
                    LIQUIDATION_EVENT_COUNT,
                ]
            )
        )

    # ========================================================
    # PROVISIONAL ROW AVAILABILITY
    # ========================================================
    #
    # Semantics B: the row describes market state at event-time
    # T and becomes usable once every observation consumed by
    # its outputs is available.
    #
    # Each equality-joined component is widened to the longest
    # trailing window any output consumes from that input, so
    # rolling and lagged features cannot claim to be knowable
    # before their inputs arrived.

    provisional_availability: pd.Series | None = None

    funding_component: tuple[
        pd.Series,
        pd.Series,
    ] | None = None

    if propagate_availability:
        windowed = [
            rolling_dependency_availability(
                values,
                consumed,
                window,
            )
            for (
                values,
                consumed,
                window,
            ) in equality_components
        ]

        provisional_availability = (
            combine_availability(
                windowed
            )
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

        if propagate_availability:
            funding_columns = (
                funding_columns
                + [
                    AVAILABLE_AT,
                ]
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
            .reset_index(drop=True)
        )

        funding_work[
            "funding_time"
        ] = pd.to_datetime(
            funding_work[
                "funding_time"
            ],
            utc=True,
        )

        # Selection under BOTH causal constraints.
        #
        #   funding_time <= row timestamp        (event-time)
        #   available_at <= provisional cutoff   (availability)
        #
        # The row timestamp is an event timestamp and is NOT
        # reused as an availability cutoff. The cutoff is the
        # row's own provisional knowability, so funding that had
        # arrived before the row became usable is not withheld.
        funding_availability = None

        if propagate_availability:
            funding_availability = (
                extract_availability(
                    funding_work,
                    label="funding_rate",
                )
            )

        selected = pit_asof_indices(
            left_event_time=result[
                "timestamp"
            ],
            left_cutoff=(
                provisional_availability
                if provisional_availability
                is not None
                else result["timestamp"]
            ),
            right_event_time=funding_work[
                "funding_time"
            ],
            right_available_at=(
                funding_availability
            ),
        )

        found = selected >= 0

        safe_positions = np.where(
            found,
            selected,
            0,
        )

        found_series = pd.Series(
            found,
            index=result.index,
        )

        payload_columns = [
            column
            for column in funding_columns
            if column != AVAILABLE_AT
        ]

        for column in payload_columns:
            taken = (
                funding_work[column]
                .iloc[safe_positions]
                .reset_index(drop=True)
            )

            taken.index = result.index

            result[column] = taken.where(
                found_series
            )

        if propagate_availability:
            taken_availability = (
                funding_availability
                .iloc[safe_positions]
                .reset_index(drop=True)
            )

            taken_availability.index = (
                result.index
            )

            funding_component = (
                taken_availability,
                found_series,
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
    # LIQUIDATION VALUES
    # ========================================================
    #
    # Availability was captured above, before funding selection.
    # Only the measured values are joined here.

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
    # OUTPUT AVAILABILITY
    # ========================================================

    if propagate_availability:
        # Final row availability:
        #
        #   max(provisional, selected funding available_at)
        #
        # By construction the selected funding availability
        # cannot exceed the provisional cutoff, so this is a
        # single-step fixpoint rather than an iteration, and the
        # result does not depend on join order.
        final_components = [
            (
                provisional_availability,
                pd.Series(
                    True,
                    index=result.index,
                ),
            ),
        ]

        if funding_component is not None:
            final_components.append(
                funding_component
            )

        result[
            AVAILABLE_AT
        ] = combine_availability(
            final_components
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
