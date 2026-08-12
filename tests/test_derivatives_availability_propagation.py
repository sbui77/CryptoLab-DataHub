import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from cryptolab.features.derivatives import (
    DerivativesFeatureError,
    build_derivatives_features,
)
from cryptolab.time_contract import (
    point_in_time_filter,
)


BARS = 24


def bar_times() -> pd.DatetimeIndex:
    return pd.date_range(
        "2024-01-01",
        periods=BARS,
        freq="5min",
        tz="UTC",
    )


def make_open_interest(
    available_at: object | None,
) -> pd.DataFrame:
    times = bar_times()

    frame = pd.DataFrame(
        {
            "exchange": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "period": "5m",
            "timestamp": times,
            "open_interest_base": np.linspace(
                100.0,
                120.0,
                BARS,
            ),
            "open_interest_quote": np.linspace(
                1_000_000.0,
                1_200_000.0,
                BARS,
            ),
        }
    )

    if available_at is not None:
        frame[
            "available_at"
        ] = available_at

    return frame


def make_basis(
    with_availability: bool,
) -> pd.DataFrame:
    times = bar_times()

    frame = pd.DataFrame(
        {
            "timestamp": times,
            "basis": 1.0,
            "basis_rate": 0.001,
            "annualized_basis_rate": 0.1,
        }
    )

    if with_availability:
        frame["available_at"] = times

    return frame


def make_taker_flow(
    with_availability: bool,
) -> pd.DataFrame:
    times = bar_times()

    frame = pd.DataFrame(
        {
            "timestamp": times,
            "buy_volume": 10.0,
            "sell_volume": 8.0,
            "buy_sell_ratio": 1.25,
        }
    )

    if with_availability:
        frame["available_at"] = times

    return frame


def make_liquidations(
    with_availability: bool,
    available_at: object | None = None,
) -> pd.DataFrame:
    times = bar_times()

    frame = pd.DataFrame(
        {
            "event_time": [
                times[3],
            ],
            "liquidation_side": [
                "long",
            ],
            "liquidation_notional": [
                5_000.0,
            ],
        }
    )

    if with_availability:
        frame["available_at"] = [
            available_at
            if available_at is not None
            else times[3]
        ]

    return frame


def make_funding(
    with_availability: bool,
    available_at: object | None = None,
) -> pd.DataFrame:
    times = bar_times()

    frame = pd.DataFrame(
        {
            "funding_time": [
                times[4],
            ],
            "funding_rate": [
                0.0001,
            ],
            "mark_price": [
                42_000.0,
            ],
            "rate_type": pd.array(
                [
                    "actual",
                ],
                dtype="string",
            ),
        }
    )

    if with_availability:
        frame["available_at"] = [
            available_at
            if available_at is not None
            else times[4]
        ]

    return frame


def build(
    funding: pd.DataFrame | None = None,
    open_interest: pd.DataFrame | None = None,
    liquidations: pd.DataFrame | None = None,
    with_availability: bool = True,
) -> pd.DataFrame:
    times = bar_times()

    return build_derivatives_features(
        open_interest=(
            open_interest
            if open_interest is not None
            else make_open_interest(
                times
                if with_availability
                else None
            )
        ),
        funding_rate=(
            funding
            if funding is not None
            else make_funding(
                with_availability
            )
        ),
        basis=make_basis(
            with_availability
        ),
        taker_flow=make_taker_flow(
            with_availability
        ),
        liquidations=(
            liquidations
            if liquidations is not None
            else make_liquidations(
                with_availability
            )
        ),
    )


# ================================================================
# LONG FIXTURES
# ================================================================
#
# Long enough to exercise the real dependency windows:
#   liquidation 4h -> 48 bars
#   basis 24h      -> 288 bars
#   OI pct_change(288) -> 289 observations

LONG_BARS = 300


def long_bar_times() -> pd.DatetimeIndex:
    return pd.date_range(
        "2024-01-01",
        periods=LONG_BARS,
        freq="5min",
        tz="UTC",
    )


def build_long(
    oi_available: object | None = None,
    basis_available: object | None = None,
    taker_available: object | None = None,
    liquidations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    times = long_bar_times()

    open_interest = pd.DataFrame(
        {
            "exchange": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "period": "5m",
            "timestamp": times,
            "open_interest_base": np.linspace(
                100.0,
                200.0,
                LONG_BARS,
            ),
            "open_interest_quote": np.linspace(
                1_000_000.0,
                2_000_000.0,
                LONG_BARS,
            ),
            "available_at": (
                times
                if oi_available is None
                else oi_available
            ),
        }
    )

    basis = pd.DataFrame(
        {
            "timestamp": times,
            "basis": 1.0,
            "basis_rate": np.linspace(
                0.001,
                0.002,
                LONG_BARS,
            ),
            "annualized_basis_rate": 0.1,
            "available_at": (
                times
                if basis_available is None
                else basis_available
            ),
        }
    )

    taker = pd.DataFrame(
        {
            "timestamp": times,
            "buy_volume": 10.0,
            "sell_volume": 8.0,
            "buy_sell_ratio": 1.25,
            "available_at": (
                times
                if taker_available is None
                else taker_available
            ),
        }
    )

    funding = pd.DataFrame(
        {
            "funding_time": [
                times[0],
            ],
            "funding_rate": [
                0.0001,
            ],
            "mark_price": [
                42_000.0,
            ],
            "rate_type": pd.array(
                [
                    "actual",
                ],
                dtype="string",
            ),
            "available_at": [
                times[0],
            ],
        }
    )

    if liquidations is None:
        liquidations = pd.DataFrame(
            {
                "event_time": [],
                "liquidation_side": [],
                "liquidation_notional": [],
                "available_at": [],
            }
        )

    return build_derivatives_features(
        open_interest=open_interest,
        funding_rate=funding,
        basis=basis,
        taker_flow=taker,
        liquidations=liquidations,
    )


# ================================================================
# CONTRACT PRESENCE
# ================================================================


def test_features_carry_available_at():
    result = build()

    assert "available_at" in result.columns

    assert result[
        "available_at"
    ].notna().all()


def test_legacy_inputs_emit_no_availability_column():
    """
    Availability is propagated, never invented. Inputs without
    the contract produce output without it rather than output
    with a fabricated one.
    """

    result = build(
        with_availability=False
    )

    assert (
        "available_at"
        not in result.columns
    )


def test_partial_availability_metadata_is_rejected():
    """
    A spine carrying the contract while a joined input does not
    is a silent-corruption hazard, so it fails loudly.
    """

    with pytest.raises(
        DerivativesFeatureError
    ):
        build_derivatives_features(
            open_interest=make_open_interest(
                bar_times()
            ),
            funding_rate=make_funding(
                True
            ),
            basis=make_basis(
                True
            ),
            taker_flow=make_taker_flow(
                True
            ),
            liquidations=make_liquidations(
                False
            ),
        )


def test_require_availability_rejects_bare_spine():
    with pytest.raises(
        DerivativesFeatureError
    ):
        build_derivatives_features(
            open_interest=make_open_interest(
                None
            ),
            funding_rate=make_funding(
                False
            ),
            basis=make_basis(
                False
            ),
            taker_flow=make_taker_flow(
                False
            ),
            liquidations=make_liquidations(
                False
            ),
            require_availability=True,
        )


# ================================================================
# NO FUTURE LEAKAGE — FUNDING PATH
# ================================================================


def test_funding_known_later_cannot_affect_earlier_rows():
    """
    The funding event's event_time is earlier than most rows,
    but it only became knowable one hour later.

    Rows whose point-in-time cutoff precedes that availability
    must not see it, even though an event-time-only asof join
    would have admitted it.
    """

    times = bar_times()

    late_availability = (
        times[4]
        + pd.Timedelta(
            hours=1
        )
    )

    result = build(
        funding=make_funding(
            True,
            available_at=(
                late_availability
            ),
        )
    )

    known = result[
        "funding_rate"
    ].notna()

    # Nothing before the availability instant may see it.
    before = (
        result["timestamp"]
        < late_availability
    )

    assert not known[
        before
    ].any()

    # From the availability instant onward it is legitimate.
    assert known[
        ~before
    ].all()

    assert (
        result.loc[
            known,
            "timestamp",
        ].min()
        == late_availability
    )


def test_event_time_only_join_would_have_leaked():
    """
    Pins the magnitude of the leak the availability constraint
    removes, so a regression to an event-time-only join is
    visible rather than silent.
    """

    times = bar_times()

    late_availability = (
        times[4]
        + pd.Timedelta(
            hours=1
        )
    )

    guarded = build(
        funding=make_funding(
            True,
            available_at=(
                late_availability
            ),
        )
    )

    event_time_only = build(
        with_availability=False
    )

    guarded_rows = int(
        guarded[
            "funding_rate"
        ].notna().sum()
    )

    leaky_rows = int(
        event_time_only[
            "funding_rate"
        ].notna().sum()
    )

    assert leaky_rows - guarded_rows == 12


def test_row_availability_reflects_late_funding():
    """
    Once a late funding observation is legitimately consumed, it
    dominates the row's availability.
    """

    times = bar_times()

    late_availability = (
        times[4]
        + pd.Timedelta(
            hours=1
        )
    )

    result = build(
        funding=make_funding(
            True,
            available_at=(
                late_availability
            ),
        )
    )

    consuming = result[
        result[
            "funding_rate"
        ].notna()
    ]

    assert (
        consuming[
            "available_at"
        ]
        >= late_availability
    ).all()


def test_unknown_funding_availability_is_not_consumed():
    """
    An observation whose availability cannot be established is
    never treated as knowable.
    """

    result = build(
        funding=make_funding(
            True,
            available_at=pd.NaT,
        )
    )

    assert result[
        "funding_rate"
    ].isna().all()

    # The row itself stays usable; the funding input simply was
    # never consumed.
    assert result[
        "available_at"
    ].notna().all()


# ================================================================
# NO FUTURE LEAKAGE — LIQUIDATION PATH
# ================================================================


def test_late_liquidation_availability_delays_row_usability():
    """
    WebSocket liquidations carry EXACT availability that is
    strictly later than event time. The bar consuming them
    becomes knowable only then.
    """

    times = bar_times()

    late = times[3] + pd.Timedelta(
        minutes=30
    )

    result = build(
        liquidations=make_liquidations(
            True,
            available_at=late,
        )
    )

    bar = result[
        result["timestamp"]
        == times[3]
    ].iloc[0]

    assert bar[
        "total_liquidation_notional"
    ] == 5_000.0

    assert (
        bar["available_at"]
        == late
    )


def test_quiet_bar_still_depends_on_rolling_liquidation_window():
    """
    A bar with no liquidations of its own has a genuine zero in
    total_liquidation_notional, but its liquidation_notional_1h
    and _4h still consume the earlier bar's observation.

    The row is therefore not fully knowable until that
    observation arrived.
    """

    times = bar_times()

    late = times[3] + pd.Timedelta(
        minutes=30
    )

    result = build(
        liquidations=make_liquidations(
            True,
            available_at=late,
        )
    )

    quiet = result[
        result["timestamp"]
        == times[5]
    ].iloc[0]

    assert quiet[
        "total_liquidation_notional"
    ] == 0.0

    assert quiet[
        "liquidation_notional_1h"
    ] == 5_000.0

    assert (
        quiet["available_at"]
        == late
    )


def test_unknown_liquidation_poisons_its_dependency_window():
    """
    Unknown availability propagates exactly as far as the
    dependency window reaches, and no further.
    """

    times = long_bar_times()

    result = build_long(
        liquidations=pd.DataFrame(
            {
                "event_time": [
                    times[3],
                ],
                "liquidation_side": [
                    "long",
                ],
                "liquidation_notional": [
                    5_000.0,
                ],
                "available_at": [
                    pd.NaT,
                ],
            }
        )
    )

    availability = result.set_index(
        "timestamp"
    )["available_at"]

    # Inside the 4h (48 bar) liquidation window.
    assert pd.isna(
        availability.loc[
            times[3]
        ]
    )

    assert pd.isna(
        availability.loc[
            times[5]
        ]
    )

    assert pd.isna(
        availability.loc[
            times[50]
        ]
    )

    # Beyond it the unknown observation is no longer consumed.
    assert availability.loc[
        times[52]
    ] == times[52]


# ================================================================
# NEUTRALITY OF THE MIGRATION
# ================================================================


def test_migration_is_numerically_neutral_on_derived_semantics():
    """
    Current stored availability is DERIVED from event time.

    Under those semantics the two-constraint point-in-time join
    must reproduce the previous event-time-only results exactly.
    Only the causal contract changes, not the present values.
    """

    migrated = build(
        with_availability=True
    )

    legacy = build(
        with_availability=False
    )

    legacy_columns = list(
        legacy.columns
    )

    assert (
        "available_at"
        not in legacy_columns
    )

    assert set(
        migrated.columns
    ) == set(
        legacy_columns
    ) | {
        "available_at",
    }

    pdt.assert_frame_equal(
        migrated[
            legacy_columns
        ],
        legacy,
    )


def test_derived_availability_equals_event_time_spine():
    """
    With derived availability throughout, a row's availability
    is its own bar timestamp.
    """

    result = build()

    times = bar_times()

    # The funding event at times[4] is derived-available at
    # times[4], so it never postdates the bars that consume it.
    pdt.assert_series_equal(
        result["available_at"],
        pd.Series(
            times,
            name="available_at",
        ),
        check_freq=False,
        check_dtype=False,
    )


# ================================================================
# DOWNSTREAM CONSUMABILITY
# ================================================================


def test_point_in_time_filter_applies_to_features():
    """
    The whole purpose of persisting available_at: curated
    features become filterable by the canonical anti-lookahead
    rule.
    """

    result = build()

    times = bar_times()

    visible = point_in_time_filter(
        result,
        as_of_time=times[10],
    )

    assert len(visible) == 11

    assert (
        visible["available_at"]
        <= times[10]
    ).all()


def test_late_availability_shrinks_the_visible_set():
    times = bar_times()

    late = times[3] + pd.Timedelta(
        minutes=30
    )

    result = build(
        liquidations=make_liquidations(
            True,
            available_at=late,
        )
    )

    visible = point_in_time_filter(
        result,
        as_of_time=times[4],
    )

    # The liquidation bar is no longer knowable at times[4].
    assert (
        times[3]
        not in set(
            visible["timestamp"]
        )
    )


# ================================================================
# ROLLING / LAGGED DEPENDENCY AVAILABILITY (FINDING A)
# ================================================================


def test_non_monotonic_basis_availability_inside_rolling_window():
    """
    Mandatory test 1.

    basis(t2) arrives late, basis(t3) on time. basis_rate_change
    at t3 consumes both, so row t3 cannot be knowable before
    basis(t2) arrived.

    Previously this row claimed availability t3 and leaked.
    """

    times = long_bar_times()

    late = times[2] + pd.Timedelta(
        minutes=7
    )

    availability = list(
        times
    )

    availability[2] = late

    result = build_long(
        basis_available=availability
    )

    row = result[
        result["timestamp"]
        == times[3]
    ].iloc[0]

    assert pd.notna(
        row["basis_rate_change"]
    )

    assert (
        row["available_at"]
        >= late
    )


def test_basis_dependency_covers_full_288_bar_window():
    """
    basis_zscore_24h consumes 288 observations, so a single late
    basis point must raise availability for the whole window and
    stop exactly at its edge.
    """

    times = long_bar_times()

    late = times[2] + pd.Timedelta(
        hours=3
    )

    availability = list(
        times
    )

    availability[2] = late

    result = build_long(
        basis_available=availability
    )

    indexed = result.set_index(
        "timestamp"
    )["available_at"]

    # Inside the 288-bar basis window.
    assert (
        indexed.loc[
            times[289]
        ]
        >= late
    )

    # First bar whose window no longer contains observation 2.
    assert (
        indexed.loc[
            times[291]
        ]
        == times[291]
    )


def test_availability_lag_greater_than_one_bar_spacing():
    """
    Mandatory test 2.

    The previously assumed "spine monotonicity is sufficient"
    argument silently relied on lag staying below one bar. A lag
    of 40 minutes spans eight bars.
    """

    times = long_bar_times()

    late = times[10] + pd.Timedelta(
        minutes=40
    )

    availability = list(
        times
    )

    availability[10] = late

    result = build_long(
        taker_available=availability
    )

    indexed = result.set_index(
        "timestamp"
    )["available_at"]

    for offset in range(
        0,
        12,
    ):
        assert (
            indexed.loc[
                times[10 + offset]
            ]
            >= late
        ), offset


def test_late_liquidation_propagates_through_every_affected_row():
    """
    Mandatory test 3.

    A late liquidation must raise availability for every row
    whose 1h or 4h rolling sum includes it - 48 bars - and no
    further.
    """

    times = long_bar_times()

    late = times[10] + pd.Timedelta(
        minutes=40
    )

    result = build_long(
        liquidations=pd.DataFrame(
            {
                "event_time": [
                    times[10],
                ],
                "liquidation_side": [
                    "long",
                ],
                "liquidation_notional": [
                    5_000.0,
                ],
                "available_at": [
                    late,
                ],
            }
        )
    )

    indexed = result.set_index(
        "timestamp"
    )

    affected = indexed.iloc[
        10:58
    ]

    assert (
        affected[
            "liquidation_notional_4h"
        ]
        > 0
    ).all()

    assert (
        affected["available_at"]
        >= late
    ).all()

    # Bar 58 no longer includes the observation in any window.
    assert (
        indexed["available_at"].iloc[
            58
        ]
        == times[58]
    )


def test_non_monotonic_taker_availability_inside_1h_window():
    """
    Mandatory test 4.
    """

    times = long_bar_times()

    late = times[20] + pd.Timedelta(
        minutes=25
    )

    availability = list(
        times
    )

    availability[20] = late

    result = build_long(
        taker_available=availability
    )

    indexed = result.set_index(
        "timestamp"
    )["available_at"]

    # futures_taker_delta_1h consumes 12 observations.
    assert (
        indexed.loc[
            times[31]
        ]
        >= late
    )

    assert (
        indexed.loc[
            times[32]
        ]
        == times[32]
    )


def test_pct_change_288_includes_the_t_minus_288_observation():
    """
    Mandatory test 5.

    oi_quote_change_pct_24h is pct_change(periods=288), which
    consumes 289 observations. An off-by-one window would miss
    the oldest one.
    """

    times = long_bar_times()

    late = times[0] + pd.Timedelta(
        hours=2
    )

    availability = list(
        times
    )

    availability[0] = late

    result = build_long(
        oi_available=availability
    )

    indexed = result.set_index(
        "timestamp"
    )

    row = indexed.iloc[
        288
    ]

    assert pd.notna(
        row[
            "oi_quote_change_pct_24h"
        ]
    )

    assert (
        row["available_at"]
        >= late
    )

    # Observation 0 leaves every window at bar 289.
    assert (
        indexed["available_at"].iloc[
            289
        ]
        == times[289]
    )


# ================================================================
# SEMANTICS B (DECISION 1)
# ================================================================


def test_funding_known_before_row_becomes_usable_is_selected():
    """
    Mandatory test 10.

    OI available 10:01, basis available 10:06, funding F2
    available 10:04.

    The row is not usable until 10:06, by which time F2 was
    known. Under the old spine-cutoff rule F2 was excluded and
    the row carried stale funding.
    """

    times = pd.date_range(
        "2024-01-01 10:00",
        periods=3,
        freq="5min",
        tz="UTC",
    )

    oi_available = pd.Timestamp(
        "2024-01-01 10:01",
        tz="UTC",
    )

    basis_available = pd.Timestamp(
        "2024-01-01 10:06",
        tz="UTC",
    )

    open_interest = pd.DataFrame(
        {
            "exchange": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "period": "5m",
            "timestamp": times,
            "open_interest_base": [
                100.0,
                101.0,
                102.0,
            ],
            "open_interest_quote": [
                1_000_000.0,
                1_010_000.0,
                1_020_000.0,
            ],
            "available_at": [
                oi_available,
            ]
            * 3,
        }
    )

    basis = pd.DataFrame(
        {
            "timestamp": times,
            "basis": 1.0,
            "basis_rate": 0.001,
            "annualized_basis_rate": 0.1,
            "available_at": [
                basis_available,
            ]
            * 3,
        }
    )

    taker = pd.DataFrame(
        {
            "timestamp": times,
            "buy_volume": 10.0,
            "sell_volume": 8.0,
            "buy_sell_ratio": 1.25,
            "available_at": [
                oi_available,
            ]
            * 3,
        }
    )

    funding = pd.DataFrame(
        {
            "funding_time": [
                pd.Timestamp(
                    "2024-01-01 09:00",
                    tz="UTC",
                ),
                pd.Timestamp(
                    "2024-01-01 10:00",
                    tz="UTC",
                ),
            ],
            "funding_rate": [
                0.0001,
                0.0009,
            ],
            "mark_price": [
                42_000.0,
                42_001.0,
            ],
            "rate_type": pd.array(
                [
                    "actual",
                    "actual",
                ],
                dtype="string",
            ),
            "available_at": [
                pd.Timestamp(
                    "2024-01-01 09:00",
                    tz="UTC",
                ),
                pd.Timestamp(
                    "2024-01-01 10:04",
                    tz="UTC",
                ),
            ],
        }
    )

    result = build_derivatives_features(
        open_interest=open_interest,
        funding_rate=funding,
        basis=basis,
        taker_flow=taker,
        liquidations=pd.DataFrame(
            {
                "event_time": [],
                "liquidation_side": [],
                "liquidation_notional": [],
                "available_at": [],
            }
        ),
    )

    assert (
        result["funding_rate"]
        == 0.0009
    ).all()

    assert (
        result["available_at"]
        == basis_available
    ).all()


def test_selected_funding_never_exceeds_provisional_cutoff():
    """
    The final availability is max(provisional, funding), and the
    funding term can never dominate because it was selected
    against the provisional cutoff. Final availability therefore
    equals the provisional value.
    """

    times = long_bar_times()

    late = times[5] + pd.Timedelta(
        minutes=20
    )

    availability = list(
        times
    )

    availability[5] = late

    with_funding = build_long(
        basis_available=availability
    )

    assert with_funding[
        "available_at"
    ].notna().all()

    assert (
        with_funding["available_at"]
        >= with_funding["timestamp"]
    ).all()
