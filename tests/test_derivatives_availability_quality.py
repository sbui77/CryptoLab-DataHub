"""
Weakest-link propagation of available_at_quality through
derivatives.core.

available_at_quality describes the evidence behind the
availability timestamp and nothing else. It is not completeness,
and not confidence in the numeric feature values.
"""

import numpy as np
import pandas as pd
import pytest

from cryptolab.features.availability import (
    FeatureAvailabilityError,
    QUALITY_DERIVED,
    QUALITY_EXACT,
    QUALITY_UNKNOWN,
)
from cryptolab.features.derivatives import (
    DerivativesFeatureError,
    build_derivatives_features,
)
from cryptolab.time_contract import (
    point_in_time_filter,
)


BARS = 60


def bar_times() -> pd.DatetimeIndex:
    return pd.date_range(
        "2024-01-01",
        periods=BARS,
        freq="5min",
        tz="UTC",
    )


def make_open_interest(
    quality: str = QUALITY_EXACT,
    availability: object | None = None,
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

    frame["available_at"] = (
        times
        if availability is None
        else availability
    )

    frame[
        "available_at_quality"
    ] = quality

    return frame


def make_basis(
    quality: str = QUALITY_EXACT,
) -> pd.DataFrame:
    times = bar_times()

    return pd.DataFrame(
        {
            "timestamp": times,
            "basis": 1.0,
            "basis_rate": np.linspace(
                0.001,
                0.002,
                BARS,
            ),
            "annualized_basis_rate": 0.1,
            "available_at": times,
            "available_at_quality": quality,
        }
    )


def make_taker_flow(
    quality: str = QUALITY_EXACT,
) -> pd.DataFrame:
    times = bar_times()

    return pd.DataFrame(
        {
            "timestamp": times,
            "buy_volume": 10.0,
            "sell_volume": 8.0,
            "buy_sell_ratio": 1.25,
            "available_at": times,
            "available_at_quality": quality,
        }
    )


def make_funding(
    events: list[
        tuple[
            pd.Timestamp,
            object,
            str,
        ]
    ]
    | None = None,
) -> pd.DataFrame:
    times = bar_times()

    if events is None:
        events = [
            (
                times[0],
                times[0],
                QUALITY_EXACT,
            ),
        ]

    return pd.DataFrame(
        {
            "funding_time": [
                event_time
                for (
                    event_time,
                    _availability,
                    _quality,
                ) in events
            ],
            "funding_rate": [
                0.0001 * (index + 1)
                for index in range(
                    len(events)
                )
            ],
            "mark_price": [
                42_000.0
                for _ in events
            ],
            "rate_type": pd.array(
                [
                    "actual"
                    for _ in events
                ],
                dtype="string",
            ),
            "available_at": [
                availability
                for (
                    _event_time,
                    availability,
                    _quality,
                ) in events
            ],
            "available_at_quality": [
                quality
                for (
                    _event_time,
                    _availability,
                    quality,
                ) in events
            ],
        }
    )


def make_liquidations(
    events: list[
        tuple[
            pd.Timestamp,
            object,
            str,
        ]
    ]
    | None = None,
) -> pd.DataFrame:
    if events is None:
        events = []

    return pd.DataFrame(
        {
            "event_time": [
                event_time
                for (
                    event_time,
                    _availability,
                    _quality,
                ) in events
            ],
            "liquidation_side": [
                "long"
                for _ in events
            ],
            "liquidation_notional": [
                5_000.0
                for _ in events
            ],
            "available_at": [
                availability
                for (
                    _event_time,
                    availability,
                    _quality,
                ) in events
            ],
            "available_at_quality": [
                quality
                for (
                    _event_time,
                    _availability,
                    quality,
                ) in events
            ],
        }
    )


def build(
    open_interest: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    basis: pd.DataFrame | None = None,
    taker_flow: pd.DataFrame | None = None,
    liquidations: pd.DataFrame | None = None,
    require_availability: bool | None = None,
) -> pd.DataFrame:
    return build_derivatives_features(
        open_interest=(
            make_open_interest()
            if open_interest is None
            else open_interest
        ),
        funding_rate=(
            make_funding()
            if funding is None
            else funding
        ),
        basis=(
            make_basis()
            if basis is None
            else basis
        ),
        taker_flow=(
            make_taker_flow()
            if taker_flow is None
            else taker_flow
        ),
        liquidations=(
            make_liquidations()
            if liquidations is None
            else liquidations
        ),
        require_availability=(
            require_availability
        ),
    )


def quality_of(
    result: pd.DataFrame,
) -> pd.Series:
    return result.set_index(
        "timestamp"
    )[
        "available_at_quality"
    ]


def assert_pair_invariant(
    result: pd.DataFrame,
) -> None:
    """
    available_at_quality == 'unknown' iff available_at is NaT.
    """

    unknown = result[
        "available_at_quality"
    ].eq(
        QUALITY_UNKNOWN
    )

    missing = result[
        "available_at"
    ].isna()

    assert (
        unknown.tolist()
        == missing.tolist()
    )


# ================================================================
# CONTRACT PRESENCE
# ================================================================


def test_features_carry_the_metadata_pair():
    result = build()

    assert (
        "available_at"
        in result.columns
    )

    assert (
        "available_at_quality"
        in result.columns
    )

    assert_pair_invariant(
        result
    )


def test_legacy_mode_emits_neither_column():
    """
    require_availability=False is the legacy path. It publishes
    no availability contract at all rather than half of one.
    """

    result = build(
        require_availability=False
    )

    assert (
        "available_at"
        not in result.columns
    )

    assert (
        "available_at_quality"
        not in result.columns
    )


def test_availability_without_quality_is_rejected():
    """
    A timestamp with no recorded evidence quality is
    indistinguishable from an exact one, so the half-contract
    fails rather than silently disabling quality propagation.
    """

    open_interest = make_open_interest().drop(
        columns=[
            "available_at_quality",
        ]
    )

    with pytest.raises(
        DerivativesFeatureError
    ):
        build(
            open_interest=open_interest,
            require_availability=True,
        )


def test_quality_without_availability_is_rejected():
    open_interest = make_open_interest().drop(
        columns=[
            "available_at",
        ]
    )

    with pytest.raises(
        DerivativesFeatureError
    ):
        build(
            open_interest=open_interest,
            require_availability=True,
        )


def test_partial_spine_metadata_is_rejected_in_auto_mode():
    """
    Auto-detection must not read a half-contract as "legacy".
    """

    open_interest = make_open_interest().drop(
        columns=[
            "available_at_quality",
        ]
    )

    with pytest.raises(
        DerivativesFeatureError
    ):
        build(
            open_interest=open_interest
        )


def test_partial_joined_input_metadata_is_rejected():
    basis = make_basis().drop(
        columns=[
            "available_at_quality",
        ]
    )

    with pytest.raises(
        DerivativesFeatureError
    ):
        build(
            basis=basis
        )


def test_inconsistent_input_pair_is_rejected():
    """
    An input claiming exact evidence for an availability it does
    not have violates the pair invariant at the source.

    It surfaces as DerivativesFeatureError like every other
    malformed-metadata failure, with the contract-layer error
    preserved as the cause.
    """

    taker_flow = make_taker_flow()

    taker_flow.loc[
        3,
        "available_at",
    ] = pd.NaT

    with pytest.raises(
        DerivativesFeatureError
    ) as raised:
        build(
            taker_flow=taker_flow
        )

    assert isinstance(
        raised.value.__cause__,
        FeatureAvailabilityError,
    )


def test_invalid_quality_label_is_rejected():
    """
    An unrecognized label is malformed metadata, not a fourth
    quality level.
    """

    with pytest.raises(
        DerivativesFeatureError
    ) as raised:
        build(
            open_interest=make_open_interest(
                quality="approximate",
            )
        )

    assert isinstance(
        raised.value.__cause__,
        FeatureAvailabilityError,
    )


def test_invalid_quality_label_on_liquidations_is_rejected():
    """
    The liquidation aggregation validates before the spine ever
    sees the events, so its failure must be normalized too.
    """

    times = bar_times()

    with pytest.raises(
        DerivativesFeatureError
    ) as raised:
        build(
            liquidations=make_liquidations(
                [
                    (
                        times[3],
                        times[3],
                        "probably",
                    ),
                ]
            )
        )

    assert isinstance(
        raised.value.__cause__,
        FeatureAvailabilityError,
    )


def test_quality_labels_are_normalized():
    """
    Casing and padding are storage noise, not distinct evidence
    levels. The raw storage layer lowercases on read and the
    feature layer must agree with it.
    """

    result = build(
        open_interest=make_open_interest(
            quality="  EXACT  ",
        ),
        basis=make_basis(
            quality="Derived",
        ),
    )

    assert (
        quality_of(
            result
        )
        == QUALITY_DERIVED
    ).all()


# ================================================================
# VALIDATION IS NOT PROPAGATION
# ================================================================


def test_legacy_mode_still_rejects_partial_metadata():
    """
    require_availability=False switches off propagation, not
    validation. Corrupt metadata consumed silently here would
    reach a later migration unnoticed.
    """

    open_interest = make_open_interest().drop(
        columns=[
            "available_at_quality",
        ]
    )

    with pytest.raises(
        DerivativesFeatureError
    ):
        build(
            open_interest=open_interest,
            require_availability=False,
        )


def test_legacy_mode_still_rejects_invalid_labels():
    with pytest.raises(
        DerivativesFeatureError
    ):
        build(
            open_interest=make_open_interest(
                quality="approximate",
            ),
            require_availability=False,
        )


def test_legacy_mode_still_rejects_broken_invariant():
    taker_flow = make_taker_flow()

    taker_flow.loc[
        3,
        "available_at",
    ] = pd.NaT

    with pytest.raises(
        DerivativesFeatureError
    ):
        build(
            taker_flow=taker_flow,
            require_availability=False,
        )


def test_no_internal_bookkeeping_columns_leak():
    """
    Liquidation availability bookkeeping is joined onto the spine
    to compute the contract. None of it belongs in the published
    artifact, in either mode.
    """

    times = bar_times()

    liquidations = make_liquidations(
        [
            (
                times[3],
                times[3],
                QUALITY_DERIVED,
            ),
        ]
    )

    for require in (
        None,
        False,
    ):
        result = build(
            liquidations=liquidations,
            require_availability=require,
        )

        leaked = [
            column
            for column in result.columns
            if column.startswith("_")
        ]

        assert not leaked, (
            require,
            leaked,
        )


# ================================================================
# WEAKEST LINK ACROSS CONSUMED INPUTS
# ================================================================


def test_all_exact_inputs_produce_exact_quality():
    result = build()

    assert (
        quality_of(
            result
        )
        == QUALITY_EXACT
    ).all()

    assert_pair_invariant(
        result
    )


def test_one_derived_input_makes_the_row_derived():
    """
    The weakest link decides. Taker flow is equality-joined on
    every bar, so a derived taker observation degrades every row
    even though every other input is exact.
    """

    result = build(
        taker_flow=make_taker_flow(
            quality=QUALITY_DERIVED,
        )
    )

    assert (
        quality_of(
            result
        )
        == QUALITY_DERIVED
    ).all()

    # Availability itself is unchanged: quality is evidence
    # metadata, not a timestamp.
    assert (
        result["available_at"]
        == result["timestamp"]
    ).all()


def test_quality_is_not_taken_from_the_max_timestamp_holder():
    """
    Input A: available_at 10:10, exact.
    Input B: available_at 10:05, derived.

    Output availability is 10:10 and output quality is derived.
    The published timestamp depends on both observations, so it
    cannot claim to be directly observed.
    """

    times = bar_times()

    later = times + pd.Timedelta(
        minutes=10
    )

    open_interest = make_open_interest(
        quality=QUALITY_EXACT,
        availability=later,
    )

    result = build(
        open_interest=open_interest,
        basis=make_basis(
            quality=QUALITY_DERIVED,
        ),
    )

    assert (
        result["available_at"]
        == later
    ).all()

    assert (
        quality_of(
            result
        )
        == QUALITY_DERIVED
    ).all()


def test_unknown_consumed_observation_makes_the_row_unknown():
    """
    An unknown liquidation is consumed by the bar it falls in,
    so that bar has neither an availability nor evidence for one.
    """

    times = bar_times()

    result = build(
        liquidations=make_liquidations(
            [
                (
                    times[3],
                    pd.NaT,
                    QUALITY_UNKNOWN,
                ),
            ]
        )
    )

    indexed = result.set_index(
        "timestamp"
    )

    assert (
        indexed[
            "available_at_quality"
        ].loc[
            times[3]
        ]
        == QUALITY_UNKNOWN
    )

    assert pd.isna(
        indexed[
            "available_at"
        ].loc[
            times[3]
        ]
    )

    assert_pair_invariant(
        result
    )


def test_mixed_quality_liquidations_in_one_bar_are_derived():
    """
    Two liquidations land in the same 5m bar. The bar consumed
    both, so it carries the worse of the two.
    """

    times = bar_times()

    exact_availability = times[
        3
    ] + pd.Timedelta(
        seconds=30
    )

    result = build(
        liquidations=make_liquidations(
            [
                (
                    times[3],
                    exact_availability,
                    QUALITY_EXACT,
                ),
                (
                    times[3]
                    + pd.Timedelta(
                        minutes=1
                    ),
                    times[3]
                    + pd.Timedelta(
                        minutes=1
                    ),
                    QUALITY_DERIVED,
                ),
            ]
        )
    )

    indexed = result.set_index(
        "timestamp"
    )

    assert (
        indexed[
            "available_at_quality"
        ].loc[
            times[3]
        ]
        == QUALITY_DERIVED
    )

    assert indexed[
        "total_liquidation_notional"
    ].loc[
        times[3]
    ] == 10_000.0


def test_unconsumed_liquidation_quality_does_not_contribute():
    """
    A bar with no liquidation of its own consumed no liquidation
    observation. Once the rolling window has moved past the only
    event, the row is exact again.
    """

    times = bar_times()

    result = build(
        liquidations=make_liquidations(
            [
                (
                    times[3],
                    times[3],
                    QUALITY_DERIVED,
                ),
            ]
        )
    )

    labels = quality_of(
        result
    )

    # Bar 2 precedes the event entirely.
    assert (
        labels.loc[
            times[2]
        ]
        == QUALITY_EXACT
    )

    # Bar 3 consumes it directly.
    assert (
        labels.loc[
            times[3]
        ]
        == QUALITY_DERIVED
    )


# ================================================================
# ROLLING / LAGGED DEPENDENCY WINDOWS
# ================================================================


def test_rolling_quality_covers_the_liquidation_window():
    """
    liquidation_notional_4h consumes 48 observations, so a
    derived liquidation degrades exactly 48 rows.
    """

    times = bar_times()

    result = build(
        liquidations=make_liquidations(
            [
                (
                    times[5],
                    times[5],
                    QUALITY_DERIVED,
                ),
            ]
        )
    )

    labels = quality_of(
        result
    )

    for offset in range(
        0,
        48,
    ):
        assert (
            labels.loc[
                times[5 + offset]
            ]
            == QUALITY_DERIVED
        ), offset

    # Bar 53 no longer consumes the observation in any window.
    assert (
        labels.loc[
            times[53]
        ]
        == QUALITY_EXACT
    )


def test_unknown_outside_the_window_does_not_poison_the_row():
    """
    Unknown travels exactly as far as the dependency window, and
    no further. Availability and quality stop at the same bar.
    """

    times = bar_times()

    result = build(
        liquidations=make_liquidations(
            [
                (
                    times[2],
                    pd.NaT,
                    QUALITY_UNKNOWN,
                ),
            ]
        )
    )

    indexed = result.set_index(
        "timestamp"
    )

    assert (
        indexed[
            "available_at_quality"
        ].loc[
            times[49]
        ]
        == QUALITY_UNKNOWN
    )

    assert pd.isna(
        indexed[
            "available_at"
        ].loc[
            times[49]
        ]
    )

    assert (
        indexed[
            "available_at_quality"
        ].loc[
            times[50]
        ]
        == QUALITY_EXACT
    )

    assert indexed[
        "available_at"
    ].loc[
        times[50]
    ] == times[
        50
    ]

    assert_pair_invariant(
        result
    )


def test_rolling_taker_quality_covers_twelve_observations():
    """
    futures_taker_delta_1h consumes 12 observations.
    """

    times = bar_times()

    taker_flow = make_taker_flow()

    taker_flow.loc[
        20,
        "available_at_quality",
    ] = QUALITY_DERIVED

    result = build(
        taker_flow=taker_flow
    )

    labels = quality_of(
        result
    )

    assert (
        labels.loc[
            times[31]
        ]
        == QUALITY_DERIVED
    )

    assert (
        labels.loc[
            times[32]
        ]
        == QUALITY_EXACT
    )


# ================================================================
# AS-OF FUNDING SELECTION
# ================================================================


def test_selected_funding_quality_propagates():
    times = bar_times()

    result = build(
        funding=make_funding(
            [
                (
                    times[0],
                    times[0],
                    QUALITY_EXACT,
                ),
                (
                    times[20],
                    times[20],
                    QUALITY_DERIVED,
                ),
            ]
        )
    )

    labels = quality_of(
        result
    )

    assert (
        labels.loc[
            times[19]
        ]
        == QUALITY_EXACT
    )

    assert (
        labels.loc[
            times[20]
        ]
        == QUALITY_DERIVED
    )


def test_unselected_funding_quality_does_not_propagate():
    """
    The second funding event exists but is not yet knowable by
    the rows in this fixture, so it was never consumed and its
    derived evidence must not reach them.
    """

    times = bar_times()

    result = build(
        funding=make_funding(
            [
                (
                    times[0],
                    times[0],
                    QUALITY_EXACT,
                ),
                (
                    times[20],
                    times[59]
                    + pd.Timedelta(
                        hours=1
                    ),
                    QUALITY_DERIVED,
                ),
            ]
        )
    )

    assert (
        quality_of(
            result
        )
        == QUALITY_EXACT
    ).all()

    assert (
        result["funding_rate"]
        == 0.0001
    ).all()


def test_unknown_funding_candidate_does_not_degrade_the_row():
    """
    A candidate whose availability is unknown is individually
    ineligible. It is not consumed, so it neither hides the older
    known candidate nor contributes unknown evidence.
    """

    times = bar_times()

    result = build(
        funding=make_funding(
            [
                (
                    times[0],
                    times[0],
                    QUALITY_EXACT,
                ),
                (
                    times[20],
                    pd.NaT,
                    QUALITY_UNKNOWN,
                ),
            ]
        )
    )

    assert (
        quality_of(
            result
        )
        == QUALITY_EXACT
    ).all()

    assert (
        result["funding_rate"]
        == 0.0001
    ).all()

    assert result[
        "available_at"
    ].notna().all()


# ================================================================
# QUALITY IS NOT A CAUSAL FILTER
# ================================================================


def test_derived_quality_does_not_change_causal_eligibility():
    """
    Quality is audit and evidence metadata. A row with a known
    available_at stays usable under the canonical rule whatever
    its evidence quality.
    """

    times = bar_times()

    exact_rows = build()

    derived_rows = build(
        taker_flow=make_taker_flow(
            quality=QUALITY_DERIVED,
        )
    )

    visible_exact = point_in_time_filter(
        exact_rows,
        as_of_time=times[10],
    )

    visible_derived = point_in_time_filter(
        derived_rows,
        as_of_time=times[10],
    )

    assert len(
        visible_exact
    ) == len(
        visible_derived
    )

    assert (
        visible_derived[
            "available_at_quality"
        ]
        == QUALITY_DERIVED
    ).all()


def test_quality_does_not_alter_values_or_timestamps():
    """
    Changing evidence quality alone must move neither the numeric
    features nor available_at.
    """

    exact_rows = build()

    derived_rows = build(
        basis=make_basis(
            quality=QUALITY_DERIVED,
        )
    )

    shared = [
        column
        for column in exact_rows.columns
        if column != "available_at_quality"
    ]

    pd.testing.assert_frame_equal(
        exact_rows[shared],
        derived_rows[shared],
    )
