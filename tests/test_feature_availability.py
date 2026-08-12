import numpy as np
import pandas as pd
import pytest

from cryptolab.features.availability import (
    FeatureAvailabilityError,
    combine_availability,
    extract_availability,
    has_availability,
    pit_asof_indices,
    rolling_dependency_availability,
)


def ts(
    value: str,
) -> pd.Timestamp:
    return pd.Timestamp(
        value,
        tz="UTC",
    )


def series(
    *values: object,
) -> pd.Series:
    return pd.Series(
        [
            pd.Timestamp(
                value,
                tz="UTC",
            )
            if value is not None
            else pd.NaT
            for value in values
        ]
    )


def mask(
    *values: bool,
) -> pd.Series:
    return pd.Series(
        list(
            values
        )
    )


# ================================================================
# EXTRACTION
# ================================================================


def test_missing_availability_is_never_fabricated():
    frame = pd.DataFrame(
        {
            "event_time": [
                ts("2024-01-01"),
            ],
        }
    )

    assert not has_availability(
        frame
    )

    with pytest.raises(
        FeatureAvailabilityError
    ):
        extract_availability(
            frame,
            label="thing",
        )


def test_unknown_availability_is_preserved_as_nat():
    """
    Raw storage encodes unknown availability as NaT with
    available_at_quality='unknown'. The feature layer must not
    silently replace it with event time.
    """

    frame = pd.DataFrame(
        {
            "event_time": [
                ts("2024-01-01"),
            ],
            "available_at": [
                pd.NaT,
            ],
        }
    )

    result = extract_availability(
        frame,
        label="thing",
    )

    assert result.isna().all()


# ================================================================
# COMBINATION — MAX_INPUT_AVAILABLE_AT
# ================================================================


def test_output_availability_is_max_of_consumed_inputs():
    result = combine_availability(
        [
            (
                series(
                    "2024-01-01 00:00",
                    "2024-01-01 00:00",
                ),
                mask(
                    True,
                    True,
                ),
            ),
            (
                series(
                    "2024-01-01 05:00",
                    "2024-01-01 00:30",
                ),
                mask(
                    True,
                    True,
                ),
            ),
        ]
    )

    assert result.tolist() == [
        ts("2024-01-01 05:00"),
        ts("2024-01-01 00:30"),
    ]


def test_unconsumed_candidates_do_not_affect_availability():
    """
    A join candidate that was not selected must not influence
    the output row, even when its availability is far later.
    """

    result = combine_availability(
        [
            (
                series(
                    "2024-01-01 00:00",
                ),
                mask(
                    True,
                ),
            ),
            (
                series(
                    "2030-01-01 00:00",
                ),
                mask(
                    False,
                ),
            ),
        ]
    )

    assert result.tolist() == [
        ts("2024-01-01 00:00"),
    ]


def test_consumed_unknown_availability_poisons_the_row():
    """
    If a contributing observation has unknown availability, the
    row's availability is unknown. Taking the max over the known
    subset would understate it and overstate usability.
    """

    result = combine_availability(
        [
            (
                series(
                    "2024-01-01 00:00",
                ),
                mask(
                    True,
                ),
            ),
            (
                series(
                    None,
                ),
                mask(
                    True,
                ),
            ),
        ]
    )

    assert result.isna().all()


def test_unconsumed_unknown_availability_is_harmless():
    result = combine_availability(
        [
            (
                series(
                    "2024-01-01 00:00",
                ),
                mask(
                    True,
                ),
            ),
            (
                series(
                    None,
                ),
                mask(
                    False,
                ),
            ),
        ]
    )

    assert result.tolist() == [
        ts("2024-01-01 00:00"),
    ]


def test_combine_requires_at_least_one_component():
    with pytest.raises(
        FeatureAvailabilityError
    ):
        combine_availability(
            []
        )


# ================================================================
# ROLLING / LAGGED DEPENDENCY AVAILABILITY
# ================================================================
#
# assert_non_decreasing was removed. Monotonic availability is no
# longer required for correctness: rolling availability is now
# derived from the actual dependency window rather than being
# assumed to be dominated by the current row. These tests pin
# that non-monotonic arrival is handled rather than rejected.


def test_rolling_window_takes_max_over_dependency():
    values, consumed = (
        rolling_dependency_availability(
            series(
                "2024-01-01 00:00",
                "2024-01-01 00:17",
                "2024-01-01 00:15",
            ),
            mask(
                True,
                True,
                True,
            ),
            window=2,
        )
    )

    assert values.tolist() == [
        ts("2024-01-01 00:00"),
        ts("2024-01-01 00:17"),
        ts("2024-01-01 00:17"),
    ]

    assert consumed.tolist() == [
        True,
        True,
        True,
    ]


def test_late_arriving_backfill_is_handled_not_rejected():
    """
    Replaces the former assert_non_decreasing guard.

    A prior observation that arrived later than the current one
    must raise the current row's availability, not raise an
    exception.
    """

    values, _ = (
        rolling_dependency_availability(
            series(
                "2024-01-03",
                "2024-01-01",
            ),
            mask(
                True,
                True,
            ),
            window=2,
        )
    )

    assert values.tolist() == [
        ts("2024-01-03"),
        ts("2024-01-03"),
    ]


def test_window_of_one_is_identity():
    values, consumed = (
        rolling_dependency_availability(
            series(
                "2024-01-01",
                "2024-01-05",
            ),
            mask(
                True,
                True,
            ),
            window=1,
        )
    )

    assert values.tolist() == [
        ts("2024-01-01"),
        ts("2024-01-05"),
    ]


def test_unknown_outside_window_does_not_poison():
    """
    An unknown observation must only affect rows whose
    dependency window actually contains it.
    """

    values, _ = (
        rolling_dependency_availability(
            series(
                None,
                "2024-01-01 01:00",
                "2024-01-01 02:00",
            ),
            mask(
                True,
                True,
                True,
            ),
            window=2,
        )
    )

    assert pd.isna(
        values.iloc[0]
    )

    assert pd.isna(
        values.iloc[1]
    )

    assert values.iloc[2] == ts(
        "2024-01-01 02:00"
    )


def test_unconsumed_inside_window_is_ignored():
    values, consumed = (
        rolling_dependency_availability(
            series(
                "2030-01-01",
                "2024-01-01 01:00",
            ),
            mask(
                False,
                True,
            ),
            window=2,
        )
    )

    assert pd.isna(
        values.iloc[0]
    )

    assert values.iloc[1] == ts(
        "2024-01-01 01:00"
    )

    assert consumed.tolist() == [
        False,
        True,
    ]


def test_rolling_preserves_nanosecond_precision():
    """
    pandas rolling aggregates through float64, which cannot
    represent nanosecond epoch values exactly. Rank encoding
    guards against silent rounding.
    """

    values, _ = (
        rolling_dependency_availability(
            series(
                "2024-01-01 00:00:00.000000001",
                "2024-01-01 00:00:00.000000002",
            ),
            mask(
                True,
                True,
            ),
            window=2,
        )
    )

    assert values.iloc[1] == pd.Timestamp(
        "2024-01-01 00:00:00.000000002",
        tz="UTC",
    )


def test_zero_window_is_rejected():
    with pytest.raises(
        FeatureAvailabilityError
    ):
        rolling_dependency_availability(
            series(
                "2024-01-01",
            ),
            mask(
                True,
            ),
            window=0,
        )


# ================================================================
# TWO-CONSTRAINT POINT-IN-TIME SELECTION
# ================================================================


def test_event_time_constraint_blocks_future_events():
    """
    A bar labelled T must describe state at T. An event after T
    cannot enter it even when already knowable.
    """

    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 00:00",
        ),
        left_cutoff=series(
            "2030-01-01 00:00",
        ),
        right_event_time=series(
            "2024-01-01 00:30",
        ),
        right_available_at=series(
            "2024-01-01 00:30",
        ),
    )

    assert selected.tolist() == [
        -1,
    ]


def test_availability_constraint_blocks_unknowable_events():
    """
    An observation whose event time is earlier but which had not
    yet arrived cannot affect the row.
    """

    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 01:00",
        ),
        left_cutoff=series(
            "2024-01-01 01:00",
        ),
        right_event_time=series(
            "2024-01-01 00:30",
        ),
        right_available_at=series(
            "2024-01-01 09:00",
        ),
    )

    assert selected.tolist() == [
        -1,
    ]


def test_falls_back_to_older_knowable_observation():
    """
    When the newest event is not yet knowable, the latest event
    satisfying both constraints is used instead.
    """

    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 02:00",
        ),
        left_cutoff=series(
            "2024-01-01 02:00",
        ),
        right_event_time=series(
            "2024-01-01 00:00",
            "2024-01-01 01:00",
        ),
        right_available_at=series(
            "2024-01-01 00:00",
            "2024-01-01 09:00",
        ),
    )

    assert selected.tolist() == [
        0,
    ]


def test_unknown_candidate_does_not_hide_later_known_one():
    """
    known, NaT, known

    The NaT candidate is individually ineligible but must not
    truncate candidacy: the latest known candidate is still
    selectable.
    """

    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 11:00",
        ),
        left_cutoff=series(
            "2024-01-01 11:00",
        ),
        right_event_time=series(
            "2024-01-01 10:00",
            "2024-01-01 10:01",
            "2024-01-01 10:02",
        ),
        right_available_at=series(
            "2024-01-01 10:00",
            None,
            "2024-01-01 10:05",
        ),
    )

    assert selected.tolist() == [
        2,
    ]


def test_multiple_interspersed_unknowns():
    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 11:00",
        ),
        left_cutoff=series(
            "2024-01-01 11:00",
        ),
        right_event_time=series(
            "2024-01-01 10:00",
            "2024-01-01 10:01",
            "2024-01-01 10:02",
            "2024-01-01 10:03",
            "2024-01-01 10:04",
        ),
        right_available_at=series(
            "2024-01-01 10:00",
            None,
            "2024-01-01 10:05",
            None,
            "2024-01-01 10:06",
        ),
    )

    assert selected.tolist() == [
        4,
    ]


def test_gap_repair_breaks_availability_monotonicity():
    """
    A missed funding event backfilled after a later live event
    makes availability non-monotonic in event order.

    The 08:00 event only arrived at 20:00 and must be excluded,
    while the later 16:00 event remains selectable.
    """

    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 17:00",
        ),
        left_cutoff=series(
            "2024-01-01 17:00",
        ),
        right_event_time=series(
            "2024-01-01 00:00",
            "2024-01-01 08:00",
            "2024-01-01 16:00",
        ),
        right_available_at=series(
            "2024-01-01 00:00",
            "2024-01-01 20:00",
            "2024-01-01 16:05",
        ),
    )

    assert selected.tolist() == [
        2,
    ]


def brute_force_selection(
    left_event_time: pd.Series,
    left_cutoff: pd.Series,
    right_event_time: pd.Series,
    right_available_at: pd.Series,
) -> list[int]:
    """
    Direct transcription of the point-in-time definition.

    Deliberately naive: this is the oracle.
    """

    answers = []

    for (
        event,
        cutoff,
    ) in zip(
        left_event_time,
        left_cutoff,
    ):
        best = -1

        for position in range(
            len(right_event_time)
        ):
            candidate_event = (
                right_event_time.iloc[
                    position
                ]
            )

            candidate_available = (
                right_available_at.iloc[
                    position
                ]
            )

            if pd.isna(
                cutoff
            ):
                continue

            if pd.isna(
                candidate_available
            ):
                continue

            if candidate_event > event:
                continue

            if candidate_available > cutoff:
                continue

            best = position

        answers.append(
            best
        )

    return answers


def test_randomized_differential_against_brute_force():
    """
    Randomized differential test over deliberately hostile
    inputs: non-monotonic availability, interspersed unknowns,
    unknown cutoffs.
    """

    rng = np.random.default_rng(
        20240101
    )

    base = pd.Timestamp(
        "2024-01-01",
        tz="UTC",
    )

    for _ in range(
        200
    ):
        candidate_count = int(
            rng.integers(
                1,
                12,
            )
        )

        row_count = int(
            rng.integers(
                1,
                15,
            )
        )

        event_offsets = np.sort(
            rng.integers(
                0,
                400,
                candidate_count,
            )
        )

        right_event = pd.Series(
            [
                base
                + pd.Timedelta(
                    minutes=int(
                        offset
                    )
                )
                for offset in event_offsets
            ]
        )

        # Availability deliberately unrelated to event order.
        right_available = pd.Series(
            [
                pd.NaT
                if rng.random() < 0.25
                else base
                + pd.Timedelta(
                    minutes=int(
                        rng.integers(
                            0,
                            400,
                        )
                    )
                )
                for _ in range(
                    candidate_count
                )
            ]
        )

        left_event = pd.Series(
            [
                base
                + pd.Timedelta(
                    minutes=int(
                        rng.integers(
                            0,
                            400,
                        )
                    )
                )
                for _ in range(
                    row_count
                )
            ]
        )

        left_cutoff = pd.Series(
            [
                pd.NaT
                if rng.random() < 0.15
                else base
                + pd.Timedelta(
                    minutes=int(
                        rng.integers(
                            0,
                            400,
                        )
                    )
                )
                for _ in range(
                    row_count
                )
            ]
        )

        actual = pit_asof_indices(
            left_event_time=left_event,
            left_cutoff=left_cutoff,
            right_event_time=right_event,
            right_available_at=(
                right_available
            ),
        ).tolist()

        expected = brute_force_selection(
            left_event,
            left_cutoff,
            right_event,
            right_available,
        )

        assert actual == expected


def test_unknown_right_availability_is_never_selected():
    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 05:00",
        ),
        left_cutoff=series(
            "2024-01-01 05:00",
        ),
        right_event_time=series(
            "2024-01-01 00:00",
            "2024-01-01 01:00",
        ),
        right_available_at=series(
            "2024-01-01 00:00",
            None,
        ),
    )

    assert selected.tolist() == [
        0,
    ]


def test_unknown_left_cutoff_selects_nothing():
    """
    A row that cannot establish its own cutoff cannot prove any
    observation was knowable to it.
    """

    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 05:00",
        ),
        left_cutoff=series(
            None,
        ),
        right_event_time=series(
            "2024-01-01 00:00",
        ),
        right_available_at=series(
            "2024-01-01 00:00",
        ),
    )

    assert selected.tolist() == [
        -1,
    ]


def test_derived_availability_matches_event_time_asof():
    """
    When availability is derived from event time, the
    two-constraint selection reduces to the classic backward
    asof join.
    """

    left_event = series(
        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
    )

    right_event = series(
        "2024-01-01 00:30",
        "2024-01-01 01:30",
    )

    with_availability = pit_asof_indices(
        left_event_time=left_event,
        left_cutoff=left_event,
        right_event_time=right_event,
        right_available_at=right_event,
    )

    event_time_only = pit_asof_indices(
        left_event_time=left_event,
        left_cutoff=left_event,
        right_event_time=right_event,
        right_available_at=None,
    )

    assert (
        with_availability.tolist()
        == event_time_only.tolist()
    )

    assert with_availability.tolist() == [
        -1,
        0,
        1,
    ]


def test_unsorted_right_event_time_is_rejected():
    with pytest.raises(
        FeatureAvailabilityError
    ):
        pit_asof_indices(
            left_event_time=series(
                "2024-01-01 05:00",
            ),
            left_cutoff=series(
                "2024-01-01 05:00",
            ),
            right_event_time=series(
                "2024-01-01 02:00",
                "2024-01-01 01:00",
            ),
            right_available_at=series(
                "2024-01-01 02:00",
                "2024-01-01 01:00",
            ),
        )


def test_selection_returns_positional_indices():
    selected = pit_asof_indices(
        left_event_time=series(
            "2024-01-01 05:00",
        ),
        left_cutoff=series(
            "2024-01-01 05:00",
        ),
        right_event_time=series(
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
        ),
        right_available_at=series(
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
        ),
    )

    assert isinstance(
        selected,
        np.ndarray,
    )

    assert selected.tolist() == [
        2,
    ]
