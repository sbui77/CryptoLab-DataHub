from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureAvailabilityError(RuntimeError):
    """Raised when feature availability semantics are violated."""


AVAILABLE_AT = "available_at"


# Bound on the boolean work matrix used by point-in-time
# selection. Rows are processed in chunks so memory stays flat
# regardless of candidate cardinality.
_SELECTION_CELL_BUDGET = 4_000_000


def _naive_utc_values(
    values: pd.Series,
) -> np.ndarray:
    """
    Return UTC-normalized, timezone-naive datetime64 values.

    Comparison and searchsorted require a single consistent
    representation on both sides of a join.
    """

    normalized = pd.to_datetime(
        values,
        utc=True,
        errors="coerce",
    )

    return (
        normalized
        .dt.tz_convert(None)
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )


def has_availability(
    df: pd.DataFrame,
) -> bool:
    """
    Return True when a frame carries the availability column.
    """

    return AVAILABLE_AT in df.columns


def extract_availability(
    df: pd.DataFrame,
    *,
    label: str,
) -> pd.Series:
    """
    Return the availability series of an input frame as UTC.

    Missing availability is never fabricated.

    A NaT value is preserved as NaT and means:

        the time at which this observation became knowable
        cannot be established

    It is deliberately NOT replaced by event time. The raw
    storage layer already encodes that case explicitly as

        available_at_quality = "unknown"
        available_at         = NaT
    """

    if not has_availability(df):
        raise FeatureAvailabilityError(
            "Missing available_at for input: "
            f"{label}"
        )

    return pd.to_datetime(
        df[AVAILABLE_AT],
        utc=True,
        errors="coerce",
    )


def combine_availability(
    components: list[
        tuple[
            pd.Series,
            pd.Series,
        ]
    ],
) -> pd.Series:
    """
    Combine consumed input availability into output availability.

    Canonical rule (AvailabilityPolicy.MAX_INPUT_AVAILABLE_AT):

        output available_at
            = max(available_at of the input observations that
                  actually contributed to that output row)

    Each component is a pair:

        (availability values, consumed mask)

    consumed = False
        The observation did not contribute to that row. It is
        ignored entirely, so unmatched join candidates cannot
        influence output availability.

    consumed = True with NaT availability
        The observation contributed but its availability is
        unknown. The output row therefore has unknown
        availability and is emitted as NaT rather than being
        silently downgraded to a known-looking timestamp.
    """

    if not components:
        raise FeatureAvailabilityError(
            "At least one availability component is required"
        )

    effective: list[pd.Series] = []

    poisoned = pd.Series(
        False,
        index=components[0][0].index,
    )

    for (
        values,
        consumed,
    ) in components:
        aligned = pd.to_datetime(
            values,
            utc=True,
            errors="coerce",
        )

        consumed_mask = consumed.astype(
            bool
        )

        poisoned = poisoned | (
            consumed_mask
            & aligned.isna()
        )

        effective.append(
            aligned.where(
                consumed_mask
            )
        )

    frame = pd.concat(
        effective,
        axis=1,
    )

    combined = frame.max(
        axis=1,
        skipna=True,
    )

    combined = combined.where(
        ~poisoned,
        pd.NaT,
    )

    return pd.to_datetime(
        combined,
        utc=True,
    )


def rolling_dependency_availability(
    values: pd.Series,
    consumed: pd.Series,
    window: int,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """
    Extend an availability component across the trailing window
    of observations a rolling or lagged feature consumes.

    A feature at row t computed from observations
    t-(window-1) .. t is not computable until every one of those
    observations has arrived. Its availability is therefore the
    maximum availability over that window, not the availability
    of row t alone.

    Window sizing
    -------------
    window is the COUNT of observations consumed:

        diff()                 -> 2
        pct_change(periods=N)  -> N + 1
        rolling(N)             -> N

    Semantics preserved from combine_availability
    ---------------------------------------------
    An observation that was not consumed contributes nothing,
    even if it falls inside the window.

    A consumed observation with unknown availability makes the
    whole dependency unknown, but only for the rows whose window
    actually contains it. An unknown observation outside the
    window must not poison the row.

    Returns
    -------
    (windowed availability, windowed consumed mask) suitable for
    passing straight to combine_availability.
    """

    if window < 1:
        raise FeatureAvailabilityError(
            "Rolling dependency window must be >= 1"
        )

    normalized = pd.to_datetime(
        values,
        utc=True,
        errors="coerce",
    )

    consumed_mask = consumed.astype(
        bool
    )

    poison = (
        consumed_mask
        & normalized.isna()
    )

    effective = normalized.where(
        consumed_mask
        & normalized.notna()
    )

    # Rank-encode before the rolling max. pandas rolling
    # aggregates through float64, which cannot represent
    # nanosecond epoch values exactly, so raw timestamps would
    # be silently rounded.
    codes = pd.Series(
        -1,
        index=normalized.index,
        dtype="int64",
    )

    present = effective.notna()

    uniques = np.array(
        [],
        dtype="int64",
    )

    if present.any():
        raw = _naive_utc_values(
            effective[present]
        ).view("int64")

        uniques = np.unique(
            raw
        )

        codes.loc[
            present
        ] = np.searchsorted(
            uniques,
            raw,
        )

    rolled_codes = (
        codes
        .rolling(
            window,
            min_periods=1,
        )
        .max()
    )

    rolled_consumed = (
        consumed_mask
        .astype("int64")
        .rolling(
            window,
            min_periods=1,
        )
        .max()
        > 0
    )

    rolled_poison = (
        poison
        .astype("int64")
        .rolling(
            window,
            min_periods=1,
        )
        .max()
        > 0
    )

    result = pd.Series(
        pd.NaT,
        index=normalized.index,
        dtype="datetime64[ns, UTC]",
    )

    resolvable = (
        rolled_codes.notna()
        & (rolled_codes >= 0)
    )

    if resolvable.any():
        selected = uniques[
            rolled_codes[resolvable]
            .astype("int64")
            .to_numpy()
        ]

        result.loc[
            resolvable
        ] = pd.to_datetime(
            selected,
            utc=True,
        )

    result = result.where(
        ~rolled_poison,
        pd.NaT,
    )

    return (
        result,
        rolled_consumed,
    )


def pit_asof_indices(
    left_event_time: pd.Series,
    left_cutoff: pd.Series,
    right_event_time: pd.Series,
    right_available_at: pd.Series | None,
) -> np.ndarray:
    """
    Select the latest right-hand observation usable by each row
    under BOTH causal constraints.

    Constraint 1 — event-time causality
    -----------------------------------
        right event_time <= left event_time

    This preserves the meaning of an event-time-labelled bar.
    A bar labelled T must describe market state at T, so an
    event occurring after T can never enter it even if it was
    already knowable.

    Constraint 2 — availability causality
    -------------------------------------
        right available_at <= left cutoff

    This prevents point-in-time leakage. An observation that had
    not yet arrived cannot influence the row.

    The two constraints are distinct and are deliberately not
    collapsed into one another: the left event time is an event
    timestamp, NOT an availability cutoff.

    Availability ordering
    ---------------------
    No monotonic relationship between event order and
    availability order is assumed. Gap repair legitimately
    produces an older event that arrived after a newer one.

    Unknown availability
    --------------------
    A candidate whose availability is unknown can never be
    proven knowable and is individually ineligible. It does NOT
    hide later candidates whose availability is known.

    Returns positional indices into the right frame, or -1 where
    no observation satisfies both constraints.

    Selection is a direct transcription of the definition above,
    evaluated over row chunks. Candidate cardinality is small
    (funding settles a few times per day), so this is both fast
    enough and simple enough to verify by inspection.
    """

    event_values = _naive_utc_values(
        left_event_time
    )

    right_event_values = _naive_utc_values(
        right_event_time
    )

    if not pd.Index(
        right_event_values
    ).is_monotonic_increasing:
        raise FeatureAvailabilityError(
            "Right-hand event_time must be sorted "
            "ascending for point-in-time selection"
        )

    by_event = np.searchsorted(
        right_event_values,
        event_values,
        side="right",
    )

    if right_available_at is None:
        return by_event - 1

    availability = pd.to_datetime(
        right_available_at,
        utc=True,
        errors="coerce",
    )

    known = (
        availability
        .notna()
        .to_numpy()
    )

    availability_values = (
        _naive_utc_values(
            availability
        )
        .view("int64")
    )

    cutoff = pd.to_datetime(
        left_cutoff,
        utc=True,
        errors="coerce",
    )

    cutoff_known = (
        cutoff
        .notna()
        .to_numpy()
    )

    cutoff_values = (
        _naive_utc_values(
            cutoff
        )
        .view("int64")
    )

    row_count = len(
        event_values
    )

    candidate_count = len(
        availability_values
    )

    selected = np.full(
        row_count,
        -1,
        dtype=np.int64,
    )

    if (
        candidate_count == 0
        or row_count == 0
    ):
        return selected

    positions = np.arange(
        candidate_count
    )

    chunk = max(
        1,
        _SELECTION_CELL_BUDGET
        // candidate_count,
    )

    for start in range(
        0,
        row_count,
        chunk,
    ):
        stop = min(
            start + chunk,
            row_count,
        )

        eligible = (
            (
                availability_values[
                    None,
                    :,
                ]
                <= cutoff_values[
                    start:stop,
                    None,
                ]
            )
            & (
                positions[
                    None,
                    :,
                ]
                < by_event[
                    start:stop,
                    None,
                ]
            )
            & known[
                None,
                :,
            ]
            & cutoff_known[
                start:stop,
                None,
            ]
        )

        any_eligible = eligible.any(
            axis=1
        )

        # Rightmost eligible candidate is the latest by event
        # order, because the right frame is sorted by event
        # time.
        last_eligible = (
            candidate_count
            - 1
            - eligible[
                :,
                ::-1,
            ].argmax(
                axis=1
            )
        )

        selected[
            start:stop
        ] = np.where(
            any_eligible,
            last_eligible,
            -1,
        )

    return selected
