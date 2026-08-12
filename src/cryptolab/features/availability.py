from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureAvailabilityError(RuntimeError):
    """Raised when feature availability semantics are violated."""


AVAILABLE_AT = "available_at"

AVAILABLE_AT_QUALITY = "available_at_quality"


QUALITY_EXACT = "exact"

QUALITY_DERIVED = "derived"

QUALITY_UNKNOWN = "unknown"


# Weakest-link ordering of timestamp evidence:
#
#     exact < derived < unknown
#
# available_at_quality describes the EVIDENCE behind the
# availability timestamp and nothing else. It is not data
# completeness, not a feature quality score, and not confidence
# in the numeric value of the feature.
QUALITY_ORDER: tuple[
    str,
    ...
] = (
    QUALITY_EXACT,
    QUALITY_DERIVED,
    QUALITY_UNKNOWN,
)

VALID_QUALITIES = frozenset(
    QUALITY_ORDER
)


_QUALITY_RANK: dict[
    str,
    int,
] = {
    quality: rank
    for rank, quality in enumerate(
        QUALITY_ORDER
    )
}

_UNKNOWN_RANK = float(
    _QUALITY_RANK[
        QUALITY_UNKNOWN
    ]
)


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


def has_availability_quality(
    df: pd.DataFrame,
) -> bool:
    """
    Return True when a frame carries the availability quality
    column.
    """

    return (
        AVAILABLE_AT_QUALITY
        in df.columns
    )


def availability_pair_state(
    df: pd.DataFrame,
    *,
    label: str,
) -> bool:
    """
    Return whether a frame participates in the availability
    contract.

    The contract is a strict pair:

        available_at
        available_at_quality

    True
        Both columns are present.

    False
        Neither column is present. The frame predates the
        contract and nothing is fabricated for it.

    A frame carrying exactly one of the two columns is partial
    metadata. It is rejected rather than silently downgraded,
    because a timestamp with no recorded evidence quality cannot
    be told apart from an exact one.
    """

    availability = has_availability(
        df
    )

    quality = has_availability_quality(
        df
    )

    if availability == quality:
        return availability

    present = (
        AVAILABLE_AT
        if availability
        else AVAILABLE_AT_QUALITY
    )

    missing = (
        AVAILABLE_AT_QUALITY
        if availability
        else AVAILABLE_AT
    )

    raise FeatureAvailabilityError(
        "Partial point-in-time availability metadata for "
        f"{label}: {present} is present but {missing} is "
        "missing"
    )


def quality_to_rank(
    values: pd.Series,
) -> pd.Series:
    """
    Encode quality labels as weakest-link ranks.

    Missing labels encode as NaN, which means "no evidence
    recorded here" rather than any particular quality. Callers
    decide what an absent label means in their context.
    """

    normalized = (
        pd.Series(
            values
        )
        .astype("string")
        .str.strip()
        .str.lower()
    )

    unrecognized = (
        normalized.notna()
        & ~normalized.isin(
            VALID_QUALITIES
        )
    )

    if unrecognized.any():
        invalid = sorted(
            set(
                normalized[
                    unrecognized
                ].tolist()
            )
        )

        raise FeatureAvailabilityError(
            "Invalid available_at_quality values: "
            f"{invalid}"
        )

    ranks = pd.Series(
        np.nan,
        index=normalized.index,
        dtype="float64",
    )

    for (
        quality,
        rank,
    ) in _QUALITY_RANK.items():
        selected = (
            normalized
            .eq(quality)
            .fillna(False)
            .astype(bool)
        )

        ranks.loc[
            selected
        ] = float(
            rank
        )

    return ranks


def rank_to_quality(
    ranks: pd.Series,
) -> pd.Series:
    """
    Decode weakest-link ranks back into quality labels.

    NaN decodes to pd.NA, preserving "no evidence recorded".
    """

    result = pd.Series(
        pd.NA,
        index=ranks.index,
        dtype="string",
    )

    for (
        quality,
        rank,
    ) in _QUALITY_RANK.items():
        selected = (
            ranks
            .eq(float(rank))
            .fillna(False)
            .astype(bool)
        )

        result.loc[
            selected
        ] = quality

    unmapped = (
        ranks.notna()
        & result.isna()
    )

    if unmapped.any():
        raise FeatureAvailabilityError(
            "Unmappable availability quality ranks: "
            f"{sorted(set(ranks[unmapped].tolist()))}"
        )

    return result


def extract_availability_quality(
    df: pd.DataFrame,
    *,
    label: str,
    availability: pd.Series | None = None,
) -> pd.Series:
    """
    Return the availability quality series of an input frame.

    Missing quality is never fabricated: a frame that declares
    the column must populate it on every row.

    Pair invariant
    --------------
    Availability and its evidence quality are two views of one
    fact, so they must agree:

        available_at_quality == "unknown"
            iff
        available_at is NaT

    Both directions are enforced. A known timestamp labelled
    unknown would be discarded by point-in-time selection that
    trusted the label, and an unknown timestamp labelled exact
    would claim evidence that does not exist.
    """

    if not has_availability_quality(
        df
    ):
        raise FeatureAvailabilityError(
            "Missing available_at_quality for input: "
            f"{label}"
        )

    values = (
        df[AVAILABLE_AT_QUALITY]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    invalid = (
        values.isna()
        | ~values.isin(
            VALID_QUALITIES
        )
    )

    if invalid.any():
        raise FeatureAvailabilityError(
            "Invalid available_at_quality for input "
            f"{label}: "
            f"{sorted(set(values[invalid].astype(object)))}"
        )

    if (
        availability is None
        and has_availability(
            df
        )
    ):
        availability = extract_availability(
            df,
            label=label,
        )

    if availability is not None:
        unknown = values.eq(
            QUALITY_UNKNOWN
        )

        missing = availability.isna()

        if (
            unknown
            & ~missing
        ).any():
            raise FeatureAvailabilityError(
                "available_at_quality='unknown' requires "
                f"available_at=NaT for input: {label}"
            )

        if (
            ~unknown
            & missing
        ).any():
            raise FeatureAvailabilityError(
                "available_at=NaT requires "
                "available_at_quality='unknown' for input: "
                f"{label}"
            )

    return values


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


def combine_availability_quality(
    components: list[
        tuple[
            pd.Series,
            pd.Series,
        ]
    ],
) -> pd.Series:
    """
    Combine consumed input quality into output quality.

    Canonical rule (weakest link):

        output available_at_quality
            = worst quality among the observations that actually
              contributed to that output row

    with

        exact < derived < unknown

    Combination is a maximum over ranks, so it is associative,
    commutative and idempotent, and therefore independent of
    join order.

    The quality of the observation that happens to determine
    max(available_at) is deliberately NOT used. The output
    timestamp is a function of every consumed observation, so it
    cannot be called exact while any of them was merely derived.

    Each component is a pair:

        (quality values, consumed mask)

    consumed = False
        The observation did not contribute, so it is ignored
        entirely — including when its quality is unknown.

    consumed = True with no recorded quality
        Contributing with no evidence at all is exactly what
        unknown means, so the weakest link takes it as unknown.

    A row that consumed nothing has no established availability,
    so its quality is unknown. This mirrors the NaT that
    combine_availability produces for the same row.

    Combining in stages
    -------------------
    The binary operation is associative, but the "consumed
    nothing" outcome is collapsed to unknown at the boundary of
    this function. An intermediate result must therefore NOT be
    re-injected as a component with consumed=True on rows where
    nothing was actually consumed: that turns "no dependency"
    into "unknown dependency" and the staged result stops
    matching the single flat combination.

    combine_availability has the identical property, since its
    "consumed nothing" outcome collapses to NaT and poisons the
    row when re-injected as consumed. Fold only where at least
    one component is genuinely consumed, or carry the real
    consumed mask through the fold.
    """

    if not components:
        raise FeatureAvailabilityError(
            "At least one quality component is required"
        )

    effective: list[pd.Series] = []

    for (
        values,
        consumed,
    ) in components:
        ranks = quality_to_rank(
            values
        )

        consumed_mask = consumed.astype(
            bool
        )

        ranks = ranks.where(
            ~(
                consumed_mask
                & ranks.isna()
            ),
            _UNKNOWN_RANK,
        )

        effective.append(
            ranks.where(
                consumed_mask
            )
        )

    frame = pd.concat(
        effective,
        axis=1,
    )

    worst = frame.max(
        axis=1,
        skipna=True,
    )

    worst = worst.fillna(
        _UNKNOWN_RANK
    )

    return rank_to_quality(
        worst
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


def rolling_dependency_quality(
    values: pd.Series,
    consumed: pd.Series,
    window: int,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """
    Extend a quality component across the trailing window of
    observations a rolling or lagged feature consumes.

    Quality travels over exactly the same window as availability,
    because it describes the evidence behind exactly the same
    consumed observations. A feature at row t computed from
    observations t-(window-1) .. t carries the worst quality over
    that window.

    Semantics preserved from combine_availability_quality
    -----------------------------------------------------
    An observation that was not consumed contributes nothing,
    even if it falls inside the window.

    An unknown observation degrades the dependency only for the
    rows whose window actually contains it. Outside the window it
    must not poison the row.

    Returns
    -------
    (windowed quality, windowed consumed mask) suitable for
    passing straight to combine_availability_quality.
    """

    if window < 1:
        raise FeatureAvailabilityError(
            "Rolling dependency window must be >= 1"
        )

    ranks = quality_to_rank(
        values
    )

    consumed_mask = consumed.astype(
        bool
    )

    ranks = ranks.where(
        ~(
            consumed_mask
            & ranks.isna()
        ),
        _UNKNOWN_RANK,
    )

    effective = ranks.where(
        consumed_mask
    )

    # Ranks are small integers, so the float64 path pandas uses
    # for rolling aggregation represents them exactly.
    rolled = (
        effective
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

    return (
        rank_to_quality(
            rolled
        ),
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
