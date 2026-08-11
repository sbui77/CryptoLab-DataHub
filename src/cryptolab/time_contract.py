from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pandas as pd


class TimeContractError(RuntimeError):
    """Raised when point-in-time semantics are violated."""


class TimeQuality(str, Enum):
    """
    Evidence quality of a timestamp.

    EXACT
        Directly observed by CryptoLab.

    DERIVED
        Deterministically inferred from source semantics.

    UNKNOWN
        Cannot be reconstructed reliably.
    """

    EXACT = "exact"
    DERIVED = "derived"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TimeContract:
    """
    Canonical point-in-time metadata contract.
    """

    event_time_column: str
    available_at_column: str = "available_at"
    ingested_at_column: str = "ingested_at"


@dataclass(frozen=True)
class HistoricalTimePolicy:
    """
    Historical migration policy for one dataset.
    """

    dataset: str
    event_time_column: str
    availability_quality: TimeQuality
    ingestion_quality: TimeQuality


HISTORICAL_TIME_POLICIES: dict[
    str,
    HistoricalTimePolicy,
] = {
    "aggtrades": HistoricalTimePolicy(
        dataset="aggtrades",
        event_time_column="trade_time",
        availability_quality=TimeQuality.DERIVED,
        ingestion_quality=TimeQuality.UNKNOWN,
    ),
    "open_interest": HistoricalTimePolicy(
        dataset="open_interest",
        event_time_column="timestamp",
        availability_quality=TimeQuality.DERIVED,
        ingestion_quality=TimeQuality.UNKNOWN,
    ),
    "funding": HistoricalTimePolicy(
        dataset="funding",
        event_time_column="funding_time",
        availability_quality=TimeQuality.DERIVED,
        ingestion_quality=TimeQuality.UNKNOWN,
    ),
    "basis": HistoricalTimePolicy(
        dataset="basis",
        event_time_column="timestamp",
        availability_quality=TimeQuality.DERIVED,
        ingestion_quality=TimeQuality.UNKNOWN,
    ),
    "taker_flow": HistoricalTimePolicy(
        dataset="taker_flow",
        event_time_column="timestamp",
        availability_quality=TimeQuality.DERIVED,
        ingestion_quality=TimeQuality.UNKNOWN,
    ),
    "liquidations": HistoricalTimePolicy(
        dataset="liquidations",
        event_time_column="event_time",
        availability_quality=TimeQuality.DERIVED,
        ingestion_quality=TimeQuality.UNKNOWN,
    ),
}


def utc_now() -> pd.Timestamp:
    """
    Return current UTC time.

    Centralized so runtime timestamp generation can be
    controlled and tested consistently.
    """

    return pd.Timestamp.now(
        tz="UTC"
    )


def ensure_utc_timestamp(
    value: object,
) -> pd.Timestamp:
    """
    Normalize timestamp-like value to UTC.
    """

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize(
            "UTC"
        )

    return ts.tz_convert(
        "UTC"
    )


def get_historical_time_policy(
    dataset: str,
) -> HistoricalTimePolicy:
    """
    Return registered historical migration policy.
    """

    key = dataset.strip().lower()

    if key not in HISTORICAL_TIME_POLICIES:
        raise TimeContractError(
            "Unknown historical time policy: "
            f"{dataset}"
        )

    return HISTORICAL_TIME_POLICIES[
        key
    ]


def normalize_time_columns(
    df: pd.DataFrame,
    contract: TimeContract,
) -> pd.DataFrame:
    """
    Normalize canonical time-contract columns to UTC.

    Missing metadata is never fabricated here.
    """

    if df.empty:
        return df.copy()

    result = df.copy()

    required = [
        contract.event_time_column,
        contract.available_at_column,
        contract.ingested_at_column,
    ]

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise TimeContractError(
            "Missing time contract columns: "
            f"{missing}"
        )

    for column in required:
        result[column] = pd.to_datetime(
            result[column],
            utc=True,
            errors="coerce",
        )

        if result[column].isna().any():
            raise TimeContractError(
                "Invalid timestamp values in "
                f"{column}"
            )

    return result


def validate_time_contract(
    df: pd.DataFrame,
    contract: TimeContract,
    allow_ingest_before_available: bool = False,
) -> None:
    """
    Validate point-in-time invariants.

    event_time <= available_at is deliberately NOT
    imposed as a universal invariant.

    Normally:

        ingested_at >= available_at
    """

    if df.empty:
        return

    work = normalize_time_columns(
        df,
        contract,
    )

    if not allow_ingest_before_available:
        invalid = (
            work[
                contract.ingested_at_column
            ]
            < work[
                contract.available_at_column
            ]
        )

        if invalid.any():
            raise TimeContractError(
                "ingested_at occurs before "
                "available_at for "
                f"{int(invalid.sum())} rows"
            )


def derive_historical_time_metadata(
    df: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """
    Attach explicit historical point-in-time metadata.

    Historical ingested_at is never invented.

    available_at
        Derived from source event timestamp.

    ingested_at
        Unknown for legacy observations.
    """

    policy = get_historical_time_policy(
        dataset
    )

    result = df.copy()

    if policy.event_time_column not in result.columns:
        raise TimeContractError(
            "Missing event-time column "
            f"{policy.event_time_column} "
            f"for dataset={dataset}"
        )

    event_time = pd.to_datetime(
        result[
            policy.event_time_column
        ],
        utc=True,
        errors="coerce",
    )

    if event_time.isna().any():
        raise TimeContractError(
            "Invalid historical event timestamps "
            f"for dataset={dataset}"
        )

    result[
        "available_at"
    ] = event_time

    result[
        "available_at_quality"
    ] = (
        policy.availability_quality.value
    )

    result[
        "ingested_at"
    ] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    result[
        "ingested_at_quality"
    ] = (
        policy.ingestion_quality.value
    )

    return result


def attach_runtime_time_metadata(
    df: pd.DataFrame,
    event_time_column: str,
    *,
    available_at: object | None = None,
    ingested_at: object | None = None,
    available_at_from_event: bool = True,
    clock: Callable[[], pd.Timestamp] = utc_now,
) -> pd.DataFrame:
    """
    Attach point-in-time metadata to newly ingested data.

    This function is intended for NEW runtime observations,
    not historical migration.

    Default REST semantics
    ----------------------
    available_at
        Derived from event time.

    available_at_quality
        "derived"

    ingested_at
        Actual CryptoLab runtime observation time.

    ingested_at_quality
        "exact"

    WebSocket semantics
    -------------------
    A caller may explicitly provide available_at equal to
    the collector receive timestamp. In that case:

        available_at_quality = "exact"

    Important
    ---------
    One ingested_at timestamp is assigned to the entire
    DataFrame/batch. This represents the observation time
    of that API response or received message.
    """

    result = df.copy()

    if event_time_column not in result.columns:
        raise TimeContractError(
            "Missing runtime event-time column: "
            f"{event_time_column}"
        )

    event_time = pd.to_datetime(
        result[event_time_column],
        utc=True,
        errors="coerce",
    )

    if event_time.isna().any():
        raise TimeContractError(
            "Invalid runtime event timestamps in "
            f"{event_time_column}"
        )

    if available_at is not None:
        available_timestamp = (
            ensure_utc_timestamp(
                available_at
            )
        )

        result[
            "available_at"
        ] = available_timestamp

        result[
            "available_at_quality"
        ] = TimeQuality.EXACT.value

    elif available_at_from_event:
        result[
            "available_at"
        ] = event_time

        result[
            "available_at_quality"
        ] = TimeQuality.DERIVED.value

    else:
        result[
            "available_at"
        ] = pd.Series(
            pd.NaT,
            index=result.index,
            dtype="datetime64[ns, UTC]",
        )

        result[
            "available_at_quality"
        ] = TimeQuality.UNKNOWN.value

    if ingested_at is None:
        ingestion_timestamp = (
            ensure_utc_timestamp(
                clock()
            )
        )
    else:
        ingestion_timestamp = (
            ensure_utc_timestamp(
                ingested_at
            )
        )

    result[
        "ingested_at"
    ] = ingestion_timestamp

    result[
        "ingested_at_quality"
    ] = TimeQuality.EXACT.value

    return result


def point_in_time_filter(
    df: pd.DataFrame,
    as_of_time: object,
    available_at_column: str = "available_at",
) -> pd.DataFrame:
    """
    Return observations known at or before as_of_time.

    Canonical anti-lookahead rule:

        available_at <= as_of_time
    """

    if df.empty:
        return df.copy()

    if available_at_column not in df.columns:
        raise TimeContractError(
            f"Missing {available_at_column}"
        )

    as_of = ensure_utc_timestamp(
        as_of_time
    )

    available = pd.to_datetime(
        df[
            available_at_column
        ],
        utc=True,
        errors="coerce",
    )

    if available.isna().any():
        raise TimeContractError(
            "Invalid available_at values"
        )

    result = df.loc[
        available <= as_of
    ].copy()

    return result.reset_index(
        drop=True
    )


def latest_point_in_time(
    df: pd.DataFrame,
    as_of_time: object,
    event_time_column: str,
    available_at_column: str = "available_at",
) -> pd.Series | None:
    """
    Return latest event actually knowable by as_of_time.
    """

    filtered = point_in_time_filter(
        df=df,
        as_of_time=as_of_time,
        available_at_column=available_at_column,
    )

    if filtered.empty:
        return None

    if event_time_column not in filtered.columns:
        raise TimeContractError(
            f"Missing {event_time_column}"
        )

    event_time = pd.to_datetime(
        filtered[
            event_time_column
        ],
        utc=True,
        errors="coerce",
    )

    if event_time.isna().any():
        raise TimeContractError(
            "Invalid event_time values"
        )

    index = (
        event_time
        .sort_values()
        .index[-1]
    )

    return filtered.loc[
        index
    ]
