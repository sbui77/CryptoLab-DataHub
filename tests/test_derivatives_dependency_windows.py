"""
Independent verification that the QUALITY dependency window of each raw
input equals its VALUE dependency window.

Method
------
The two row sets are derived separately and never from each other:

    value-dependency rows
        perturb the numeric VALUE of one observation and record which
        output rows change

    quality-dependency rows
        degrade the available_at_quality of that SAME observation and
        record which output rows change label

Comparing the propagation constants against the availability constants
would be circular: both come from the same table. Perturbing the input
values is the only check that can catch a window that is wrong in both.

Floating-point note
-------------------
pandas aggregates rolling windows through a running accumulator, so an
early value leaves float residue in every later window even after it has
left the window. Measured on this fixture:

    largest residue OUTSIDE the true window   3.3e-13 (relative)
    smallest real change INSIDE it            3.6e-07 (relative)

A relative tolerance of 1e-9 sits ~360x above the residue and ~3000x
below the smallest real change, so residue cannot manufacture a false
dependency and no true dependency is missed. The fixture is fully
deterministic, so that margin does not drift between runs.
"""

import numpy as np
import pandas as pd

from cryptolab.features.derivatives import (
    BASIS_DEPENDENCY_BARS,
    LIQUIDATION_DEPENDENCY_BARS,
    OI_DEPENDENCY_BARS,
    TAKER_DEPENDENCY_BARS,
    build_derivatives_features,
)


BARS = 400

PERTURBED = 60

RELATIVE_TOLERANCE = 1e-9


def bar_times() -> pd.DatetimeIndex:
    return pd.date_range(
        "2024-01-01",
        periods=BARS,
        freq="5min",
        tz="UTC",
    )


TIMES = bar_times()


def make_open_interest(
    degraded: int | None = None,
    perturbed: int | None = None,
) -> pd.DataFrame:
    quality = np.array(
        ["exact"] * BARS,
        dtype=object,
    )

    quote = np.linspace(
        1_000_000.0,
        1_200_000.0,
        BARS,
    )

    if degraded is not None:
        quality[degraded] = "derived"

    if perturbed is not None:
        quote = quote.copy()
        quote[perturbed] += 3_333.0

    return pd.DataFrame(
        {
            "exchange": "binance",
            "market": "futures",
            "symbol": "BTCUSDT",
            "period": "5m",
            "timestamp": TIMES,
            "open_interest_base": np.linspace(
                100.0,
                120.0,
                BARS,
            ),
            "open_interest_quote": quote,
            "available_at": TIMES,
            "available_at_quality": quality,
        }
    )


def make_basis(
    degraded: int | None = None,
    perturbed: int | None = None,
) -> pd.DataFrame:
    quality = np.array(
        ["exact"] * BARS,
        dtype=object,
    )

    rate = np.linspace(
        0.001,
        0.002,
        BARS,
    )

    if degraded is not None:
        quality[degraded] = "derived"

    if perturbed is not None:
        rate = rate.copy()
        rate[perturbed] += 0.0007

    return pd.DataFrame(
        {
            "timestamp": TIMES,
            "basis": 1.0,
            "basis_rate": rate,
            "annualized_basis_rate": 0.1,
            "available_at": TIMES,
            "available_at_quality": quality,
        }
    )


def make_taker_flow(
    degraded: int | None = None,
    perturbed: int | None = None,
) -> pd.DataFrame:
    quality = np.array(
        ["exact"] * BARS,
        dtype=object,
    )

    buy = np.full(
        BARS,
        10.0,
    )

    if degraded is not None:
        quality[degraded] = "derived"

    if perturbed is not None:
        buy = buy.copy()
        buy[perturbed] += 4.0

    return pd.DataFrame(
        {
            "timestamp": TIMES,
            "buy_volume": buy,
            "sell_volume": 8.0,
            "buy_sell_ratio": 1.25,
            "available_at": TIMES,
            "available_at_quality": quality,
        }
    )


def make_funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": [
                TIMES[0],
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
                TIMES[0],
            ],
            "available_at_quality": [
                "exact",
            ],
        }
    )


def make_liquidations(
    at: int | None = None,
    degraded: bool = False,
    perturbed: bool = False,
) -> pd.DataFrame:
    if at is None:
        return pd.DataFrame(
            {
                "event_time": [],
                "liquidation_side": [],
                "liquidation_notional": [],
                "available_at": [],
                "available_at_quality": [],
            }
        )

    return pd.DataFrame(
        {
            "event_time": [
                TIMES[at],
            ],
            "liquidation_side": [
                "long",
            ],
            "liquidation_notional": [
                5_777.0
                if perturbed
                else 5_000.0
            ],
            "available_at": [
                TIMES[at],
            ],
            "available_at_quality": [
                "derived"
                if degraded
                else "exact"
            ],
        }
    )


def build(
    open_interest: pd.DataFrame | None = None,
    basis: pd.DataFrame | None = None,
    taker_flow: pd.DataFrame | None = None,
    liquidations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_derivatives_features(
        open_interest=(
            make_open_interest()
            if open_interest is None
            else open_interest
        ),
        funding_rate=make_funding(),
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
    )


def value_dependency_rows(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
) -> set[int]:
    """
    Rows whose feature VALUES respond to the perturbed observation.
    """

    rows: set[int] = set()

    for column in baseline.columns:
        if column in (
            "timestamp",
            "available_at",
            "available_at_quality",
        ):
            continue

        left = baseline[column]
        right = perturbed[column]

        if pd.api.types.is_numeric_dtype(
            left
        ):
            x = left.astype(float).to_numpy()
            y = right.astype(float).to_numpy()

            both_missing = (
                np.isnan(x)
                & np.isnan(y)
            )

            one_missing = (
                np.isnan(x)
                ^ np.isnan(y)
            )

            scale = np.maximum(
                np.maximum(
                    np.abs(x),
                    np.abs(y),
                ),
                1e-12,
            )

            differs = np.where(
                both_missing,
                False,
                np.where(
                    one_missing,
                    True,
                    np.abs(x - y)
                    > RELATIVE_TOLERANCE * scale,
                ),
            )

        else:
            differs = np.asarray(
                ~(
                    (left == right)
                    | (
                        left.isna()
                        & right.isna()
                    )
                )
            )

        rows |= set(
            int(index)
            for index in np.flatnonzero(
                differs
            )
        )

    return rows


def quality_dependency_rows(
    baseline: pd.DataFrame,
    degraded: pd.DataFrame,
) -> set[int]:
    """
    Rows whose published evidence quality responds to the degraded
    observation.
    """

    changed = (
        degraded["available_at_quality"]
        != baseline[
            "available_at_quality"
        ]
    )

    return set(
        int(index)
        for index in np.flatnonzero(
            changed.to_numpy()
        )
    )


def assert_windows_agree(
    name: str,
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
    degraded: pd.DataFrame,
    window: int,
) -> None:
    values = value_dependency_rows(
        baseline,
        perturbed,
    )

    quality = quality_dependency_rows(
        baseline,
        degraded,
    )

    expected = set(
        range(
            PERTURBED,
            PERTURBED + window,
        )
    )

    assert values == quality, (
        name,
        sorted(values - quality)[:8],
        sorted(quality - values)[:8],
    )

    assert values == expected, (
        name,
        min(values),
        max(values),
        len(values),
    )


def test_open_interest_dependency_is_289_observations():
    """
    oi_quote_change_pct_24h is pct_change(288), which consumes 289
    observations. An off-by-one window would drop the oldest one.
    """

    assert OI_DEPENDENCY_BARS == 289

    baseline = build()

    assert_windows_agree(
        "open_interest",
        baseline,
        build(
            open_interest=make_open_interest(
                perturbed=PERTURBED,
            )
        ),
        build(
            open_interest=make_open_interest(
                degraded=PERTURBED,
            )
        ),
        OI_DEPENDENCY_BARS,
    )


def test_basis_dependency_is_288_observations():
    """
    basis_zscore_24h is rolling(288).
    """

    assert BASIS_DEPENDENCY_BARS == 288

    baseline = build()

    assert_windows_agree(
        "basis",
        baseline,
        build(
            basis=make_basis(
                perturbed=PERTURBED,
            )
        ),
        build(
            basis=make_basis(
                degraded=PERTURBED,
            )
        ),
        BASIS_DEPENDENCY_BARS,
    )


def test_taker_flow_dependency_is_12_observations():
    """
    futures_taker_delta_1h is rolling(12).
    """

    assert TAKER_DEPENDENCY_BARS == 12

    baseline = build()

    assert_windows_agree(
        "taker_flow",
        baseline,
        build(
            taker_flow=make_taker_flow(
                perturbed=PERTURBED,
            )
        ),
        build(
            taker_flow=make_taker_flow(
                degraded=PERTURBED,
            )
        ),
        TAKER_DEPENDENCY_BARS,
    )


def test_liquidation_dependency_is_48_observations():
    """
    liquidation_notional_4h is rolling(48).
    """

    assert LIQUIDATION_DEPENDENCY_BARS == 48

    baseline = build(
        liquidations=make_liquidations(
            at=PERTURBED,
        )
    )

    assert_windows_agree(
        "liquidations",
        baseline,
        build(
            liquidations=make_liquidations(
                at=PERTURBED,
                perturbed=True,
            )
        ),
        build(
            liquidations=make_liquidations(
                at=PERTURBED,
                degraded=True,
            )
        ),
        LIQUIDATION_DEPENDENCY_BARS,
    )


def test_perturbation_probe_is_not_vacuous():
    """
    Guards the method itself: an unperturbed rebuild must produce an
    empty dependency set, otherwise the comparison above would pass
    for the wrong reason.
    """

    baseline = build()

    assert not value_dependency_rows(
        baseline,
        build(),
    )

    assert not quality_dependency_rows(
        baseline,
        build(),
    )
