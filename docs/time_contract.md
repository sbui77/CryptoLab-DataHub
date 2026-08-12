# CryptoLab Point-in-Time Time Contract

Version: 0.1

## 1. Purpose

The Point-in-Time Time Contract defines when an observation is
legitimately usable by research, feature engineering, and backtesting.

Its primary purpose is to prevent look-ahead bias.

---

## 2. Canonical timestamps

### event_time

The time represented by an observation or market event.

Current mappings:

- aggTrades: `trade_time`
- Open Interest: `timestamp`
- Funding Rate: `funding_time`
- Basis: `timestamp`
- Futures Taker Flow: `timestamp`
- Liquidations: `event_time`

`event_time` alone does not determine when a record may be used.

---

### available_at

The earliest time at which an observation could legitimately have
been known by a causal research process.

This is the canonical information-availability timestamp.

A record is usable iff:

```text
available_at <= as_of_time
```

An `available_at` of NaT means the availability of that
observation cannot be established. Such a record is never
provably usable and is excluded from point-in-time selection.
It is never silently replaced by `event_time`.

---

## 3. Feature-layer availability

Version: 0.2 (derivatives.core)

Raw ingestion establishes availability. The feature layer
propagates it. A curated feature artifact that carries no
`available_at` cannot be filtered causally, so the anti-lookahead
guarantee would stop at ingestion.

### Output rule

For `AvailabilityPolicy.MAX_INPUT_AVAILABLE_AT`:

```text
output available_at
    = max(available_at of the input observations that actually
          contributed to that output row)
```

Only observations genuinely consumed by a join or aggregation
count. Unmatched join candidates contribute nothing. A bar that
aggregated no events consumed no observation, so its zero is a
real zero and does not inherit availability from a neighbour.

If a consumed observation has unknown availability, the output
row's availability is unknown (NaT). Taking the max over the
known subset would understate it and overstate usability.

### Row semantics

A feature row describes market state at event-time T. It becomes
usable once every observation consumed by its outputs is
available.

Equality-joined observations *constitute* the row: there is no
alternative observation for bar T, so they determine when the row
becomes knowable.

As-of observations are *selected* from eligible historical
candidates against that row knowability.

### Event time is not an availability cutoff

An event timestamp and an availability cutoff are different
quantities and are never substituted for one another.

The cutoff is the row's own provisional knowability, computed
from the equality-joined inputs alone:

```text
provisional_available_at
    = max(windowed availability of the equality-joined inputs)
```

Funding is then selected against that cutoff, and:

```text
available_at = max(provisional_available_at,
                   selected_funding.available_at)
```

Because the selected funding availability cannot exceed the
cutoff it was selected against, this is a single-step fixpoint
rather than an iteration, and the result does not depend on join
order.

The spine availability alone is deliberately NOT used as the
cutoff. Doing so withheld funding that had already arrived before
the row became usable, publishing a row that claimed one
knowability time while using the information set of an earlier
one.

### Selection under two constraints

An as-of join has a genuine choice of which observation to use,
so both causal constraints apply:

```text
selected.event_time   <= row event_time             (event-time)
selected.available_at <= provisional_available_at   (availability)
```

The first preserves the meaning of an event-time-labelled bar: a
bar labelled T describes market state at T, so a later event
cannot enter it even when already knowable. The second prevents
point-in-time leakage: an observation that had not yet arrived
cannot influence the row.

No relationship is assumed between event order and availability
order. Gap repair legitimately produces an older event that
arrived after a newer one.

A candidate whose availability is unknown is individually
ineligible, because it can never be proven knowable. It does not
hide later candidates whose availability is known. The policy is
to use the latest provably-known eligible candidate rather than
to make the whole row unknown.

### Rolling and lagged features

A feature at row t computed from observations t-(w-1) .. t is not
computable until every one of those observations has arrived. Its
availability is the maximum over that window, not the
availability of row t alone.

Window sizing is the count of observations consumed:

```text
diff()                 -> 2
pct_change(periods=N)  -> N + 1
rolling(N)             -> N
```

Each input is widened to the longest window any output consumes
from it. For `derivatives.core`:

| Input | Observations | Driven by |
|---|---|---|
| Open Interest | 289 | `oi_quote_change_pct_24h` |
| Basis | 288 | `basis_zscore_24h` |
| Taker flow | 12 | `futures_taker_delta_1h` |
| Liquidations | 48 | `liquidation_notional_4h` |
| Funding | 1 | current row only |

An unknown observation makes the dependency unknown only for the
rows whose window actually contains it; outside that window it
must not poison the row.

Monotonic availability is NOT assumed and is not required for
correctness. An earlier guard that rejected non-decreasing
violations has been removed: late-arriving backfill is now
handled by widening the dependency window rather than by
rejecting the data.

### Availability evidence quality

Version: 0.3 (derivatives.core)

`available_at_quality` records the evidence behind an
availability timestamp, and nothing else. It is not data
completeness, not a feature quality score, and not confidence in
the numeric value of a feature.

```text
exact    directly observed by CryptoLab
derived  deterministically inferred from source semantics
unknown  cannot be reconstructed reliably
```

### Weakest-link propagation

```text
output available_at_quality
    = worst quality among the observations actually consumed by
      that output row

exact < derived < unknown
```

Combination is a maximum over that ordering, so it is
associative, commutative and idempotent, and therefore
independent of join order.

The quality of the observation that happens to determine
`max(available_at)` is deliberately NOT used. Given

```text
A: available_at = 10:10, quality = exact
B: available_at = 10:05, quality = derived
```

the output is `available_at = 10:10` with quality `derived`. The
output timestamp is a function of every consumed observation:
had B arrived later than its derived estimate, the true output
availability would move. Calling it exact would claim direct
observation of a bound that partly rests on inference.

### Strict pair

`available_at` and `available_at_quality` are one contract. A
migrated artifact emits both or neither, and

```text
available_at_quality == "unknown"   iff   available_at is NaT
```

holds on inputs and on outputs. A frame carrying exactly one of
the two columns is partial metadata and is rejected: an
availability timestamp with no recorded evidence quality cannot
be distinguished from an exact one. Legacy inputs
(`require_availability=False`) emit neither column rather than a
fabricated one.

`require_availability=False` switches off *propagation*, not
*validation*. Metadata that is present on an input must still be
internally valid in every mode; only the output contract is
withheld. Consuming corrupt metadata silently would carry it
into a later migration unnoticed.

Malformed availability metadata reaches the caller of
`derivatives.core` as `DerivativesFeatureError` — partial pair,
invalid label, or invariant violation alike — with the
underlying `FeatureAvailabilityError` preserved as the cause.

### Combining in stages

The combination is associative, but "consumed nothing" is
collapsed to `unknown` (and to `NaT` on the availability side)
at the boundary of the combinator. An intermediate result must
therefore not be re-injected as a component with
`consumed=True` on rows where nothing was consumed: that turns
*no dependency* into *unknown dependency*, and the staged result
stops matching the flat combination. Fold only where at least
one component is genuinely consumed, or carry the real consumed
mask through the fold.

### Windows, as-of selection and filtering

Quality travels over exactly the same consumed observations as
availability:

- rolling and lagged dependencies take the worst quality over
  the same trailing window, and an unknown observation outside
  that window does not poison the row;
- an unmatched join candidate contributes nothing, whatever its
  quality;
- the as-of funding candidate that is actually selected
  contributes its own quality; candidates passed over contribute
  nothing.

Quality is audit and evidence metadata, not a causal filter. A
row with a known `available_at` and quality `derived` stays
usable under the canonical rule `available_at <= as_of_time`.
`point_in_time_filter` is unchanged.

### Not yet defined

Whether a *fallback* — an as-of selection that skipped a newer
candidate because that candidate's availability was unprovable —
needs its own column. Today such a row is indistinguishable from
one where no newer candidate existed. This is recorded as
migration debt, not addressed here.
