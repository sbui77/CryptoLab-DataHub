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
