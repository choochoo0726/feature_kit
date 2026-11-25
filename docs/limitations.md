# Known limitations

Behaviour to account for when using the signals for research or trading.

## Look-ahead risk

### Pooled winsorization

`WinsorizeTransformation` is point-in-time safe by default (quantiles per date). The opt-in `scope="pooled"` computes quantiles over the whole sample, so in a backtest a 2022 value is clipped with bounds that depend on 2024 data. See [Transformations](guide/transformations.md#clip-vs-winsorize).

### Point-in-time correctness of the input

Every component treats `date_col` as the date on which the value was known. If your vendor restates or backfills history, use a point-in-time snapshot, or the signals will look better in backtests than they will live.

## Windows

### Missing days count as zero in rolling sums

Base-feature windows cover exact calendar days, and a missing day contributes **0** to the sum (up to the `min_coverage` threshold, 20% missing by default). For a metric where a missing day means "no data" rather than "no activity", the sum understates the true total by up to that share. Raise `min_coverage` (e.g. `1.0`) if that matters. See [Base features](guide/base-features.md#calendar-windows-and-coverage).

### Comparison lookbacks count observations

`ZScoreComparison`, `RankComparison` and `ResidualComparison` use the entity's **last W rows**, not the last W calendar days. With complete daily data the two are the same. With gaps, the lookback spans more calendar time.

## Benchmarks

- **Cohort includes the entity itself by default.** Pass `exclude_self=True` to `CohortRollingSumBenchmark` for a peers-only (leave-one-out) benchmark.
- **`rolling_rank` is marked unstable in Polars.** `RankComparison` relies on it. A regression test pins its values, so a behaviour change in a future Polars release would show up as a test failure.
