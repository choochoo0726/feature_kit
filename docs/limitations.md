# Known limitations

The current behaviour, verified against the code, that you should account for when using signals for research or trading. Each item is a candidate for a future fix.

## Look-ahead risk

### Pooled winsorization

`WinsorizeTransformation` is point-in-time safe by default (quantiles per date). The opt-in `scope="pooled"` computes quantiles over the whole sample, so in a backtest a 2022 value is clipped with bounds that depend on 2024 data. See [Transformations](guide/transformations.md#clip-vs-winsorize).

### Point-in-time correctness of the input

Every component treats `date_col` as the date on which the value was known. If your vendor restates or backfills history, use a point-in-time snapshot, or the signals will look better in backtests than they will live.

## Data layout

### Rolling windows count rows, not days

Rolling sums, z-scores, ranks and residuals use the **last N rows** of each entity. Lags and YoY/PoP offsets match **exact calendar dates**. With a missing day, a "28-day" sum covers 29 calendar days while lags remain calendar-exact. Reindex each entity to a complete daily calendar before computing.

### Standalone transformations default to `specid` and `DATADATE`

A transformation used outside the builder without `params=` assumes the key and date columns are `specid` and `DATADATE`. With other names, pass `params`, or it will treat your key columns as features (clip fails on string keys, and numeric keys are silently modified). Inside `FeatureBuilder` the builder's `params` is used automatically.

## Pairing and output size

### Unprefixed benchmarks pair with every metric

`ZERO_BENCHMARK` and the IBES columns have no metric prefix, so with `metrics=["SALES", "TXN"]` the IBES *sales* consensus is also compared with `TXN` growth.

### Comparisons without `params` skip same-metric filtering

`DifferenceComparison()`, `SignComparison()` and `RatioComparison()` without `params` still pair feature-derived benchmarks only with their source feature. But a custom benchmark column with a metric prefix that doesn't match a full feature name is then paired with every feature. See [Concepts → pairing](concepts.md#featurebenchmark-pairing).

## Benchmarks

- **Cohort includes the entity itself.** In small cohorts this pulls the benchmark toward the entity's own value. A leave-one-out mean is not available yet.
- **IBES row order.** The IBES join sorts by date only and joins with `by=keys`, which triggers a Polars warning that sortedness can't be checked. It works when the input is sorted by date, but the output row order is by date rather than by entity.

## Edge values

- **Partial early windows:** rolling sums use `min_samples=1`, so each entity's first `w − 1` rows are partial sums rather than nulls.
- **Division by zero:** YoY/PoP growth with a zero prior sum and `RatioComparison` with a zero benchmark give `inf`/`NaN` (not null). Filter with `pl.col(c).is_finite()`.

## Performance

`RankComparison` (Python callback in `rolling_map`) and `ResidualComparison` (NumPy loop over every row) scale poorly. On a panel of thousands of entities × years × many column pairs, they dominate runtime.
