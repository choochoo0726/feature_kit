# Known limitations

The current behaviour, verified against the code, that you should account for when using signals for research or trading. Each item is a candidate for a future fix.

## Look-ahead risk

### Winsorization uses the full sample

`WinsorizeTransformation` computes each column's quantiles over **every row of the input**, across all dates. In a backtest, values from 2022 are clipped with bounds that depend on 2024 data. Use `ClipTransformation` with fixed bounds, or winsorize per date downstream (see [Transformations](guide/transformations.md#clip-vs-winsorize)).

### Point-in-time correctness of the input

Every component treats `date_col` as the date on which the value was known. If your vendor restates or backfills history, use a point-in-time snapshot, or the signals will look better in backtests than they will live.

## Data layout

### Rolling windows count rows, not days

Rolling sums, z-scores, ranks and residuals use the **last N rows** of each entity. Lags and YoY/PoP offsets match **exact calendar dates**. With a missing day, a "28-day" sum covers 29 calendar days while lags remain calendar-exact. Reindex each entity to a complete daily calendar before computing.

### Transformations assume `specid` and `DATADATE`

All transformations exclude the hardcoded column names `specid` and `DATADATE` rather than reading `Param.keys` and `Param.date_col`. With other key names (e.g. `ticker`, `dt`), they try to clip the key columns and fail with `InvalidOperationError`, or, for numeric keys, silently modify them.

### Ramp transformations can't run inside `FeatureBuilder`

The builder's output drops every column except keys, date and comparisons, so the age column a ramp needs is gone. Apply ramps after joining the age column (see [Ramps](guide/transformations.md#ramps)).

## Pairing and output size

### Comparisons pair across windows

Same-metric pairing filters by metric only. With windows `quadweekly` and `quarterly`, `SALES__YOY_GROWTH__QUADWEEKLY` is also compared with `SALES__YOY_GROWTH__QUARTERLY__LAG1WEEK`. Those cross-window columns are rarely meaningful and double the output width. Select the same-window columns downstream if needed:

```python
def is_same_window(col: str, metrics: list[str]) -> bool:
    base, bench = col.split("__VS__")
    metric_prefixed = any(bench.startswith(m + "__") for m in metrics)
    return bench.startswith(base) or not metric_prefixed   # keep ZERO/IBES pairs

signal_cols = [c for c in signals.columns if "__VS__" in c]
signals = signals.select(
    ["specid", "DATADATE", *(c for c in signal_cols if is_same_window(c, params.metrics))]
)
```

### Unprefixed benchmarks pair with every metric

`ZERO_BENCHMARK` and the IBES columns have no metric prefix, so with `metrics=["SALES", "TXN"]` the IBES *sales* consensus is also compared with `TXN` growth.

### Comparisons without `params` pair across metrics

`DifferenceComparison()`, `SignComparison()` and `RatioComparison()` without `params` disable same-metric filtering.

## Benchmarks

- **Cohort includes the entity itself.** In small cohorts this pulls the benchmark toward the entity's own value. A leave-one-out mean is not available yet.
- **IBES row order.** The IBES join sorts by date only and joins with `by=keys`, which triggers a Polars warning that sortedness can't be checked. It works when the input is sorted by date, but the output row order is by date rather than by entity.

## Edge values

- **Partial early windows:** rolling sums use `min_samples=1`, so each entity's first `w − 1` rows are partial sums rather than nulls.
- **Division by zero:** YoY/PoP growth with a zero prior sum and `RatioComparison` with a zero benchmark give `inf`/`NaN` (not null). Filter with `pl.col(c).is_finite()`.

## Performance

`RankComparison` (Python callback in `rolling_map`) and `ResidualComparison` (NumPy loop over every row) scale poorly. On a panel of thousands of entities × years × many column pairs, they dominate runtime.
