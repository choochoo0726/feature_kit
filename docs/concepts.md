# Concepts

## `Param`: one shared configuration

All components that need column information take the same [`Param`](api/params.md):

```python
Param(keys=["specid"], metrics=["SALES", "TXN"], date_col="DATADATE")
```

- `keys` are the grouping columns. Rolling sums, z-scores, ranks and residuals are computed **within** each entity.
- `metrics` lists the raw columns to process. Each one becomes the prefix of every column derived from it.
- `date_col` must be a `pl.Date`.

## Column naming

Column names record how each column was computed, using double underscores as separators:

```text
{METRIC}__{FEATURE}__{WINDOW}                                    base feature
{METRIC}__{FEATURE}__{WINDOW}__{BENCHMARK SUFFIX}                metric-derived benchmark
ZERO_BENCHMARK                                                   standalone benchmark
{METRIC}__IBES_{FREQ}_SALES_GROWTH_BENCHMARK                     metric-scoped external benchmark
{base_col}__VS__{bench_col}__{COMPARISON}                        comparison (final signal)
```

For example, `SALES__YOY_GROWTH__QUADWEEKLY__VS__SALES__YOY_GROWTH__QUADWEEKLY__LAG1MONTH__ZSCORE` is the z-score of (28-day YoY sales growth − the same growth one month earlier).

Components rely on these names:

- Lagged and cohort benchmarks find their inputs by substring (`__YOY_GROWTH__`, `__POP_GROWTH__`, `__ROLLING_SUM__`), so a benchmark must be paired with a matching base feature, or it produces no columns.
- Comparisons pair features with benchmarks by metric prefix (see below).

| Base feature | Compatible benchmarks |
|---|---|
| `RollingSumBaseFeature` | `ZeroBenchmark`, `LaggedRollingSumBenchmark`, `LaggedYOYProjectedRollingSumBenchmark`, `CohortRollingSumBenchmark` |
| `YOYGrowthBaseFeature` | `ZeroBenchmark`, `LaggedYOYGrowthBenchmark`, `IBESConsensusSalesGrowthBenchmark` |
| `POPGrowthBaseFeature` | `ZeroBenchmark`, `LaggedPOPGrowthBenchmark` |

## Date alignment: lag by shifting dates

Lags are implemented by **shifting the date column forward** and joining back on `(keys, date)`, not by shifting rows. For example, `LaggedYOYGrowthBenchmark` with `("month", 28)` gives, on date *t*, the feature value from calendar date *t − 28*.

This has two consequences:

1. If date *t − 28* is missing for an entity, the benchmark is null. Gaps never pull in a value from the wrong day.
2. The `yoy_days` default of **364** (52 weeks) keeps the day of week aligned, which matters for spending data with strong weekly seasonality.

## Feature–benchmark pairing

A comparison receives the base-feature frame and one benchmark frame, then builds pairs:

1. A benchmark column **derived from a specific feature** (named `{feature}__...`, e.g. `SALES__YOY_GROWTH__QUADWEEKLY__LAG1MONTH`) is paired **only with that feature**: same metric *and* same window. The 28-day feature is never compared with the lagged 91-day benchmark, and `SALES` is never compared with lagged `TXN`.
2. A benchmark column with a **metric prefix** that doesn't match a whole feature name (e.g. `SALES__IBES_QUARTERLY_SALES_GROWTH_BENCHMARK`) is paired with every feature of **that metric**.
3. A benchmark column **without** a metric prefix (`ZERO_BENCHMARK`) is paired with **every** feature.

Metrics are read from the feature column names (the text before the first `__`), so these rules apply whether or not the comparison was given `params`.

## How `FeatureBuilder` combines stages

`FeatureBuilder.compute(df)`:

1. runs the base feature once;
2. for **every benchmark × every comparison**, computes the comparison;
3. left-joins all comparison outputs on `(keys, date)` into one wide frame;
4. applies transformations in order.

The output contains **only** keys, date and comparison columns. Base features and benchmarks are not included; compute them separately if you need them.

The number of output columns grows multiplicatively:

```text
n_metrics × n_windows × n_benchmark_columns_per_feature × n_comparisons
```

For example, 2 metrics × 2 windows with `LaggedYOYGrowthBenchmark` (3 default periods × `num_lags=2` = 6 columns per feature) and 3 comparisons gives 2 × 2 × 6 × 3 = **72** columns.
