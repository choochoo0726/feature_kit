# Benchmarks

A benchmark is the reference a feature is measured against. A raw growth number such as "+8% YoY" is only informative relative to something: zero, last month's growth, peers, or analyst consensus.

Every benchmark's `compute(base_df, raw_df, external_dfs=None)` returns one row per `(keys, date)` in `base_df`.

| Class | Needs base feature | Benchmark value at date *t* | Output column |
|---|---|---|---|
| [`ZeroBenchmark`](../api/benchmarks.md#feature_kit.benchmarks.ZeroBenchmark) | any | `0.0` | `ZERO_BENCHMARK` |
| [`LaggedYOYGrowthBenchmark`](../api/benchmarks.md#feature_kit.benchmarks.LaggedYOYGrowthBenchmark) | YoY growth | feature at *t − n·days* | `…__LAG1MONTH` |
| [`LaggedPOPGrowthBenchmark`](../api/benchmarks.md#feature_kit.benchmarks.LaggedPOPGrowthBenchmark) | PoP growth | feature at *t − n·days* | `…__LAG1MONTH` |
| [`LaggedRollingSumBenchmark`](../api/benchmarks.md#feature_kit.benchmarks.LaggedRollingSumBenchmark) | rolling sum | rolling sum at *t − n·days* | `…__LAG1MONTH` |
| [`LaggedYOYProjectedRollingSumBenchmark`](../api/benchmarks.md#feature_kit.benchmarks.LaggedYOYProjectedRollingSumBenchmark) | rolling sum | \(S_{t-Y} \cdot S_{t-L} / S_{t-Y-L}\) | `…__LAG1MONTH_YOY_PROJECTED` |
| [`CohortRollingSumBenchmark`](../api/benchmarks.md#feature_kit.benchmarks.CohortRollingSumBenchmark) | rolling sum | cohort mean/median on date *t* | `…__COHORT_BENCHMARK` |
| [`IBESConsensusSalesGrowthBenchmark`](../api/benchmarks.md#feature_kit.benchmarks.IBESConsensusSalesGrowthBenchmark) | YoY growth | latest consensus dated ≤ *t* | `IBES_QUARTERLY_SALES_GROWTH_BENCHMARK` |

## Lagged benchmarks: `periods` and `num_lags`

The four lagged benchmarks share one interface. `periods` is a list of `(name, days)` pairs (default `week` 7, `month` 28, `quarter` 91). For each period, lags `1·days … num_lags·days` are produced:

```python
LaggedYOYGrowthBenchmark(params, periods=[("week", 7)], num_lags=2)
# → SALES__YOY_GROWTH__QUADWEEKLY__LAG1WEEK, SALES__YOY_GROWTH__QUADWEEKLY__LAG2WEEK, ...
```

Compared with a **difference**, these give momentum or acceleration signals: "has YoY growth improved over the last month?"

## Projected rolling sum

`LaggedYOYProjectedRollingSumBenchmark` asks what this period's sum would be if the YoY growth observed *L* days ago had continued:

$$
\hat S_t = S_{t-Y}\cdot\frac{S_{t-L}}{S_{t-Y-L}}
$$

With `RatioComparison`, the signal is how far the current period is running ahead of or behind that trend.

## Cohort benchmark

`CohortRollingSumBenchmark(params, cohort_cols=["SECTOR"], agg="median")` reads `SECTOR` from `raw_df`, so the raw frame must contain it. The entity is included in its own cohort's aggregate, which pulls the benchmark toward the entity's own value in small cohorts.

!!! tip
    Cohort means of rolling-sum **levels** are dominated by the largest entities. For a relative-performance signal, usually compare growth rates, or use the ratio comparison on per-entity normalised levels.

## IBES consensus

This benchmark needs an external table, which the builder passes through `external_dfs`:

```python
ibes_df = pl.DataFrame({
    "specid": [...], "DATADATE": [...],             # consensus as-of date
    "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK": [...],
})
builder.compute(raw_df, external_dfs={"ibes": ibes_df})
```

The backward as-of join uses only consensus published on or before each date. The column has no metric prefix, so it is compared with every feature column. Use it with a single-metric `Param`, or one whose metric is actually sales.
