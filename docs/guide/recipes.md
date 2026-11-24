# Recipes

Typical signal sets for nowcasting company KPIs from daily alternative data. Each recipe assumes:

```python
params = Param(keys=["specid"], metrics=["SALES"])
```

## 1. Growth level: is the company growing?

```python
FeatureBuilder(
    params,
    YOYGrowthBaseFeature(params, windows=[("quarterly", 91)]),
    benchmarks=[ZeroBenchmark(params)],
    comparisons=[DifferenceComparison(params), SignComparison(params)],
    transformations=[ClipTransformation(-1.0, 1.0)],
)
```

Trailing-quarter YoY growth, closely tracking the reported revenue growth that consensus forecasts.

## 2. Growth momentum: is growth accelerating?

```python
FeatureBuilder(
    params,
    YOYGrowthBaseFeature(params, windows=[("quadweekly", 28)]),
    benchmarks=[LaggedYOYGrowthBenchmark(params, periods=[("month", 28)], num_lags=3)],
    comparisons=[DifferenceComparison(params), ZScoreComparison(params, lookback_window=91)],
    transformations=[ClipTransformation(-3.0, 3.0)],
)
```

The change in 28-day YoY growth versus 1, 2 and 3 months ago, raw and normalised by each company's own volatility.

## 3. Versus consensus: is the data beating the Street?

```python
builder = FeatureBuilder(
    params,
    YOYGrowthBaseFeature(params, windows=[("quarterly", 91)]),
    benchmarks=[IBESConsensusSalesGrowthBenchmark(params, freq="quarterly")],
    comparisons=[DifferenceComparison(params), RankComparison(params, lookback_window=252)],
)
signals = builder.compute(raw_df, external_dfs={"ibes": ibes_df})
```

Observed growth minus the latest consensus growth: an earnings-surprise proxy. Align the 91-day window with the fiscal quarter the consensus refers to before relying on it.

## 4. Trend deviation: ahead of the seasonal run-rate?

```python
FeatureBuilder(
    params,
    RollingSumBaseFeature(params, windows=[("quadweekly", 28)]),
    benchmarks=[LaggedYOYProjectedRollingSumBenchmark(params, periods=[("month", 28)])],
    comparisons=[RatioComparison(params)],
    transformations=[ClipTransformation(-0.5, 0.5)],
)
```

How much the last 28 days exceed what last year's level × last month's YoY trend implies.

## 5. Relative to peers

```python
FeatureBuilder(
    params,
    RollingSumBaseFeature(params, windows=[("quadweekly", 28)]),
    benchmarks=[CohortRollingSumBenchmark(params, cohort_cols=["SECTOR"], agg="median")],
    comparisons=[RatioComparison(params)],
)
```

Requires a `SECTOR` column in `raw_df`. This compares **levels**, so it mostly reflects company size. It is more useful when the metric is already normalised per entity (e.g. indexed to its own history).
