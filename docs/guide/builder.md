# Builder & presets

## `FeatureBuilder`

[`FeatureBuilder`](../api/builder.md) runs one base feature against a list of benchmarks and comparisons:

```python
builder = FeatureBuilder(
    params=params,
    base_feature=YOYGrowthBaseFeature(params),
    benchmarks=[ZeroBenchmark(params), LaggedYOYGrowthBenchmark(params)],
    comparisons=[DifferenceComparison(params), ZScoreComparison(params, 91)],
    transformations=[ClipTransformation(-3.0, 3.0)],
)
signals = builder.compute(raw_df, external_dfs=None)
```

- Every comparison is applied to every benchmark.
- `benchmarks` and `comparisons` must be non-empty. With no comparison output, `compute` raises an `IndexError`.
- The output has keys, date and the comparison columns only.

### Combining several base features

A builder has a single base feature. To produce YoY **and** rolling-sum signals, run one builder per base feature and join the results:

```python
yoy = FeatureBuilder(params, YOYGrowthBaseFeature(params),
                     [LaggedYOYGrowthBenchmark(params)], [DifferenceComparison(params)])
lvl = FeatureBuilder(params, RollingSumBaseFeature(params, windows=[("quadweekly", 28)]),
                     [LaggedYOYProjectedRollingSumBenchmark(params)], [RatioComparison(params)])

on = ["specid", "DATADATE"]
signals = yoy.compute(raw_df).join(lvl.compute(raw_df), on=on, how="left")
```

## Presets

Presets are functions that return a configured `FeatureBuilder`.

| Preset | Base feature | Benchmarks | Comparisons |
|---|---|---|---|
| [`sales_yoy_vs_zero`](../api/presets.md#feature_kit.presets.sales_yoy_vs_zero) | YoY growth (28, 91) | zero | difference, sign |
| [`sales_rolling_vs_zero`](../api/presets.md#feature_kit.presets.sales_rolling_vs_zero) | rolling sum (1, 7, 28, 91) | zero | difference, sign |

```python
from feature_kit.presets import sales_yoy_vs_zero

signals = sales_yoy_vs_zero(Param(keys=["specid"], metrics=["SALES"])).compute(raw_df)
```

Presets are imported from `feature_kit.presets`; they are not re-exported from the top-level `feature_kit` package. The comparisons in presets are created without `params`, which makes no difference with `ZeroBenchmark` (it pairs with every feature either way).
