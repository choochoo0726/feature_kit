# Comparisons

A comparison turns each (feature, benchmark) pair into a signal column named `{base_col}__VS__{bench_col}__{NAME}`. See [Concepts → pairing](../concepts.md#featurebenchmark-pairing) for which pairs are formed.

Let \(d_t = \text{feature}_t - \text{benchmark}_t\).

| Class | Signal | Kind | `params` |
|---|---|---|---|
| [`DifferenceComparison`](../api/comparisons.md#feature_kit.comparisons.DifferenceComparison) | \(d_t\) | pointwise | optional |
| [`SignComparison`](../api/comparisons.md#feature_kit.comparisons.SignComparison) | \(\operatorname{sign}(d_t) \in \{-1, 0, 1\}\) | pointwise | optional |
| [`RatioComparison`](../api/comparisons.md#feature_kit.comparisons.RatioComparison) | \(d_t / \text{benchmark}_t\) | pointwise | optional |
| [`ZScoreComparison`](../api/comparisons.md#feature_kit.comparisons.ZScoreComparison) | \((d_t - \bar d) / s_d\) over last *W* rows | rolling, per entity | required |
| [`RankComparison`](../api/comparisons.md#feature_kit.comparisons.RankComparison) | percentile of \(d_t\) in last *W* rows, scaled to \([-1, 1]\) | rolling, per entity | required |
| [`ResidualComparison`](../api/comparisons.md#feature_kit.comparisons.ResidualComparison) | OLS residual of feature ~ benchmark over last *W* rows | rolling, per entity | required |

## Pointwise comparisons

- **Difference** is the default. With growth features it is in growth units (e.g. +0.03 = 3 points above benchmark).
- **Sign** is a robust direction-only signal and is useful as a model feature that ignores magnitude outliers.
- **Ratio** is for level benchmarks (lagged or projected rolling sums), where it gives a growth rate. Don't use it with `ZeroBenchmark` (division by zero) or with growth benchmarks near zero.

## Rolling comparisons

`ZScoreComparison`, `RankComparison` and `ResidualComparison` normalise each entity's \(d_t\) against **its own recent history**, which makes signals comparable across entities with different volatility.

- The window `lookback_window` counts rows, including today. The first *W − 1* rows of each entity are null.
- All three are **time-series** statistics within an entity. None is cross-sectional across entities. For a cross-sectional rank on each date, rank the output downstream, e.g. `pl.col(c).rank().over("DATADATE")`.
- **Speed:** z-score is vectorised Polars. Rank uses `rolling_map` with a Python callback, and residual is a NumPy loop over every row. Both are much slower on large panels, so restrict them to the few feature/benchmark pairs you need.

```python
from feature_kit import ZScoreComparison

z = ZScoreComparison(params, lookback_window=91)
z_df = z.compute(base_df, bench_df)
```
