# Base features

A base feature turns raw daily metrics into the series you actually want to forecast with. All three types are built on an N-day **rolling sum** of each metric. A window is a `(name, size)` pair, and the name is upper-cased into the column name.

| Class | Formula | Output column | Default windows |
|---|---|---|---|
| [`RollingSumBaseFeature`](../api/base_features.md#feature_kit.base_features.RollingSumBaseFeature) | \(S^{(w)}_t = \sum_{i=0}^{w-1} x_{t-i}\) | `SALES__ROLLING_SUM__WEEKLY` | daily 1, weekly 7, quadweekly 28, quarterly 91 |
| [`YOYGrowthBaseFeature`](../api/base_features.md#feature_kit.base_features.YOYGrowthBaseFeature) | \((S_t - S_{t-364}) / \lvert S_{t-364}\rvert\) | `SALES__YOY_GROWTH__QUADWEEKLY` | quadweekly 28, quarterly 91 |
| [`POPGrowthBaseFeature`](../api/base_features.md#feature_kit.base_features.POPGrowthBaseFeature) | \((S_t - S_{t-w}) / \lvert S_{t-w}\rvert\) | `SALES__POP_GROWTH__QUARTERLY` | quadweekly 28, quarterly 91 |

## Choosing one

- **YoY growth** is the default for earnings-surprise work. It removes annual seasonality and corresponds directly to reported YoY revenue growth and consensus. It needs at least a year of history before producing values.
- **PoP growth** reacts faster (last 28 days vs. the 28 before), but it carries seasonality, so compare it against its own lags or cohorts rather than zero.
- **Rolling sums** are levels. They feed the projected, cohort and lagged rolling-sum benchmarks, and give growth rates when paired with `RatioComparison`.

## Details that affect values

- Growth denominators use `abs(prior)`, so the growth sign is correct even when the prior-period sum is negative (e.g. net flows). A zero prior sum gives `inf`/`NaN`.
- The rolling sum uses `min_samples=1`: each entity's first `w − 1` rows are partial sums, not nulls. For `RollingSumBaseFeature` these early values understate the full-window sum.
- The YoY and PoP prior values are matched by **exact calendar date**, so a missing date yields null growth instead of a misaligned value.

```python
from feature_kit import Param, YOYGrowthBaseFeature

params = Param(keys=["specid"], metrics=["SALES", "TXN"])
base = YOYGrowthBaseFeature(params, windows=[("monthly", 28)], yoy_days=364)
base_df = base.compute(raw_df)
# columns: specid, DATADATE, SALES__YOY_GROWTH__MONTHLY, TXN__YOY_GROWTH__MONTHLY
```
