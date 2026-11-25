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

## Calendar windows and coverage

A window of *w* covers exactly *w* **calendar days** ending on the row's date. Gaps in an entity's dates never stretch the window:

- **Missing days count as 0.** A day with no row, or a row with a null value, adds nothing to the sum.
- **Coverage threshold.** The value is null when fewer than `min_coverage` of the *w* days have data. The default `0.8` allows up to 20% missing days; `1.0` requires every day.
- **Full history required.** The value is null until the entity has *w* days of history, so each entity's first *w − 1* days are null rather than partial sums.

```python
RollingSumBaseFeature(params, windows=[("quadweekly", 28)], min_coverage=0.9)
```

## Details that affect values

- Growth denominators use `abs(prior)`, so the growth sign is correct even when the prior-period sum is negative (e.g. net flows). A zero prior sum gives null.
- The YoY and PoP prior values are matched by **exact calendar date**, and both sums follow the coverage rule above. A missing date or low coverage on either side gives null growth instead of a misaligned value.

```python
from feature_kit import Param, YOYGrowthBaseFeature

params = Param(keys=["specid"], metrics=["SALES", "TXN"])
base = YOYGrowthBaseFeature(params, windows=[("monthly", 28)], yoy_days=364)
base_df = base.compute(raw_df)
# columns: specid, DATADATE, SALES__YOY_GROWTH__MONTHLY, TXN__YOY_GROWTH__MONTHLY
```
