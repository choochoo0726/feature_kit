# Transformations

Transformations post-process signal columns. They take and return a frame with the same columns. Every column except the key and date columns (and the age column, for ramps) is treated as a feature.

Key and date names come from the optional `params=` argument. Without it, `specid` / `DATADATE` are assumed, and a `ValueError` asks for `params` if those columns aren't in the frame. Inside `FeatureBuilder` you can leave it out: the builder supplies its own `params`.

| Class | Effect |
|---|---|
| [`ClipTransformation(lower, upper)`](../api/transformations.md#feature_kit.transformations.ClipTransformation) | clip to a fixed range |
| [`WinsorizeTransformation(lower_q, upper_q, scope="date")`](../api/transformations.md#feature_kit.transformations.WinsorizeTransformation) | clip to each column's quantiles, computed **per date** across entities |
| [`RampUpTransformation(age_col, min_age, max_age)`](../api/transformations.md#feature_kit.transformations.RampUpTransformation) | multiply by a 0 → 1 ramp in entity age |
| [`RampDownTransformation(age_col, min_age, max_age)`](../api/transformations.md#feature_kit.transformations.RampDownTransformation) | multiply by a 1 → 0 ramp in entity age |

Inside `FeatureBuilder`, transformations run in list order on the merged comparison output.

## Clip vs. winsorize

- **Clip** uses fixed bounds and only looks at each row's own value, so it is safe for backtests. Use it for features with a natural scale, e.g. z-scores in `[-3, 3]` or growth in `[-1, 1]`.
- **Winsorize** clips to quantiles. By default (`scope="date"`) they are computed **per date across entities**, so each row only sees data from its own date. This is point-in-time safe, and it needs a reasonably wide cross-section (with 5 entities, the 1%/99% quantiles are almost the min/max).
- `scope="pooled"` computes one set of quantiles over the whole frame, across all dates. That leaks future information into past values, so don't use it in backtests.

```python
WinsorizeTransformation(lower_q=0.01, upper_q=0.99)                   # per date (default)
WinsorizeTransformation(lower_q=0.01, upper_q=0.99, scope="pooled")   # full sample: look-ahead!
```

## Ramps

Ramps down-weight signals by entity "age", e.g. days since data coverage began, so newly onboarded entities with noisy, short histories contribute less:

$$
s = \operatorname{clip}\left(\frac{\text{age} - \text{min\_age}}{\text{max\_age} - \text{min\_age}},\ 0,\ 1\right),\qquad f' = f\cdot s
$$

**Inside `FeatureBuilder`**, put the age column in the raw input. The builder joins it in for the ramp step only and drops it afterwards, so other transformations never see it and the output stays signals-only:

```python
raw_df = raw_df.join(age_df, on=["specid", "DATADATE"], how="left")
builder = FeatureBuilder(
    params, YOYGrowthBaseFeature(params), [ZeroBenchmark(params)], [DifferenceComparison(params)],
    transformations=[RampUpTransformation("AGE_DAYS", min_age=90, max_age=365)],
)
signals = builder.compute(raw_df)   # no AGE_DAYS column in the output
```

If the age column is missing from the input, `compute` raises a `ValueError` naming it. `max_age` must be greater than `min_age`.

**Standalone**, the age column must be in the frame you pass to `compute`:

```python
signals = signals.join(age_df, on=["specid", "DATADATE"], how="left")
signals = RampUpTransformation("AGE_DAYS", min_age=90, max_age=365).compute(signals)
```
