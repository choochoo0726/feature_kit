# Transformations

Transformations post-process signal columns. They take and return a frame with the same columns; every column except `specid` and `DATADATE` (and the age column, for ramps) is treated as a feature.

| Class | Effect |
|---|---|
| [`ClipTransformation(lower, upper)`](../api/transformations.md#feature_kit.transformations.ClipTransformation) | clip to a fixed range |
| [`WinsorizeTransformation(lower_q, upper_q)`](../api/transformations.md#feature_kit.transformations.WinsorizeTransformation) | clip to each column's quantiles (pooled over **all rows**) |
| [`RampUpTransformation(age_col, min_age, max_age)`](../api/transformations.md#feature_kit.transformations.RampUpTransformation) | multiply by a 0 → 1 ramp in entity age |
| [`RampDownTransformation(age_col, min_age, max_age)`](../api/transformations.md#feature_kit.transformations.RampDownTransformation) | multiply by a 1 → 0 ramp in entity age |

Inside `FeatureBuilder`, transformations run in list order on the merged comparison output.

## Clip vs. winsorize

- **Clip** uses fixed bounds and only looks at each row's own value, so it is safe for backtests. Use it for features with a natural scale, e.g. z-scores in `[-3, 3]` or growth in `[-1, 1]`.
- **Winsorize** computes quantiles over the **whole input frame**, across all dates. In a backtest this leaks future information into past values. Prefer clipping, or winsorize per date downstream:

```python
signals.with_columns(
    pl.col(c).clip(
        pl.col(c).quantile(0.01).over("DATADATE"),
        pl.col(c).quantile(0.99).over("DATADATE"),
    )
    for c in feature_cols
)
```

## Ramps

Ramps down-weight signals by entity "age", e.g. days since data coverage began, so newly onboarded entities with noisy, short histories contribute less:

$$
s = \operatorname{clip}\left(\frac{\text{age} - \text{min\_age}}{\text{max\_age} - \text{min\_age}},\ 0,\ 1\right),\qquad f' = f\cdot s
$$

!!! warning "Apply ramps outside the builder"
    The age column must be in the frame passed to `compute`, but `FeatureBuilder` output keeps only keys, date and comparison columns, so a ramp inside `transformations=[...]` fails with `ColumnNotFoundError`. Join the age column first and apply the ramp yourself:

    ```python
    signals = builder.compute(raw_df)
    signals = signals.join(age_df, on=["specid", "DATADATE"], how="left")
    signals = RampUpTransformation("AGE_DAYS", min_age=90, max_age=365).compute(signals)
    ```
