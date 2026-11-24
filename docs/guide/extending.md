# Writing custom components

Each stage is an abstract base class in [`feature_kit.params`](../api/params.md). Subclass it, implement `compute`, and follow the naming convention so the other stages can find your columns.

## Custom base feature

```python
import polars as pl
from feature_kit.params import BaseFeature, Param


class RollingMeanBaseFeature(BaseFeature):
    """N-day rolling mean of each metric."""

    def __init__(self, params: Param, windows: list[tuple[str, int]]) -> None:
        self.params = params
        self.windows = windows

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        keys, date_col = self.params.keys, self.params.date_col
        exprs = [
            pl.col(m).rolling_mean(window_size=w, min_samples=w).over(keys)
              .alias(f"{m}__ROLLING_MEAN__{name.upper()}")
            for m in self.params.metrics for name, w in self.windows
        ]
        out = df.sort([*keys, date_col]).with_columns(exprs)
        return out.select([*keys, date_col, *(e.meta.output_name() for e in exprs)])
```

Contract:

- Return `keys + date_col + feature columns` only.
- Name features `{METRIC}__{FEATURE}__{WINDOW}`. The metric prefix enables same-metric pairing.

## Custom benchmark

```python
from feature_kit.params import Benchmark


class ConstantBenchmark(Benchmark):
    """A fixed hurdle, e.g. 5% growth."""

    def __init__(self, params: Param, value: float) -> None:
        self.params, self.value = params, value

    def compute(self, base_df, raw_df, external_dfs=None):
        keys, date_col = self.params.keys, self.params.date_col
        return base_df.select([*keys, date_col]).with_columns(
            pl.lit(self.value).alias("CONSTANT_BENCHMARK")
        )
```

Contract:

- Return one row per `(keys, date_col)` in `base_df`.
- Columns derived from a specific feature should be named `{feature}__{SUFFIX}` so they pair only with that metric. Columns without a metric prefix pair with every feature.
- Read external data from `external_dfs[...]` and raise `ValueError` if it is missing.

## Custom comparison

Reuse the helpers in `feature_kit.comparisons` to get the same pairing rules:

```python
from feature_kit.comparisons import _pair_cols, _shared_and_feature_cols
from feature_kit.params import Comparison


class LogRatioComparison(Comparison):
    def __init__(self, params: Param) -> None:
        self.params = params

    def compute(self, base_df, benchmark_df):
        shared, feats, benches = _shared_and_feature_cols(base_df, benchmark_df)
        joined = base_df.join(benchmark_df, on=shared, how="left")
        exprs = [
            (pl.col(f) / pl.col(b)).log().alias(f"{f}__VS__{b}__LOG_RATIO")
            for f, b in _pair_cols(feats, benches, self.params.metrics)
        ]
        return joined.with_columns(exprs).select([*shared, *(e.meta.output_name() for e in exprs)])
```

The underscore-prefixed helpers are internal and may change; copy them if you need stability.

## Custom transformation

Take a frame and return one with the same columns, leaving key and date columns untouched. Two optional hooks connect it to `FeatureBuilder`:

- `self.params`: if it is `None`, the builder sets it (on a copy) to the builder's `Param`, so your transformation knows the key and date column names.
- `required_columns`: extra input columns you need (e.g. a weight). The builder joins them from the raw input for your step only and drops them afterwards.

```python
from feature_kit.params import Transformation


class WeightTransformation(Transformation):
    """Multiply every signal by a per-row weight column."""

    def __init__(self, weight_col: str, params: Param | None = None) -> None:
        self.weight_col, self.params = weight_col, params

    @property
    def required_columns(self) -> list[str]:
        return [self.weight_col]

    def compute(self, df):
        p = self.params
        skip = {*(p.keys if p else ["specid"]), p.date_col if p else "DATADATE", self.weight_col}
        return df.with_columns(
            pl.col(c) * pl.col(self.weight_col) for c in df.columns if c not in skip
        )
```
