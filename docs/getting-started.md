# Getting started

## Installation

feature_kit needs Python ≥ 3.10, Polars ≥ 0.20 and NumPy ≥ 1.26.

```bash
git clone git@github.com:choochoo0726/feature_kit.git
cd feature_kit
uv sync
```

To install it into another uv project from a local checkout:

```bash
uv add --editable ../feature_kit
```

The import name is `feature_kit`, and the distribution name in `pyproject.toml` is `feature-kit`.

## Input format

Every component takes a **long-format Polars DataFrame** with one row per entity per day:

| Column | Type | Description |
|---|---|---|
| entity key(s) | any | e.g. `specid`. Set via `Param.keys`. |
| date | `pl.Date` | e.g. `DATADATE`. Set via `Param.date_col`. |
| metric columns | numeric | e.g. `SALES`, `TXN`. Set via `Param.metrics`. |
| extra columns | any | Only needed by specific components, e.g. a cohort column for `CohortRollingSumBenchmark`. |

!!! warning "Use one row per calendar day, with no gaps"
    Rolling windows count **rows**, while lags and YoY offsets are matched by **calendar date**. These agree only when every entity has a row for every day. Fill missing days (e.g. with zero or null) before computing. See [Known limitations](limitations.md#rolling-windows-count-rows-not-days).

## End-to-end example

This builds two years of synthetic daily sales for three companies and computes YoY-growth signals.

```python
from datetime import date, timedelta
import math
import polars as pl

from feature_kit import (
    FeatureBuilder, Param,
    YOYGrowthBaseFeature, ZeroBenchmark, LaggedYOYGrowthBenchmark,
    DifferenceComparison, ZScoreComparison, ClipTransformation,
)

days = [date(2023, 1, 1) + timedelta(days=i) for i in range(730)]
raw_df = pl.DataFrame({
    "specid":   [c for c in ["AAA", "BBB", "CCC"] for _ in days],
    "DATADATE": days * 3,
    "SALES":    [
        100 * (1 + g) ** (i / 365) * (1 + 0.1 * math.sin(i / 7))
        for g in (0.05, 0.15, -0.05) for i in range(len(days))
    ],
})

params = Param(keys=["specid"], metrics=["SALES"])

builder = FeatureBuilder(
    params=params,
    base_feature=YOYGrowthBaseFeature(params, windows=[("quadweekly", 28)]),
    benchmarks=[
        ZeroBenchmark(params),
        LaggedYOYGrowthBenchmark(params, periods=[("month", 28)]),
    ],
    comparisons=[
        DifferenceComparison(params),
        ZScoreComparison(params, lookback_window=56),
    ],
    transformations=[ClipTransformation(-3.0, 3.0)],
)

signals = builder.compute(raw_df)
print(signals.columns)
```

```text
['specid', 'DATADATE',
 'SALES__YOY_GROWTH__QUADWEEKLY__VS__ZERO_BENCHMARK__DIFFERENCE',
 'SALES__YOY_GROWTH__QUADWEEKLY__VS__ZERO_BENCHMARK__ZSCORE',
 'SALES__YOY_GROWTH__QUADWEEKLY__VS__SALES__YOY_GROWTH__QUADWEEKLY__LAG1MONTH__DIFFERENCE',
 'SALES__YOY_GROWTH__QUADWEEKLY__VS__SALES__YOY_GROWTH__QUADWEEKLY__LAG1MONTH__ZSCORE']
```

Early rows are null. YoY growth needs a full prior year (364 days), and the lagged benchmark and the 56-row z-score window each need more history on top of that, so here only the last ~283 of 730 days per entity have every column filled. Drop those rows, or keep them as nulls for your model to handle.

## Running stages individually

Each component has its own `.compute()`, which helps when inspecting intermediate results:

```python
base = YOYGrowthBaseFeature(params).compute(raw_df)
bench = LaggedYOYGrowthBenchmark(params).compute(base, raw_df)
diff = DifferenceComparison(params).compute(base, bench)
```

A worked notebook with a five-company synthetic dataset is in `notebooks/notebook.ipynb`.

## Building these docs

```bash
uv sync --group docs
uv run --group docs mkdocs serve      # live preview at http://127.0.0.1:8000
uv run --group docs mkdocs build      # static site in site/
```
