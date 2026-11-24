# feature_kit

A Python library for computing KPI features from daily alternative data, designed to support earnings surprise prediction and consensus change analysis. Built on [Polars](https://pola.rs/) for fast, memory-efficient computation.

## Overview

`feature_kit` follows a **builder pattern** with four independently composable components:

```
raw_df
  └── BaseFeature      → rolling sums, YoY growth, PoP growth
        └── Benchmark  → zero, lagged, cohort, IBES consensus
              └── Comparison → difference, sign, ratio, z-score, rank, residual
                    └── Transformation → clip, winsorize, ramp up/down
```

Each component is independently callable via `.compute()`, or they can be orchestrated together through `FeatureBuilder`.

## Documentation

The full docs (user guide, recipes, known limitations and API reference) live in `docs/` and build with MkDocs Material:

```bash
uv run --group docs mkdocs serve   # live preview at http://127.0.0.1:8000
```

## Installation

```bash
# Install with uv (recommended)
uv add feature-kit

# Or clone and install for development
git clone git@github.com:choochoo0726/feature_kit.git
cd feature_kit
uv sync
```

**Requirements:** Python ≥ 3.11, Polars ≥ 0.20, NumPy ≥ 1.26

## Input Format

All components expect a Polars DataFrame with:

| Column | Type | Description |
|--------|------|-------------|
| `specid` | `str` | Entity identifier (e.g. company ticker). Configurable via `Param`. |
| `DATADATE` | `date` | Observation date. Configurable via `Param`. |
| metric columns | `float` | One or more numeric metrics (e.g. `SALES`, `REVENUE`). |

```python
import polars as pl
from datetime import date

raw_df = pl.DataFrame({
    "specid":   ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
    "DATADATE": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)] * 2,
    "SALES":    [100.0, 110.0, 105.0, 200.0, 195.0, 210.0],
})
```

> **Note:** Rolling windows are **row-count-based**, not calendar-day-based. For daily data with no gaps, these are equivalent. Ensure there are no missing dates within each entity if calendar-day semantics matter.

---

## Quick Start: Presets

The fastest way to get started is with the built-in presets:

```python
from feature_kit import FeatureBuilder, Param
from feature_kit.presets import sales_yoy_vs_zero, sales_rolling_vs_zero

params = Param(keys=["specid"], metrics=["SALES"])

# YoY growth features compared against zero (difference + sign)
builder = sales_yoy_vs_zero(params)
result = builder.compute(raw_df)
print(result)
# ┌────────┬────────────┬─────────────────────────────────────────────────────┬──────┐
# │ specid ┆ DATADATE   ┆ SALES__YOY_GROWTH__QUADWEEKLY__VS__ZERO_BENCHMARK__ ┆ ...  │
# │        ┆            ┆ DIFFERENCE                                          ┆      │

# Rolling sum features compared against zero (difference + sign)
builder = sales_rolling_vs_zero(params)
result = builder.compute(raw_df)
```

---

## Building a Custom Pipeline

### 1. Define Parameters

```python
from feature_kit import Param

params = Param(
    keys=["specid"],       # entity identifier column(s)
    metrics=["SALES"],     # raw metric column(s) to process
    date_col="DATADATE",   # default; override if your data uses a different name
)
```

### 2. Choose a Base Feature

Base features transform raw metrics into comparable signals.

```python
from feature_kit import RollingSumBaseFeature, YOYGrowthBaseFeature, POPGrowthBaseFeature

# Rolling sum over custom windows: list of (name, days) tuples
base = RollingSumBaseFeature(params, windows=[("weekly", 7), ("monthly", 28)])
# Output columns: SALES__ROLLING_SUM__WEEKLY, SALES__ROLLING_SUM__MONTHLY

# YoY growth of rolling sums (current vs. same period one year ago)
base = YOYGrowthBaseFeature(params, windows=[("quarterly", 91)], yoy_days=364)
# Output columns: SALES__YOY_GROWTH__QUARTERLY

# Period-over-period growth (28-day window vs. 28 days ago)
base = POPGrowthBaseFeature(params, windows=[("quadweekly", 28)])
# Output columns: SALES__POP_GROWTH__QUADWEEKLY

# Compute independently
base_df = base.compute(raw_df)
```

**Default windows** when `windows=None`:

| Feature | Default windows |
|---------|----------------|
| `RollingSumBaseFeature` | daily (1), weekly (7), quadweekly (28), quarterly (91) |
| `YOYGrowthBaseFeature` | quadweekly (28), quarterly (91) |
| `POPGrowthBaseFeature` | quadweekly (28), quarterly (91) |

### 3. Choose Benchmarks

Benchmarks define what each feature is compared against.

```python
from feature_kit import (
    ZeroBenchmark,
    LaggedYOYGrowthBenchmark,
    LaggedPOPGrowthBenchmark,
    LaggedRollingSumBenchmark,
    LaggedYOYProjectedRollingSumBenchmark,
    IBESConsensusSalesGrowthBenchmark,
    CohortRollingSumBenchmark,
)

# Constant zero — "is growth positive?"
bench = ZeroBenchmark(params)

# Lagged versions of YoY growth (7, 14, 28, 91 days ago by default)
bench = LaggedYOYGrowthBenchmark(params, lags=[7, 28])
# Output columns: SALES__YOY_GROWTH__QUARTERLY__LAG7_BENCHMARK, ...

# Lagged rolling sum
bench = LaggedRollingSumBenchmark(params, lags=[7, 28])

# YoY-projected rolling sum: Sum_yoy * Sum_lag / Sum_(yoy+lag)
bench = LaggedYOYProjectedRollingSumBenchmark(params, lags=[7, 28])

# IBES sell-side consensus (requires external data)
bench = IBESConsensusSalesGrowthBenchmark(params, freq="quarterly")
# Pass external_dfs={"ibes": ibes_df} to builder.compute()

# Cohort peer benchmark (mean or median within cohort group per date)
bench = CohortRollingSumBenchmark(params, cohort_cols=["sector"], agg="median")

# Compute independently
bench_df = bench.compute(base_df, raw_df)
```

### 4. Choose Comparisons

Comparisons measure how the base feature differs from its benchmark.

```python
from feature_kit import (
    DifferenceComparison,
    SignComparison,
    RatioComparison,
    ZScoreComparison,
    RankComparison,
    ResidualComparison,
)

# base - benchmark
comp = DifferenceComparison()
# Output: SALES__YOY_GROWTH__QUARTERLY__VS__ZERO_BENCHMARK__DIFFERENCE

# sign(base - benchmark): +1, 0, -1
comp = SignComparison()

# (base - benchmark) / benchmark
comp = RatioComparison()

# Rolling z-score of (base - benchmark) over a lookback window
comp = ZScoreComparison(params, lookback_window=52)

# Cross-sectional [-1, 1] percentile rank of (base - benchmark)
comp = RankComparison(params, lookback_window=52)

# OLS residual of (base ~ benchmark + const) over a rolling window
comp = ResidualComparison(params, lookback_window=52)

# Compute independently
comp_df = comp.compute(base_df, bench_df)
```

**Output column naming:** `{BASE_FEATURE_COL}__VS__{BENCHMARK_COL}__{COMPARISON_TYPE}`

### 5. Optionally Add Transformations

Transformations normalize or scale the comparison output.

```python
from feature_kit import (
    ClipTransformation,
    WinsorizeTransformation,
    RampUpTransformation,
    RampDownTransformation,
)

# Hard clip to [lower, upper]
t = ClipTransformation(lower=-3.0, upper=3.0)

# Clip to [lower_q, upper_q] quantiles (cross-sectional)
t = WinsorizeTransformation(lower_q=0.01, upper_q=0.99)

# Scale 0→1 based on entity age: useful for ramping in new entities
t = RampUpTransformation(age_col="age_days", min_age=0, max_age=365)

# Scale 1→0 based on entity age: useful for ramping out stale entities
t = RampDownTransformation(age_col="age_days", min_age=0, max_age=365)

# Compute independently (operates on all non-key, non-date columns)
result_df = t.compute(comp_df)
```

### 6. Assemble with FeatureBuilder

```python
from feature_kit import (
    FeatureBuilder, Param,
    YOYGrowthBaseFeature,
    ZeroBenchmark, LaggedYOYGrowthBenchmark,
    DifferenceComparison, ZScoreComparison,
    WinsorizeTransformation,
)

params = Param(keys=["specid"], metrics=["SALES"])

builder = FeatureBuilder(
    params=params,
    base_feature=YOYGrowthBaseFeature(params, windows=[("quarterly", 91)]),
    benchmarks=[
        ZeroBenchmark(params),
        LaggedYOYGrowthBenchmark(params, lags=[28, 91]),
    ],
    comparisons=[
        DifferenceComparison(),
        ZScoreComparison(params, lookback_window=52),
    ],
    transformations=[
        WinsorizeTransformation(lower_q=0.01, upper_q=0.99),
    ],
)

result = builder.compute(raw_df)
print(result.columns)
# ['specid', 'DATADATE',
#  'SALES__YOY_GROWTH__QUARTERLY__VS__ZERO_BENCHMARK__DIFFERENCE',
#  'SALES__YOY_GROWTH__QUARTERLY__VS__ZERO_BENCHMARK__ZSCORE',
#  'SALES__YOY_GROWTH__QUARTERLY__VS__SALES__YOY_GROWTH__QUARTERLY__LAG28_BENCHMARK__DIFFERENCE',
#  'SALES__YOY_GROWTH__QUARTERLY__VS__SALES__YOY_GROWTH__QUARTERLY__LAG28_BENCHMARK__ZSCORE',
#  ...]
```

### Using External Data (IBES)

```python
ibes_df = pl.DataFrame({
    "specid":   ["AAPL", "AAPL"],
    "DATADATE": [date(2023, 1, 1), date(2023, 4, 1)],
    "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK": [0.05, 0.08],
})

builder = FeatureBuilder(
    params=params,
    base_feature=YOYGrowthBaseFeature(params),
    benchmarks=[IBESConsensusSalesGrowthBenchmark(params, freq="quarterly")],
    comparisons=[DifferenceComparison()],
)

result = builder.compute(raw_df, external_dfs={"ibes": ibes_df})
```

---

## Component Reference

### Base Features

| Class | Output columns | Key params |
|-------|---------------|------------|
| `RollingSumBaseFeature` | `{METRIC}__ROLLING_SUM__{WINDOW}` | `windows` |
| `YOYGrowthBaseFeature` | `{METRIC}__YOY_GROWTH__{WINDOW}` | `windows`, `yoy_days=364` |
| `POPGrowthBaseFeature` | `{METRIC}__POP_GROWTH__{WINDOW}` | `windows` |

### Benchmarks

| Class | Output columns | Key params |
|-------|---------------|------------|
| `ZeroBenchmark` | `ZERO_BENCHMARK` | — |
| `LaggedYOYGrowthBenchmark` | `{col}__LAG{K}_BENCHMARK` | `lags=[7,14,28,91]` |
| `LaggedPOPGrowthBenchmark` | `{col}__LAG{K}_BENCHMARK` | `lags=[7,14,28,91]` |
| `LaggedRollingSumBenchmark` | `{col}__LAG{K}_BENCHMARK` | `lags=[7,14,28,91]` |
| `LaggedYOYProjectedRollingSumBenchmark` | `{col}__LAG{K}_YOY_PROJECTED_BENCHMARK` | `lags`, `yoy_days=364` |
| `IBESConsensusSalesGrowthBenchmark` | `IBES_{FREQ}_SALES_GROWTH_BENCHMARK` | `freq="quarterly"` |
| `CohortRollingSumBenchmark` | `{col}__COHORT_BENCHMARK` | `cohort_cols`, `agg="mean"` |

### Comparisons

| Class | Output suffix | Key params |
|-------|--------------|------------|
| `DifferenceComparison` | `__DIFFERENCE` | — |
| `SignComparison` | `__SIGN` | — |
| `RatioComparison` | `__RATIO` | — |
| `ZScoreComparison` | `__ZSCORE` | `lookback_window` |
| `RankComparison` | `__RANK` | `lookback_window` |
| `ResidualComparison` | `__RESIDUAL` | `lookback_window` |

### Transformations

| Class | Behavior | Key params |
|-------|---------|------------|
| `ClipTransformation` | Hard clip to `[lower, upper]` | `lower`, `upper` |
| `WinsorizeTransformation` | Clip to quantile range, per date across entities (point-in-time safe) | `lower_q=0.01`, `upper_q=0.99`, `scope="date"` (`"pooled"` = full sample, look-ahead) |
| `RampUpTransformation` | Multiply by `(age - min) / (max - min)` clamped to [0,1] | `age_col`, `min_age`, `max_age` |
| `RampDownTransformation` | Multiply by `1 - (age - min) / (max - min)` clamped to [0,1] | `age_col`, `min_age`, `max_age` |

---

## Development

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_builder.py -v
```

Tests are organized to mirror the module structure under `tests/`.
