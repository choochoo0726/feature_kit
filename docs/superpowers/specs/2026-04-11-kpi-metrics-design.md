# kpi_metrics — Design Spec
**Date:** 2026-04-11
**Status:** Approved

## Purpose

A Python library for computing alternative data KPI features used to predict company earnings surprises and consensus changes. Built on Polars for performance. Each component is independently callable and composable via `KpiMetricsBuilder`.

---

## Project Structure

```
kpi_metrics/
├── kpi_metrics/
│   ├── __init__.py          # public API re-exports
│   ├── params.py            # Param dataclass
│   ├── base_features.py     # RollingSumBaseFeature, YOYGrowthBaseFeature, POPGrowthBaseFeature
│   ├── benchmarks.py        # Zero, LaggedYOY, LaggedPOP, LaggedRollingSum, IBES, Cohort benchmarks
│   ├── comparisons.py       # Difference, Sign, Ratio, Residual, ZScore, Rank comparisons
│   ├── transformations.py   # Clip, Winsorize, RampUp, RampDown transformations
│   ├── builder.py           # KpiMetricsBuilder
│   └── presets.py           # curated default pipelines
├── tests/
│   ├── test_base_features.py
│   ├── test_benchmarks.py
│   ├── test_comparisons.py
│   ├── test_transformations.py
│   ├── test_builder.py
│   └── test_presets.py
├── pyproject.toml           # uv + hatchling
└── README.md
```

Files are consolidated by section. If any file exceeds ~400 lines it should be split into a subpackage.

---

## Tooling

- **Package manager:** `uv` with `hatchling`
- **DataFrame library:** Polars (all computations — no pandas)
- **Python:** 3.11+

---

## Data Contract

### Input DataFrame

| Column | Type | Notes |
|---|---|---|
| `specid` | str/int | entity identifier (default key; user-configurable) |
| `DATADATE` | `Date` | daily granularity |
| `metric_*` | Float64 | one or more raw metric columns |

- Must be sorted by `(specid, DATADATE)` ascending — caller's responsibility
- No sparse date filling — caller's responsibility
- `specid` is the default key; users configure via `Param`

### Output DataFrame

| Column | Type | Notes |
|---|---|---|
| `specid` | str/int | same as input |
| `DATADATE` | `Date` | |
| `{feature_name}` | Float64 | one column per computed feature |

Only keys, `DATADATE`, and computed feature columns are returned — no raw metrics carried through.

### Output Column Naming Convention

```
{METRIC}__{BASE_FEATURE}__VS__{BENCHMARK}__{COMPARISON}
```

Example: `SALES__QUARTERLY_YOY_GROWTH__VS__ZERO_BENCHMARK__DIFFERENCE`

---

## Param

```python
@dataclass
class Param:
    keys: list[str]        # entity identifier columns, default ["specid"]
    metrics: list[str]     # raw metric columns to compute features from
    date_col: str = "DATADATE"
```

Passed to every component at construction time.

---

## Abstract Base Classes

Each section defines one abstract base class. Every component implements `.compute()` so it can be called independently, standalone or via the builder.

```python
class BaseFeature:
    def compute(self, df: pl.DataFrame) -> pl.DataFrame: ...

class Benchmark:
    # base_df: output of a BaseFeature; raw_df: original input (for lagged lookups)
    def compute(self, base_df: pl.DataFrame, raw_df: pl.DataFrame) -> pl.DataFrame: ...

class Comparison:
    def compute(self, base_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> pl.DataFrame: ...

class Transformation:
    def compute(self, df: pl.DataFrame) -> pl.DataFrame: ...
```

---

## 1. Base Features (`base_features.py`)

### Period Specification
Periods are defined as `(name: str, days: int)` tuples, e.g. `("weekly", 7)`, `("quadweekly", 28)`, `("quarterly", 91)`. The library is calendar-agnostic and works purely in day offsets.

### `RollingSumBaseFeature`
N-day rolling sum of each metric.

```python
class RollingSumBaseFeature(BaseFeature):
    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] = [("daily",1),("weekly",7),("quadweekly",28),("quarterly",91)],
    )
    # Output columns: {METRIC}__ROLLING_SUM__{WINDOW_NAME} per metric per window
    # Built-in: DAILY_SUM, WEEKLY_SUM, QUADWEEKLY_SUM, QUARTERLY_SUM
```

### `YOYGrowthBaseFeature`
YoY growth of N-day rolling sum: `(sum_t - sum_{t-364}) / abs(sum_{t-364})`.

```python
class YOYGrowthBaseFeature(BaseFeature):
    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] = [("quadweekly",28),("quarterly",91)],
        yoy_days: int = 364,
    )
    # Output columns: {METRIC}__YOY_GROWTH__{WINDOW_NAME} per metric per window
    # Built-in: QUADWEEKLY_GROWTH, QUARTERLY_GROWTH
```

### `POPGrowthBaseFeature`
Period-over-period growth of N-day rolling sum: `(sum_t - sum_{t-P}) / abs(sum_{t-P})`.

```python
class POPGrowthBaseFeature(BaseFeature):
    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] = [("quadweekly",28),("quarterly",91)],
    )
    # Output columns: {METRIC}__POP_GROWTH__{WINDOW_NAME} per metric per window
    # Built-in: QUADWEEKLY_MOM_GROWTH, QUARTERLY_QOQ_GROWTH
```

---

## 2. Benchmarks (`benchmarks.py`)

Benchmarks are joined onto base features using `join_asof` (Polars equivalent of `merge_asof`), matching on `DATADATE` per entity.

### `ZeroBenchmark`
Constant 0.0, used as a neutral baseline.

```python
class ZeroBenchmark(Benchmark):
    def __init__(self, params: Param)
    # Output: constant 0.0 column aligned to base_df
```

### `LaggedYOYGrowthBenchmark`
YoY growth values lagged by K days.

```python
class LaggedYOYGrowthBenchmark(Benchmark):
    def __init__(self, params: Param, lags: list[int] = [7, 14, 28, 91])
    # Output columns: {METRIC}__YOY_GROWTH__{WINDOW}__LAG{K}_BENCHMARK per lag
```

### `LaggedPOPGrowthBenchmark`
PoP growth values lagged by K days.

```python
class LaggedPOPGrowthBenchmark(Benchmark):
    def __init__(self, params: Param, lags: list[int] = [7, 14, 28, 91])
    # Output columns: {METRIC}__POP_GROWTH__{WINDOW}__LAG{K}_BENCHMARK per lag
```

### `LaggedRollingSumBenchmark`
Rolling sum lagged by K days: `Sum_{K days ago}`.

```python
class LaggedRollingSumBenchmark(Benchmark):
    def __init__(self, params: Param, lags: list[int])
    # Output columns: {METRIC}__ROLLING_SUM__{WINDOW}__LAG{K}_BENCHMARK per lag
```

### `LaggedYOYProjectedRollingSumBenchmark`
Projected rolling sum using YoY seasonal adjustment:
`Sum_{364 days ago} * Sum_{K days ago} / Sum_{(364+K) days ago}`

```python
class LaggedYOYProjectedRollingSumBenchmark(Benchmark):
    def __init__(self, params: Param, lags: list[int])
```

### `IBESConsensusSalesGrowthBenchmark`
External IBES consensus sales growth, joined via `join_asof`.

```python
class IBESConsensusSalesGrowthBenchmark(Benchmark):
    def __init__(self, params: Param, freq: str = "quarterly")  # "quarterly" | "semiannual"
    # Requires benchmark_dfs["ibes"] to be passed to KpiMetricsBuilder.compute()
    # Built-in: IBES_QUARTERLY_SALES_GROWTH_BENCHMARK, IBES_SEMIANNUAL_SALES_GROWTH_BENCHMARK
```

### `CohortRollingSumBenchmark`
Mean or median of historical rolling sum values within a cohort group (e.g., brand, product age).

```python
class CohortRollingSumBenchmark(Benchmark):
    def __init__(self, params: Param, cohort_cols: list[str], agg: str = "mean")  # "mean" | "median"
```

---

## 3. Comparisons (`comparisons.py`)

Each comparison takes the aligned `(base_df, benchmark_df)` and returns feature columns.

| Class | Formula |
|---|---|
| `DifferenceComparison` | `base - benchmark` |
| `SignComparison` | `sign(base - benchmark)` |
| `RatioComparison` | `(base - benchmark) / benchmark` |
| `ResidualComparison(params, lookback_window)` | OLS residual of `base ~ benchmark + const` over rolling window |
| `ZScoreComparison(params, lookback_window)` | z-score of `(base - benchmark)` over window/group |
| `RankComparison(params, lookback_window)` | `[-1, 1]` rank of `(base - benchmark)` over window/group |

```python
class DifferenceComparison(Comparison):
    def __init__(self)

class SignComparison(Comparison):
    def __init__(self)

class RatioComparison(Comparison):
    def __init__(self)

class ResidualComparison(Comparison):
    def __init__(self, params: Param, lookback_window: int)
    # groups by params.keys; rolling OLS over lookback_window days

class ZScoreComparison(Comparison):
    def __init__(self, params: Param, lookback_window: int)
    # groups by params.keys; rolling z-score over lookback_window days

class RankComparison(Comparison):
    def __init__(self, params: Param, lookback_window: int)
    # groups by params.keys; cross-sectional [-1,1] rank over lookback_window days
```

---

## 4. Transformations (`transformations.py`)

Applied to the feature DataFrame after comparisons. All operate column-wise.

```python
class ClipTransformation(Transformation):
    def __init__(self, lower: float, upper: float)
    # Clips all feature columns to [lower, upper]

class WinsorizeTransformation(Transformation):
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99)
    # Quantiles computed cross-sectionally per DATADATE

class RampUpTransformation(Transformation):
    def __init__(self, age_col: str, min_age: int, max_age: int)
    # Scales feature from 0.0 → 1.0 based on entity age

class RampDownTransformation(Transformation):
    def __init__(self, age_col: str, min_age: int, max_age: int)
    # Scales feature from 1.0 → 0.0 based on entity age
```

---

## 5. Builder (`builder.py`)

```python
class KpiMetricsBuilder:
    def __init__(
        self,
        params: Param,
        base_feature: BaseFeature,
        benchmarks: list[Benchmark],
        comparisons: list[Comparison],
        transformations: list[Transformation] | None = None,
    )

    def compute(
        self,
        df: pl.DataFrame,
        benchmark_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        ...
```

**Execution steps:**
1. `base_feature.compute(df)` → `base_df`
2. For each benchmark: `benchmark.compute(base_df, df)` → `benchmark_df`, joined via `join_asof` on `DATADATE` per entity
3. For each `(benchmark, comparison)` pair: `comparison.compute(base_df, benchmark_df)` → feature columns
4. Merge all feature columns into one DataFrame
5. For each transformation: `transformation.compute(feature_df)` → `feature_df`
6. Return `keys + DATADATE + all feature columns`

`benchmark_dfs` accepts externally supplied DataFrames keyed by name (e.g., `{"ibes": ibes_df}`) for benchmarks that require external data.

### Example Usage

```python
from kpi_metrics import Param, KpiMetricsBuilder
from kpi_metrics import YOYGrowthBaseFeature
from kpi_metrics import LaggedYOYGrowthBenchmark, ZeroBenchmark
from kpi_metrics import DifferenceComparison, ZScoreComparison
from kpi_metrics import WinsorizeTransformation

params = Param(
    keys=["specid"],
    metrics=["SALES", "NUM_TRANSACTIONS"],
)

builder = KpiMetricsBuilder(
    params=params,
    base_feature=YOYGrowthBaseFeature(params),
    benchmarks=[
        ZeroBenchmark(params),
        LaggedYOYGrowthBenchmark(params, lags=[7, 14, 28, 91]),
    ],
    comparisons=[
        DifferenceComparison(),
        ZScoreComparison(keys=["specid"], lookback_window=364),
    ],
    transformations=[
        WinsorizeTransformation(lower_q=0.01, upper_q=0.99),
    ],
)

features_df = builder.compute(df)
```

---

## 6. Presets (`presets.py`)

Presets return a fully configured `KpiMetricsBuilder`. Users can call `.compute(df)` directly.

```python
def default_yoy_pipeline(params: Param) -> KpiMetricsBuilder:
    # YOYGrowthBaseFeature
    # Benchmarks: ZeroBenchmark, LaggedYOYGrowthBenchmark(lags=[7,14,28,91])
    # Comparisons: DifferenceComparison, ZScoreComparison(lookback_window=364)

def default_pop_pipeline(params: Param) -> KpiMetricsBuilder:
    # POPGrowthBaseFeature
    # Benchmarks: ZeroBenchmark, LaggedPOPGrowthBenchmark(lags=[7,14,28,91])
    # Comparisons: DifferenceComparison

def default_rolling_sum_pipeline(params: Param) -> KpiMetricsBuilder:
    # RollingSumBaseFeature
    # Benchmarks: LaggedRollingSumBenchmark(lags=[7,14,28,91])
    # Comparisons: DifferenceComparison, RankComparison(lookback_window=364)
```

---

## Design Principles

- **Independent callability:** Every component exposes `.compute()` and works standalone
- **Polars throughout:** No pandas — all operations use Polars lazy/eager API for performance
- **Calendar-agnostic:** Periods defined as `(name, days)` tuples — no calendar dependency
- **Caller owns data quality:** Sorting, date filling, and missing value handling are the caller's responsibility
- **Composable, not magic:** The builder is a thin orchestrator; no hidden behavior
