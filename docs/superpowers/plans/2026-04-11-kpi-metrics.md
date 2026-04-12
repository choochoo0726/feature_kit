# kpi_metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a composable Python library for computing KPI features from daily alternative data, used to predict company earnings surprises.

**Architecture:** Each feature type (base features, benchmarks, comparisons, transformations) is a class with an independent `.compute()` method; `KpiMetricsBuilder` orchestrates them into a full pipeline. All computation uses Polars for performance.

**Tech Stack:** Python 3.11+, Polars, NumPy (for rolling OLS), uv + hatchling.

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, uv/hatchling config |
| `kpi_metrics/params.py` | `Param` dataclass + abstract base classes |
| `kpi_metrics/base_features.py` | `RollingSumBaseFeature`, `YOYGrowthBaseFeature`, `POPGrowthBaseFeature` |
| `kpi_metrics/benchmarks.py` | All benchmark classes |
| `kpi_metrics/comparisons.py` | All comparison classes |
| `kpi_metrics/transformations.py` | All transformation classes |
| `kpi_metrics/builder.py` | `KpiMetricsBuilder` |
| `kpi_metrics/presets.py` | Default pipeline factory functions |
| `kpi_metrics/__init__.py` | Public API re-exports |
| `tests/conftest.py` | Shared test fixtures |
| `tests/test_base_features.py` | Tests for base features |
| `tests/test_benchmarks.py` | Tests for benchmarks |
| `tests/test_comparisons.py` | Tests for comparisons |
| `tests/test_transformations.py` | Tests for transformations |
| `tests/test_builder.py` | Integration tests for builder |
| `tests/test_presets.py` | Tests for preset pipelines |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `kpi_metrics/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize project with uv**

```bash
cd /Users/wangxiao/Desktop/Claude/kpi_metrics
uv init --no-readme --lib --build-backend hatchling
```

- [ ] **Step 2: Replace generated pyproject.toml**

Write `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "kpi-metrics"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "polars>=0.20.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["kpi_metrics"]
```

- [ ] **Step 3: Install dependencies**

```bash
cd /Users/wangxiao/Desktop/Claude/kpi_metrics
uv sync --extra dev
```

Expected: dependencies installed, `.venv` created.

- [ ] **Step 4: Create package and test directories**

```bash
mkdir -p kpi_metrics tests
touch kpi_metrics/__init__.py tests/__init__.py
```

- [ ] **Step 5: Create shared test fixture**

Write `tests/conftest.py`:
```python
import pytest
import polars as pl
from datetime import date, timedelta


def make_daily_df(
    companies: list[str],
    start: date,
    n_days: int,
    metrics: dict[str, list[float]],
) -> pl.DataFrame:
    """Build a synthetic daily DataFrame sorted by (specid, DATADATE)."""
    rows = []
    dates = [start + timedelta(days=i) for i in range(n_days)]
    for company in companies:
        for i, d in enumerate(dates):
            row = {"specid": company, "DATADATE": d}
            for metric, values in metrics.items():
                row[metric] = float(values[i % len(values)])
            rows.append(row)
    return pl.DataFrame(rows).sort(["specid", "DATADATE"])


@pytest.fixture
def small_df():
    """10 days, 2 companies, 1 metric (SALES = 1..10)."""
    return make_daily_df(
        companies=["A", "B"],
        start=date(2023, 1, 1),
        n_days=10,
        metrics={"SALES": list(range(1, 11))},
    )


@pytest.fixture
def yoy_df():
    """730 days (2 years), 2 companies. SALES = day index + 1."""
    n = 730
    return make_daily_df(
        companies=["A", "B"],
        start=date(2022, 1, 1),
        n_days=n,
        metrics={"SALES": list(range(1, n + 1))},
    )
```

- [ ] **Step 6: Verify pytest runs**

```bash
cd /Users/wangxiao/Desktop/Claude/kpi_metrics
uv run pytest tests/ -v
```

Expected: `no tests ran`, exit 0.

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml kpi_metrics/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold kpi_metrics project with uv + hatchling"
```

---

## Task 2: Param & Abstract Base Classes (`params.py`)

**Files:**
- Create: `kpi_metrics/params.py`
- Create: `tests/test_params.py`

- [ ] **Step 1: Write failing test**

Write `tests/test_params.py`:
```python
import polars as pl
import pytest
from kpi_metrics.params import Param, BaseFeature, Benchmark, Comparison, Transformation


def test_param_defaults():
    p = Param(keys=["specid"], metrics=["SALES"])
    assert p.date_col == "DATADATE"
    assert p.keys == ["specid"]
    assert p.metrics == ["SALES"]


def test_param_custom_date_col():
    p = Param(keys=["cusip"], metrics=["REVENUE"], date_col="DATE")
    assert p.date_col == "DATE"


def test_base_feature_is_abstract():
    with pytest.raises(TypeError):
        BaseFeature()


def test_benchmark_is_abstract():
    with pytest.raises(TypeError):
        Benchmark()


def test_comparison_is_abstract():
    with pytest.raises(TypeError):
        Comparison()


def test_transformation_is_abstract():
    with pytest.raises(TypeError):
        Transformation()
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_params.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement `params.py`**

Write `kpi_metrics/params.py`:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl


@dataclass
class Param:
    keys: list[str]
    metrics: list[str]
    date_col: str = "DATADATE"


class BaseFeature(ABC):
    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute base features from raw daily DataFrame.

        Args:
            df: sorted by (keys, date_col), contains raw metric columns.

        Returns:
            DataFrame with keys + date_col + feature columns only.
        """


class Benchmark(ABC):
    @abstractmethod
    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Compute benchmark values aligned to base_df.

        Args:
            base_df: output of a BaseFeature.compute().
            raw_df: original raw daily input DataFrame.
            external_dfs: optional external DataFrames keyed by name (e.g. {"ibes": ...}).

        Returns:
            DataFrame with keys + date_col + benchmark columns.
        """


class Comparison(ABC):
    @abstractmethod
    def compute(
        self,
        base_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compare base features against a benchmark.

        Args:
            base_df: output of BaseFeature.compute().
            benchmark_df: output of Benchmark.compute(), aligned on keys + date_col.

        Returns:
            DataFrame with keys + date_col + comparison feature columns.
            Columns named: {base_col}__VS__{benchmark_col}__{comparison_name}
        """


class Transformation(ABC):
    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply transformation to feature columns in-place.

        Args:
            df: DataFrame with keys + date_col + feature columns.

        Returns:
            DataFrame with same structure, feature values transformed.
        """
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_params.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/params.py tests/test_params.py
git commit -m "feat: add Param dataclass and abstract base classes"
```

---

## Task 3: RollingSumBaseFeature (`base_features.py`)

**Files:**
- Create: `kpi_metrics/base_features.py`
- Create: `tests/test_base_features.py`

- [ ] **Step 1: Write failing test**

Write `tests/test_base_features.py`:
```python
import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.base_features import RollingSumBaseFeature


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def small_df():
    return pl.DataFrame({
        "specid": ["A"] * 5,
        "DATADATE": [date(2023, 1, i + 1) for i in range(5)],
        "SALES": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


class TestRollingSumBaseFeature:
    def test_output_columns(self, params, small_df):
        feat = RollingSumBaseFeature(params, windows=[("weekly", 3)])
        result = feat.compute(small_df)
        assert "SALES__ROLLING_SUM__WEEKLY" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "SALES__ROLLING_SUM__WEEKLY"}

    def test_rolling_sum_values(self, params, small_df):
        feat = RollingSumBaseFeature(params, windows=[("weekly", 3)])
        result = feat.compute(small_df)
        vals = result["SALES__ROLLING_SUM__WEEKLY"].to_list()
        assert vals == pytest.approx([1.0, 3.0, 6.0, 9.0, 12.0])

    def test_multiple_windows(self, params, small_df):
        feat = RollingSumBaseFeature(params, windows=[("daily", 1), ("weekly", 3)])
        result = feat.compute(small_df)
        assert "SALES__ROLLING_SUM__DAILY" in result.columns
        assert "SALES__ROLLING_SUM__WEEKLY" in result.columns

    def test_multiple_companies(self, params):
        df = pl.DataFrame({
            "specid": ["A", "A", "A", "B", "B", "B"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(3)] * 2,
            "SALES": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        feat = RollingSumBaseFeature(params, windows=[("weekly", 3)])
        result = feat.compute(df)
        a_vals = result.filter(pl.col("specid") == "A")["SALES__ROLLING_SUM__WEEKLY"].to_list()
        b_vals = result.filter(pl.col("specid") == "B")["SALES__ROLLING_SUM__WEEKLY"].to_list()
        assert a_vals == pytest.approx([1.0, 3.0, 6.0])
        assert b_vals == pytest.approx([10.0, 30.0, 60.0])

    def test_default_windows(self, params, small_df):
        feat = RollingSumBaseFeature(params)
        result = feat.compute(small_df)
        for name in ["DAILY", "WEEKLY", "QUADWEEKLY", "QUARTERLY"]:
            assert f"SALES__ROLLING_SUM__{name}" in result.columns
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_base_features.py -v
```

Expected: `ImportError` — `base_features` not found.

- [ ] **Step 3: Implement `RollingSumBaseFeature`**

Write `kpi_metrics/base_features.py`:
```python
from __future__ import annotations
from itertools import product as iproduct

import polars as pl

from kpi_metrics.params import BaseFeature, Param

_DEFAULT_ROLLING_WINDOWS: list[tuple[str, int]] = [
    ("daily", 1),
    ("weekly", 7),
    ("quadweekly", 28),
    ("quarterly", 91),
]

_DEFAULT_YOY_WINDOWS: list[tuple[str, int]] = [
    ("quadweekly", 28),
    ("quarterly", 91),
]

_DEFAULT_POP_WINDOWS: list[tuple[str, int]] = [
    ("quadweekly", 28),
    ("quarterly", 91),
]


def _rolling_sum(
    df: pl.DataFrame,
    metrics: list[str],
    keys: list[str],
    date_col: str,
    windows: list[tuple[str, int]],
) -> pl.DataFrame:
    """Compute group-wise rolling sums. Assumes df is sorted by (keys, date_col)."""
    exprs = [
        pl.col(metric)
        .rolling_sum(window_size=days, min_periods=1)
        .over(keys)
        .alias(f"{metric}__ROLLING_SUM__{name.upper()}")
        for metric, (name, days) in iproduct(metrics, windows)
    ]
    return df.with_columns(exprs)


class RollingSumBaseFeature(BaseFeature):
    """N-day rolling sum of each metric, grouped by entity keys."""

    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] = _DEFAULT_ROLLING_WINDOWS,
    ) -> None:
        self.params = params
        self.windows = windows

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = [
            f"{metric}__ROLLING_SUM__{name.upper()}"
            for metric, (name, _) in iproduct(self.params.metrics, self.windows)
        ]
        result = _rolling_sum(
            df.sort([*self.params.keys, self.params.date_col]),
            metrics=self.params.metrics,
            keys=self.params.keys,
            date_col=self.params.date_col,
            windows=self.windows,
        )
        return result.select([*self.params.keys, self.params.date_col, *feature_cols])
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_base_features.py::TestRollingSumBaseFeature -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/base_features.py tests/test_base_features.py
git commit -m "feat: add RollingSumBaseFeature"
```

---

## Task 4: YOYGrowthBaseFeature (`base_features.py`)

**Files:**
- Modify: `kpi_metrics/base_features.py`
- Modify: `tests/test_base_features.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/test_base_features.py`:
```python
from kpi_metrics.base_features import YOYGrowthBaseFeature


class TestYOYGrowthBaseFeature:
    def test_output_columns(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = YOYGrowthBaseFeature(params, windows=[("quadweekly", 4)], yoy_days=7)
        # Build 14-day df so yoy (7 days lag) is computable
        df = pl.DataFrame({
            "specid": ["A"] * 14,
            "DATADATE": [date(2023, 1, i + 1) for i in range(14)],
            "SALES": [float(i + 1) for i in range(14)],
        })
        result = feat.compute(df)
        assert "SALES__YOY_GROWTH__QUADWEEKLY" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "SALES__YOY_GROWTH__QUADWEEKLY"}

    def test_yoy_growth_values(self):
        # Window=1 (daily), yoy_days=7: growth at day 8 = (val8 - val1) / abs(val1)
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7)
        df = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        })
        result = feat.compute(df)
        # Days 1-7: prior is null (< yoy_days ago), growth = null
        # Day 8: (20 - 10) / 10 = 1.0
        vals = result["SALES__YOY_GROWTH__DAILY"].to_list()
        assert vals[:7] == [None] * 7
        assert vals[7] == pytest.approx(1.0)

    def test_multiple_companies_independent(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=3)
        df = pl.DataFrame({
            "specid": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(4)] * 2,
            "SALES": [10.0, 10.0, 10.0, 20.0, 5.0, 5.0, 5.0, 15.0],
        })
        result = feat.compute(df)
        a_vals = result.filter(pl.col("specid") == "A")["SALES__YOY_GROWTH__DAILY"].to_list()
        b_vals = result.filter(pl.col("specid") == "B")["SALES__YOY_GROWTH__DAILY"].to_list()
        # Day 4: (20 - 10) / 10 = 1.0 for A; (15 - 5) / 5 = 2.0 for B
        assert a_vals[-1] == pytest.approx(1.0)
        assert b_vals[-1] == pytest.approx(2.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_base_features.py::TestYOYGrowthBaseFeature -v
```

Expected: `ImportError` — `YOYGrowthBaseFeature` not defined.

- [ ] **Step 3: Implement `YOYGrowthBaseFeature`**

Add to `kpi_metrics/base_features.py`:
```python
class YOYGrowthBaseFeature(BaseFeature):
    """YoY growth of N-day rolling sum: (sum_t - sum_{t-yoy_days}) / abs(sum_{t-yoy_days})."""

    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] = _DEFAULT_YOY_WINDOWS,
        yoy_days: int = 364,
    ) -> None:
        self.params = params
        self.windows = windows
        self.yoy_days = yoy_days

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        metrics = self.params.metrics

        # Step 1: compute rolling sums
        roll_df = _rolling_sum(
            df.sort([*keys, date_col]),
            metrics=metrics,
            keys=keys,
            date_col=date_col,
            windows=self.windows,
        )

        roll_cols = [
            f"{m}__ROLLING_SUM__{name.upper()}"
            for m, (name, _) in iproduct(metrics, self.windows)
        ]

        # Step 2: build prior-year version by offsetting DATADATE forward by yoy_days
        prior_df = (
            roll_df
            .select([*keys, date_col, *roll_cols])
            .with_columns(
                (pl.col(date_col) + pl.duration(days=self.yoy_days)).alias(date_col)
            )
            .rename({c: f"{c}__PRIOR" for c in roll_cols})
        )

        # Step 3: join current with prior on (keys, date_col)
        joined = roll_df.join(prior_df, on=[*keys, date_col], how="left")

        # Step 4: compute growth = (curr - prior) / abs(prior)
        feature_exprs = []
        feature_cols = []
        for m, (name, _) in iproduct(metrics, self.windows):
            curr = f"{m}__ROLLING_SUM__{name.upper()}"
            prior = f"{curr}__PRIOR"
            feat = f"{m}__YOY_GROWTH__{name.upper()}"
            feature_exprs.append(
                ((pl.col(curr) - pl.col(prior)) / pl.col(prior).abs()).alias(feat)
            )
            feature_cols.append(feat)

        return (
            joined
            .with_columns(feature_exprs)
            .select([*keys, date_col, *feature_cols])
        )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_base_features.py::TestYOYGrowthBaseFeature -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/base_features.py tests/test_base_features.py
git commit -m "feat: add YOYGrowthBaseFeature"
```

---

## Task 5: POPGrowthBaseFeature (`base_features.py`)

**Files:**
- Modify: `kpi_metrics/base_features.py`
- Modify: `tests/test_base_features.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/test_base_features.py`:
```python
from kpi_metrics.base_features import POPGrowthBaseFeature


class TestPOPGrowthBaseFeature:
    def test_output_columns(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = POPGrowthBaseFeature(params, windows=[("quadweekly", 4)])
        df = pl.DataFrame({
            "specid": ["A"] * 8,
            "DATADATE": [date(2023, 1, i + 1) for i in range(8)],
            "SALES": [float(i + 1) for i in range(8)],
        })
        result = feat.compute(df)
        assert "SALES__POP_GROWTH__QUADWEEKLY" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "SALES__POP_GROWTH__QUADWEEKLY"}

    def test_pop_growth_values(self):
        # window=1 (daily sum = value itself), period=3 days
        # growth at day 4 = (val4 - val1) / abs(val1) = (4 - 1) / 1 = 3.0
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = POPGrowthBaseFeature(params, windows=[("period3", 1)])
        # Override period days for PoP lag via a custom subclass isn't needed;
        # instead test with window=3 (rolling sum of 3 days) and period=3
        feat2 = POPGrowthBaseFeature.__new__(POPGrowthBaseFeature)
        feat2.params = params
        feat2.windows = [("daily", 1)]
        feat2.pop_days_map = {"daily": 3}
        # Use standard constructor with pop_days override
        # Test with window=1, pop lag = 3
        params2 = Param(keys=["specid"], metrics=["SALES"])

        class _Feat(POPGrowthBaseFeature):
            def _pop_days(self, name: str, window_days: int) -> int:
                return 3

        df = pl.DataFrame({
            "specid": ["A"] * 6,
            "DATADATE": [date(2023, 1, i + 1) for i in range(6)],
            "SALES": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        feat3 = _Feat(params2, windows=[("daily", 1)])
        result = feat3.compute(df)
        vals = result["SALES__POP_GROWTH__DAILY"].to_list()
        # Days 1-3: prior=null; Day 4: (4-1)/1=3.0; Day 5: (5-2)/2=1.5; Day6: (6-3)/3=1.0
        assert vals[:3] == [None, None, None]
        assert vals[3] == pytest.approx(3.0)

    def test_default_windows(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = POPGrowthBaseFeature(params)
        df = pl.DataFrame({
            "specid": ["A"] * 5,
            "DATADATE": [date(2023, 1, i + 1) for i in range(5)],
            "SALES": [float(i + 1) for i in range(5)],
        })
        result = feat.compute(df)
        assert "SALES__POP_GROWTH__QUADWEEKLY" in result.columns
        assert "SALES__POP_GROWTH__QUARTERLY" in result.columns
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_base_features.py::TestPOPGrowthBaseFeature -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `POPGrowthBaseFeature`**

Add to `kpi_metrics/base_features.py`:
```python
class POPGrowthBaseFeature(BaseFeature):
    """Period-over-period growth: (rolling_sum_t - rolling_sum_{t-window_days}) / abs(rolling_sum_{t-window_days}).

    The PoP lag equals the window size (e.g., 28-day window compared to 28 days ago).
    """

    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] = _DEFAULT_POP_WINDOWS,
    ) -> None:
        self.params = params
        self.windows = windows

    def _pop_days(self, name: str, window_days: int) -> int:
        """Number of days to lag for PoP comparison. Defaults to window_days."""
        return window_days

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        metrics = self.params.metrics

        roll_df = _rolling_sum(
            df.sort([*keys, date_col]),
            metrics=metrics,
            keys=keys,
            date_col=date_col,
            windows=self.windows,
        )

        roll_cols = [
            f"{m}__ROLLING_SUM__{name.upper()}"
            for m, (name, _) in iproduct(metrics, self.windows)
        ]

        # For each window, build prior with lag = window_days
        feature_exprs = []
        feature_cols = []

        for name, window_days in self.windows:
            pop_lag = self._pop_days(name, window_days)
            cols_for_window = [f"{m}__ROLLING_SUM__{name.upper()}" for m in metrics]

            prior_df = (
                roll_df
                .select([*keys, date_col, *cols_for_window])
                .with_columns(
                    (pl.col(date_col) + pl.duration(days=pop_lag)).alias(date_col)
                )
                .rename({c: f"{c}__PRIOR" for c in cols_for_window})
            )

            roll_df = roll_df.join(prior_df, on=[*keys, date_col], how="left")

            for m in metrics:
                curr = f"{m}__ROLLING_SUM__{name.upper()}"
                prior = f"{curr}__PRIOR"
                feat = f"{m}__POP_GROWTH__{name.upper()}"
                feature_exprs.append(
                    ((pl.col(curr) - pl.col(prior)) / pl.col(prior).abs()).alias(feat)
                )
                feature_cols.append(feat)

        return (
            roll_df
            .with_columns(feature_exprs)
            .select([*keys, date_col, *feature_cols])
        )
```

- [ ] **Step 4: Run all base feature tests**

```bash
uv run pytest tests/test_base_features.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/base_features.py tests/test_base_features.py
git commit -m "feat: add POPGrowthBaseFeature"
```

---

## Task 6: ZeroBenchmark & Lagged Growth Benchmarks (`benchmarks.py`)

**Files:**
- Create: `kpi_metrics/benchmarks.py`
- Create: `tests/test_benchmarks.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_benchmarks.py`:
```python
import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.base_features import YOYGrowthBaseFeature, POPGrowthBaseFeature
from kpi_metrics.benchmarks import (
    ZeroBenchmark,
    LaggedYOYGrowthBenchmark,
    LaggedPOPGrowthBenchmark,
)


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def base_yoy_df(params):
    raw = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES": [float(i + 1) for i in range(10)],
    })
    return YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=3).compute(raw), raw


class TestZeroBenchmark:
    def test_output_columns(self, params, base_yoy_df):
        base_df, raw_df = base_yoy_df
        bench = ZeroBenchmark(params)
        result = bench.compute(base_df, raw_df)
        assert "ZERO_BENCHMARK" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "ZERO_BENCHMARK"}

    def test_all_zeros(self, params, base_yoy_df):
        base_df, raw_df = base_yoy_df
        bench = ZeroBenchmark(params)
        result = bench.compute(base_df, raw_df)
        assert result["ZERO_BENCHMARK"].to_list() == [0.0] * len(base_df)

    def test_aligned_to_base_df(self, params, base_yoy_df):
        base_df, raw_df = base_yoy_df
        bench = ZeroBenchmark(params)
        result = bench.compute(base_df, raw_df)
        assert len(result) == len(base_df)


class TestLaggedYOYGrowthBenchmark:
    def test_output_columns(self, params, base_yoy_df):
        base_df, raw_df = base_yoy_df
        bench = LaggedYOYGrowthBenchmark(params, lags=[2])
        result = bench.compute(base_df, raw_df)
        assert "SALES__YOY_GROWTH__DAILY__LAG2_BENCHMARK" in result.columns

    def test_lagged_values(self, params, base_yoy_df):
        base_df, raw_df = base_yoy_df
        bench = LaggedYOYGrowthBenchmark(params, lags=[1])
        result = bench.compute(base_df, raw_df)
        col = "SALES__YOY_GROWTH__DAILY__LAG1_BENCHMARK"
        # Lag-1 benchmark at row i should equal base_df value at row i-1
        base_vals = base_df["SALES__YOY_GROWTH__DAILY"].to_list()
        lag_vals = result[col].to_list()
        # First row lag is null; remaining should shift by 1 day
        assert lag_vals[0] is None or lag_vals[0] != lag_vals[1]


class TestLaggedPOPGrowthBenchmark:
    def test_output_columns(self, params):
        raw = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [float(i + 1) for i in range(10)],
        })
        base_df = POPGrowthBaseFeature(params, windows=[("daily", 1)]).compute(raw)
        bench = LaggedPOPGrowthBenchmark(params, lags=[2])
        result = bench.compute(base_df, raw)
        assert "SALES__POP_GROWTH__DAILY__LAG2_BENCHMARK" in result.columns
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_benchmarks.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement benchmarks**

Write `kpi_metrics/benchmarks.py`:
```python
from __future__ import annotations
from itertools import product as iproduct

import polars as pl

from kpi_metrics.params import Benchmark, Param


def _lag_feature_cols(
    base_df: pl.DataFrame,
    keys: list[str],
    date_col: str,
    feature_cols: list[str],
    lag_days: int,
    suffix: str,
) -> pl.DataFrame:
    """Return a DataFrame with feature_cols shifted forward by lag_days, renamed with suffix."""
    lag_df = (
        base_df
        .select([*keys, date_col, *feature_cols])
        .with_columns(
            (pl.col(date_col) + pl.duration(days=lag_days)).alias(date_col)
        )
        .rename({c: f"{c}__{suffix}" for c in feature_cols})
    )
    return base_df.join(lag_df, on=[*keys, date_col], how="left")


class ZeroBenchmark(Benchmark):
    """Constant 0.0 benchmark aligned to base_df."""

    def __init__(self, params: Param) -> None:
        self.params = params

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        return base_df.select([*keys, date_col]).with_columns(
            pl.lit(0.0).alias("ZERO_BENCHMARK")
        )


class LaggedYOYGrowthBenchmark(Benchmark):
    """YoY growth feature values lagged by K days, one column per lag."""

    def __init__(self, params: Param, lags: list[int] = [7, 14, 28, 91]) -> None:
        self.params = params
        self.lags = lags

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        yoy_cols = [c for c in base_df.columns if "__YOY_GROWTH__" in c]

        result = base_df.select([*keys, date_col])
        for lag in self.lags:
            lag_df = (
                base_df
                .select([*keys, date_col, *yoy_cols])
                .with_columns(
                    (pl.col(date_col) + pl.duration(days=lag)).alias(date_col)
                )
                .rename({c: f"{c}__LAG{lag}_BENCHMARK" for c in yoy_cols})
            )
            bench_cols = [f"{c}__LAG{lag}_BENCHMARK" for c in yoy_cols]
            result = result.join(
                lag_df.select([*keys, date_col, *bench_cols]),
                on=[*keys, date_col],
                how="left",
            )
        return result


class LaggedPOPGrowthBenchmark(Benchmark):
    """PoP growth feature values lagged by K days, one column per lag."""

    def __init__(self, params: Param, lags: list[int] = [7, 14, 28, 91]) -> None:
        self.params = params
        self.lags = lags

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        pop_cols = [c for c in base_df.columns if "__POP_GROWTH__" in c]

        result = base_df.select([*keys, date_col])
        for lag in self.lags:
            lag_df = (
                base_df
                .select([*keys, date_col, *pop_cols])
                .with_columns(
                    (pl.col(date_col) + pl.duration(days=lag)).alias(date_col)
                )
                .rename({c: f"{c}__LAG{lag}_BENCHMARK" for c in pop_cols})
            )
            bench_cols = [f"{c}__LAG{lag}_BENCHMARK" for c in pop_cols]
            result = result.join(
                lag_df.select([*keys, date_col, *bench_cols]),
                on=[*keys, date_col],
                how="left",
            )
        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_benchmarks.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/benchmarks.py tests/test_benchmarks.py
git commit -m "feat: add ZeroBenchmark, LaggedYOYGrowthBenchmark, LaggedPOPGrowthBenchmark"
```

---

## Task 7: Remaining Benchmarks (`benchmarks.py`)

**Files:**
- Modify: `kpi_metrics/benchmarks.py`
- Modify: `tests/test_benchmarks.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/test_benchmarks.py`:
```python
from kpi_metrics.benchmarks import (
    LaggedRollingSumBenchmark,
    LaggedYOYProjectedRollingSumBenchmark,
    IBESConsensusSalesGrowthBenchmark,
    CohortRollingSumBenchmark,
)
from kpi_metrics.base_features import RollingSumBaseFeature


@pytest.fixture
def rolling_setup(params):
    raw = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES": [float(i + 1) for i in range(10)],
    })
    base_df = RollingSumBaseFeature(params, windows=[("daily", 1)]).compute(raw)
    return base_df, raw


class TestLaggedRollingSumBenchmark:
    def test_output_columns(self, params, rolling_setup):
        base_df, raw_df = rolling_setup
        bench = LaggedRollingSumBenchmark(params, lags=[2])
        result = bench.compute(base_df, raw_df)
        assert "SALES__ROLLING_SUM__DAILY__LAG2_BENCHMARK" in result.columns

    def test_lagged_values(self, params, rolling_setup):
        base_df, raw_df = rolling_setup
        bench = LaggedRollingSumBenchmark(params, lags=[1])
        result = bench.compute(base_df, raw_df)
        col = "SALES__ROLLING_SUM__DAILY__LAG1_BENCHMARK"
        vals = result[col].to_list()
        # Row 0 (day 1): 1 day ago is before data → null
        assert vals[0] is None
        # Row 1 (day 2): lag-1 = day 1 value = 1.0
        assert vals[1] == pytest.approx(1.0)


class TestLaggedYOYProjectedRollingSumBenchmark:
    def test_output_columns(self, params):
        raw = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [float(i + 1) for i in range(10)],
        })
        base_df = RollingSumBaseFeature(params, windows=[("daily", 1)]).compute(raw)
        bench = LaggedYOYProjectedRollingSumBenchmark(params, lags=[2], yoy_days=3)
        result = bench.compute(base_df, raw)
        assert any("PROJECTED" in c for c in result.columns)


class TestIBESConsensusSalesGrowthBenchmark:
    def test_output_columns(self, params):
        raw = pl.DataFrame({
            "specid": ["A"] * 5,
            "DATADATE": [date(2023, 1, i + 1) for i in range(5)],
            "SALES": [float(i + 1) for i in range(5)],
        })
        base_df = YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=3).compute(raw)
        ibes_df = pl.DataFrame({
            "specid": ["A"] * 5,
            "DATADATE": [date(2023, 1, i + 1) for i in range(5)],
            "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK": [0.05, 0.06, 0.07, 0.08, 0.09],
        })
        bench = IBESConsensusSalesGrowthBenchmark(params, freq="quarterly")
        result = bench.compute(base_df, raw, external_dfs={"ibes": ibes_df})
        assert "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK" in result.columns


class TestCohortRollingSumBenchmark:
    def test_output_columns(self, params):
        raw = pl.DataFrame({
            "specid": ["A", "A", "B", "B"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 2)] * 2,
            "SALES": [1.0, 2.0, 3.0, 4.0],
            "sector": ["tech", "tech", "tech", "tech"],
        })
        base_df = RollingSumBaseFeature(params, windows=[("daily", 1)]).compute(raw)
        bench = CohortRollingSumBenchmark(params, cohort_cols=["sector"], agg="mean")
        result = bench.compute(base_df, raw)
        assert any("COHORT" in c for c in result.columns)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_benchmarks.py -v -k "TestLaggedRollingSum or TestLaggedYOY or TestIBES or TestCohort"
```

Expected: `ImportError`.

- [ ] **Step 3: Implement remaining benchmarks**

Add to `kpi_metrics/benchmarks.py`:
```python
class LaggedRollingSumBenchmark(Benchmark):
    """Rolling sum values lagged by K days: Sum_{K days ago}."""

    def __init__(self, params: Param, lags: list[int]) -> None:
        self.params = params
        self.lags = lags

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        roll_cols = [c for c in base_df.columns if "__ROLLING_SUM__" in c]

        result = base_df.select([*keys, date_col])
        for lag in self.lags:
            lag_df = (
                base_df
                .select([*keys, date_col, *roll_cols])
                .with_columns(
                    (pl.col(date_col) + pl.duration(days=lag)).alias(date_col)
                )
                .rename({c: f"{c}__LAG{lag}_BENCHMARK" for c in roll_cols})
            )
            bench_cols = [f"{c}__LAG{lag}_BENCHMARK" for c in roll_cols]
            result = result.join(
                lag_df.select([*keys, date_col, *bench_cols]),
                on=[*keys, date_col],
                how="left",
            )
        return result


class LaggedYOYProjectedRollingSumBenchmark(Benchmark):
    """Projected rolling sum: Sum_{yoy_days ago} * Sum_{K days ago} / Sum_{(yoy_days+K) days ago}."""

    def __init__(self, params: Param, lags: list[int], yoy_days: int = 364) -> None:
        self.params = params
        self.lags = lags
        self.yoy_days = yoy_days

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        roll_cols = [c for c in base_df.columns if "__ROLLING_SUM__" in c]

        # We need raw rolling sums at various lags; compute from base_df
        result = base_df.select([*keys, date_col])

        for lag in self.lags:
            # Prior-year: shift forward by yoy_days
            yoy_df = (
                base_df.select([*keys, date_col, *roll_cols])
                .with_columns((pl.col(date_col) + pl.duration(days=self.yoy_days)).alias(date_col))
                .rename({c: f"{c}__YOY" for c in roll_cols})
            )
            # K-days-ago: shift forward by lag
            lag_df = (
                base_df.select([*keys, date_col, *roll_cols])
                .with_columns((pl.col(date_col) + pl.duration(days=lag)).alias(date_col))
                .rename({c: f"{c}__LAG{lag}" for c in roll_cols})
            )
            # (yoy_days + lag) days ago: shift forward by yoy_days + lag
            yoy_lag_df = (
                base_df.select([*keys, date_col, *roll_cols])
                .with_columns((pl.col(date_col) + pl.duration(days=self.yoy_days + lag)).alias(date_col))
                .rename({c: f"{c}__YOYLAG{lag}" for c in roll_cols})
            )

            joined = (
                base_df.select([*keys, date_col])
                .join(yoy_df, on=[*keys, date_col], how="left")
                .join(lag_df, on=[*keys, date_col], how="left")
                .join(yoy_lag_df, on=[*keys, date_col], how="left")
            )

            proj_exprs = []
            for c in roll_cols:
                feat = f"{c}__LAG{lag}_YOY_PROJECTED_BENCHMARK"
                proj_exprs.append(
                    (pl.col(f"{c}__YOY") * pl.col(f"{c}__LAG{lag}") / pl.col(f"{c}__YOYLAG{lag}")).alias(feat)
                )

            proj_cols = [f"{c}__LAG{lag}_YOY_PROJECTED_BENCHMARK" for c in roll_cols]
            result = result.join(
                joined.with_columns(proj_exprs).select([*keys, date_col, *proj_cols]),
                on=[*keys, date_col],
                how="left",
            )

        return result


class IBESConsensusSalesGrowthBenchmark(Benchmark):
    """External IBES consensus sales growth, joined via join_asof on DATADATE per entity."""

    COLUMN_MAP = {
        "quarterly": "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK",
        "semiannual": "IBES_SEMIANNUAL_SALES_GROWTH_BENCHMARK",
    }

    def __init__(self, params: Param, freq: str = "quarterly") -> None:
        if freq not in self.COLUMN_MAP:
            raise ValueError(f"freq must be 'quarterly' or 'semiannual', got {freq!r}")
        self.params = params
        self.freq = freq

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if external_dfs is None or "ibes" not in external_dfs:
            raise ValueError("IBESConsensusSalesGrowthBenchmark requires external_dfs['ibes']")

        keys = self.params.keys
        date_col = self.params.date_col
        ibes_df = external_dfs["ibes"]
        bench_col = self.COLUMN_MAP[self.freq]

        result = base_df.select([*keys, date_col])
        for key in keys:
            # join_asof per entity: sort both by date, join backward
            base_entity = result.sort(date_col)
            ibes_entity = ibes_df.sort(date_col)
            joined = base_entity.join_asof(
                ibes_entity.select([*keys, date_col, bench_col]),
                on=date_col,
                by=keys,
                strategy="backward",
            )
            return joined.select([*keys, date_col, bench_col])

        return result


class CohortRollingSumBenchmark(Benchmark):
    """Mean or median of rolling sum values within a cohort group per date."""

    def __init__(self, params: Param, cohort_cols: list[str], agg: str = "mean") -> None:
        if agg not in ("mean", "median"):
            raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
        self.params = params
        self.cohort_cols = cohort_cols
        self.agg = agg

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        roll_cols = [c for c in base_df.columns if "__ROLLING_SUM__" in c]

        # Join cohort columns from raw_df onto base_df
        enriched = base_df.join(
            raw_df.select([*keys, date_col, *self.cohort_cols]).unique([*keys, date_col]),
            on=[*keys, date_col],
            how="left",
        )

        agg_fn = pl.mean if self.agg == "mean" else pl.median
        cohort_agg = (
            enriched
            .group_by([*self.cohort_cols, date_col])
            .agg([agg_fn(c).alias(f"{c}__COHORT_BENCHMARK") for c in roll_cols])
        )

        result = enriched.join(cohort_agg, on=[*self.cohort_cols, date_col], how="left")
        bench_cols = [f"{c}__COHORT_BENCHMARK" for c in roll_cols]
        return result.select([*keys, date_col, *bench_cols])
```

- [ ] **Step 4: Run all benchmark tests**

```bash
uv run pytest tests/test_benchmarks.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/benchmarks.py tests/test_benchmarks.py
git commit -m "feat: add LaggedRollingSum, LaggedYOYProjected, IBES, Cohort benchmarks"
```

---

## Task 8: Simple Comparisons (`comparisons.py`)

**Files:**
- Create: `kpi_metrics/comparisons.py`
- Create: `tests/test_comparisons.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_comparisons.py`:
```python
import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.comparisons import DifferenceComparison, SignComparison, RatioComparison


@pytest.fixture
def aligned_dfs():
    base_df = pl.DataFrame({
        "specid": ["A", "A", "A"],
        "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
        "SALES__YOY_GROWTH__DAILY": [0.1, 0.2, -0.1],
    })
    bench_df = pl.DataFrame({
        "specid": ["A", "A", "A"],
        "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
        "ZERO_BENCHMARK": [0.0, 0.0, 0.0],
    })
    return base_df, bench_df


class TestDifferenceComparison:
    def test_difference_values(self, aligned_dfs):
        base_df, bench_df = aligned_dfs
        result = DifferenceComparison().compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__DIFFERENCE"
        assert col in result.columns
        assert result[col].to_list() == pytest.approx([0.1, 0.2, -0.1])

    def test_output_has_only_keys_date_features(self, aligned_dfs):
        base_df, bench_df = aligned_dfs
        result = DifferenceComparison().compute(base_df, bench_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns
        assert "SALES__YOY_GROWTH__DAILY" not in result.columns


class TestSignComparison:
    def test_sign_values(self, aligned_dfs):
        base_df, bench_df = aligned_dfs
        result = SignComparison().compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__SIGN"
        assert result[col].to_list() == pytest.approx([1.0, 1.0, -1.0])


class TestRatioComparison:
    def test_ratio_values(self):
        base_df = pl.DataFrame({
            "specid": ["A", "A"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 2)],
            "SALES__YOY_GROWTH__DAILY": [0.3, 0.6],
        })
        bench_df = pl.DataFrame({
            "specid": ["A", "A"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 2)],
            "LAGGED_BENCHMARK": [0.1, 0.2],
        })
        result = RatioComparison().compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__LAGGED_BENCHMARK__RATIO"
        # (0.3 - 0.1) / 0.1 = 2.0; (0.6 - 0.2) / 0.2 = 2.0
        assert result[col].to_list() == pytest.approx([2.0, 2.0])
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_comparisons.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement simple comparisons**

Write `kpi_metrics/comparisons.py`:
```python
from __future__ import annotations

import polars as pl

from kpi_metrics.params import Comparison, Param


def _get_feature_and_bench_cols(
    base_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    keys: list[str],
    date_col: str,
) -> tuple[list[str], list[str]]:
    """Return (feature_cols, benchmark_cols) — columns that are not keys or date."""
    exclude = set(keys) | {date_col}
    feature_cols = [c for c in base_df.columns if c not in exclude]
    bench_cols = [c for c in benchmark_df.columns if c not in exclude]
    return feature_cols, bench_cols


def _align(
    base_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    keys: list[str],
    date_col: str,
) -> pl.DataFrame:
    """Join base_df with benchmark_df on keys + date_col."""
    return base_df.join(benchmark_df, on=[*keys, date_col], how="left")


class DifferenceComparison(Comparison):
    """base - benchmark for each (base_col, bench_col) pair."""

    def compute(
        self,
        base_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> pl.DataFrame:
        # Infer keys/date_col from shared columns
        shared = [c for c in base_df.columns if c in benchmark_df.columns]
        keys_and_date = shared  # all shared cols are keys + date_col
        feature_cols = [c for c in base_df.columns if c not in keys_and_date]
        bench_cols = [c for c in benchmark_df.columns if c not in keys_and_date]

        joined = base_df.join(benchmark_df, on=keys_and_date, how="left")
        exprs = []
        result_cols = []
        for fc in feature_cols:
            for bc in bench_cols:
                name = f"{fc}__VS__{bc}__DIFFERENCE"
                exprs.append((pl.col(fc) - pl.col(bc)).alias(name))
                result_cols.append(name)

        return joined.with_columns(exprs).select([*keys_and_date, *result_cols])


class SignComparison(Comparison):
    """sign(base - benchmark) for each (base_col, bench_col) pair."""

    def compute(
        self,
        base_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> pl.DataFrame:
        shared = [c for c in base_df.columns if c in benchmark_df.columns]
        feature_cols = [c for c in base_df.columns if c not in shared]
        bench_cols = [c for c in benchmark_df.columns if c not in shared]

        joined = base_df.join(benchmark_df, on=shared, how="left")
        exprs = []
        result_cols = []
        for fc in feature_cols:
            for bc in bench_cols:
                name = f"{fc}__VS__{bc}__SIGN"
                exprs.append((pl.col(fc) - pl.col(bc)).sign().cast(pl.Float64).alias(name))
                result_cols.append(name)

        return joined.with_columns(exprs).select([*shared, *result_cols])


class RatioComparison(Comparison):
    """(base - benchmark) / benchmark for each (base_col, bench_col) pair."""

    def compute(
        self,
        base_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> pl.DataFrame:
        shared = [c for c in base_df.columns if c in benchmark_df.columns]
        feature_cols = [c for c in base_df.columns if c not in shared]
        bench_cols = [c for c in benchmark_df.columns if c not in shared]

        joined = base_df.join(benchmark_df, on=shared, how="left")
        exprs = []
        result_cols = []
        for fc in feature_cols:
            for bc in bench_cols:
                name = f"{fc}__VS__{bc}__RATIO"
                exprs.append(
                    ((pl.col(fc) - pl.col(bc)) / pl.col(bc)).alias(name)
                )
                result_cols.append(name)

        return joined.with_columns(exprs).select([*shared, *result_cols])
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_comparisons.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/comparisons.py tests/test_comparisons.py
git commit -m "feat: add DifferenceComparison, SignComparison, RatioComparison"
```

---

## Task 9: ZScoreComparison & RankComparison (`comparisons.py`)

**Files:**
- Modify: `kpi_metrics/comparisons.py`
- Modify: `tests/test_comparisons.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/test_comparisons.py`:
```python
from kpi_metrics.params import Param
from kpi_metrics.comparisons import ZScoreComparison, RankComparison


@pytest.fixture
def rolling_aligned_dfs():
    """10-row aligned pair for rolling window tests."""
    base_df = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES__YOY_GROWTH__DAILY": [float(i) for i in range(10)],
    })
    bench_df = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "ZERO_BENCHMARK": [0.0] * 10,
    })
    return base_df, bench_df


class TestZScoreComparison:
    def test_output_columns(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = ZScoreComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__ZSCORE"
        assert col in result.columns

    def test_zscore_nulls_for_insufficient_window(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = ZScoreComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__ZSCORE"
        # First 4 rows (< window=5) should be null
        vals = result[col].to_list()
        assert all(v is None for v in vals[:4])

    def test_zscore_last_value_reasonable(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = ZScoreComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__ZSCORE"
        assert result[col][-1] is not None


class TestRankComparison:
    def test_output_columns(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = RankComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RANK"
        assert col in result.columns

    def test_rank_range(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = RankComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RANK"
        vals = [v for v in result[col].to_list() if v is not None]
        assert all(-1.0 <= v <= 1.0 for v in vals)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_comparisons.py::TestZScoreComparison tests/test_comparisons.py::TestRankComparison -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement ZScoreComparison and RankComparison**

Add to `kpi_metrics/comparisons.py`:
```python
class ZScoreComparison(Comparison):
    """Rolling z-score of (base - benchmark) grouped by params.keys."""

    def __init__(self, params: Param, lookback_window: int) -> None:
        self.params = params
        self.lookback_window = lookback_window

    def compute(
        self,
        base_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        shared = [c for c in base_df.columns if c in benchmark_df.columns]
        feature_cols = [c for c in base_df.columns if c not in shared]
        bench_cols = [c for c in benchmark_df.columns if c not in shared]

        joined = base_df.join(benchmark_df, on=shared, how="left")

        diff_exprs = []
        diff_cols = []
        for fc in feature_cols:
            for bc in bench_cols:
                diff_col = f"__diff__{fc}__{bc}"
                diff_exprs.append((pl.col(fc) - pl.col(bc)).alias(diff_col))
                diff_cols.append((diff_col, fc, bc))

        joined = joined.with_columns(diff_exprs).sort([*keys, date_col])

        zscore_exprs = []
        result_cols = []
        for diff_col, fc, bc in diff_cols:
            name = f"{fc}__VS__{bc}__ZSCORE"
            mean_col = pl.col(diff_col).rolling_mean(
                window_size=self.lookback_window, min_periods=self.lookback_window
            ).over(keys)
            std_col = pl.col(diff_col).rolling_std(
                window_size=self.lookback_window, min_periods=self.lookback_window
            ).over(keys)
            zscore_exprs.append(
                ((pl.col(diff_col) - mean_col) / std_col).alias(name)
            )
            result_cols.append(name)

        return joined.with_columns(zscore_exprs).select([*shared, *result_cols])


class RankComparison(Comparison):
    """Cross-sectional [-1, 1] rank of (base - benchmark) over a rolling window grouped by params.keys."""

    def __init__(self, params: Param, lookback_window: int) -> None:
        self.params = params
        self.lookback_window = lookback_window

    def compute(
        self,
        base_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        shared = [c for c in base_df.columns if c in benchmark_df.columns]
        feature_cols = [c for c in base_df.columns if c not in shared]
        bench_cols = [c for c in benchmark_df.columns if c not in shared]

        joined = base_df.join(benchmark_df, on=shared, how="left")

        diff_exprs = []
        diff_cols = []
        for fc in feature_cols:
            for bc in bench_cols:
                diff_col = f"__diff__{fc}__{bc}"
                diff_exprs.append((pl.col(fc) - pl.col(bc)).alias(diff_col))
                diff_cols.append((diff_col, fc, bc))

        joined = joined.with_columns(diff_exprs).sort([*keys, date_col])

        rank_exprs = []
        result_cols = []
        for diff_col, fc, bc in diff_cols:
            name = f"{fc}__VS__{bc}__RANK"
            # Percentile rank [0,1] using rolling window, scaled to [-1, 1]
            pct_col = pl.col(diff_col).rolling_map(
                function=lambda s: (s.rank(method="average") / len(s) * 2 - 1).tail(1),
                window_size=self.lookback_window,
                min_periods=self.lookback_window,
            ).over(keys)
            rank_exprs.append(pct_col.alias(name))
            result_cols.append(name)

        return joined.with_columns(rank_exprs).select([*shared, *result_cols])
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_comparisons.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/comparisons.py tests/test_comparisons.py
git commit -m "feat: add ZScoreComparison and RankComparison"
```

---

## Task 10: ResidualComparison (`comparisons.py`)

**Files:**
- Modify: `kpi_metrics/comparisons.py`
- Modify: `tests/test_comparisons.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/test_comparisons.py`:
```python
from kpi_metrics.comparisons import ResidualComparison


class TestResidualComparison:
    def test_output_columns(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = ResidualComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RESIDUAL"
        assert col in result.columns

    def test_residuals_null_before_window(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = ResidualComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RESIDUAL"
        vals = result[col].to_list()
        assert all(v is None for v in vals[:4])

    def test_residuals_not_null_after_window(self, rolling_aligned_dfs):
        params = Param(keys=["specid"], metrics=["SALES"])
        base_df, bench_df = rolling_aligned_dfs
        result = ResidualComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RESIDUAL"
        vals = result[col].to_list()
        assert result[col][-1] is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_comparisons.py::TestResidualComparison -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement ResidualComparison**

Add to `kpi_metrics/comparisons.py` (requires `import numpy as np` at top of file):
```python
import numpy as np


class ResidualComparison(Comparison):
    """OLS residual of (base ~ benchmark + const) over a rolling window, grouped by params.keys."""

    def __init__(self, params: Param, lookback_window: int) -> None:
        self.params = params
        self.lookback_window = lookback_window

    def compute(
        self,
        base_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        shared = [c for c in base_df.columns if c in benchmark_df.columns]
        feature_cols = [c for c in base_df.columns if c not in shared]
        bench_cols = [c for c in benchmark_df.columns if c not in shared]

        joined = base_df.join(benchmark_df, on=shared, how="left").sort([*keys, date_col])

        result_rows = []
        W = self.lookback_window

        for fc in feature_cols:
            for bc in bench_cols:
                out_col = f"{fc}__VS__{bc}__RESIDUAL"
                residuals = []

                # Process group by group
                for group_keys, group_df in joined.group_by(keys, maintain_order=True):
                    y = group_df[fc].to_numpy()
                    x = group_df[bc].to_numpy()
                    n = len(y)
                    resid = np.full(n, np.nan)

                    for i in range(W - 1, n):
                        y_win = y[i - W + 1: i + 1]
                        x_win = x[i - W + 1: i + 1]
                        if np.any(np.isnan(y_win)) or np.any(np.isnan(x_win)):
                            continue
                        X = np.column_stack([x_win, np.ones(W)])
                        try:
                            beta, *_ = np.linalg.lstsq(X, y_win, rcond=None)
                            resid[i] = y[i] - (x[i] * beta[0] + beta[1])
                        except np.linalg.LinAlgError:
                            pass

                    residuals.extend(resid.tolist())

                # Assign residuals back in sorted order
                joined = joined.with_columns(
                    pl.Series(out_col, [None if np.isnan(v) else v for v in residuals])
                )

        result_cols = [
            f"{fc}__VS__{bc}__RESIDUAL"
            for fc in feature_cols
            for bc in bench_cols
        ]
        return joined.select([*shared, *result_cols])
```

- [ ] **Step 4: Run all comparison tests**

```bash
uv run pytest tests/test_comparisons.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/comparisons.py tests/test_comparisons.py
git commit -m "feat: add ResidualComparison with rolling OLS"
```

---

## Task 11: Transformations (`transformations.py`)

**Files:**
- Create: `kpi_metrics/transformations.py`
- Create: `tests/test_transformations.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_transformations.py`:
```python
import polars as pl
import pytest
from datetime import date
from kpi_metrics.transformations import (
    ClipTransformation,
    WinsorizeTransformation,
    RampUpTransformation,
    RampDownTransformation,
)


@pytest.fixture
def feature_df():
    return pl.DataFrame({
        "specid": ["A", "A", "A", "B", "B", "B"],
        "DATADATE": [date(2023, 1, i + 1) for i in range(3)] * 2,
        "FEAT": [-5.0, 0.5, 10.0, -3.0, 1.0, 8.0],
    })


class TestClipTransformation:
    def test_clip_values(self, feature_df):
        result = ClipTransformation(lower=-2.0, upper=5.0).compute(feature_df)
        assert result["FEAT"].to_list() == pytest.approx([-2.0, 0.5, 5.0, -2.0, 1.0, 5.0])

    def test_preserves_keys_and_date(self, feature_df):
        result = ClipTransformation(lower=-2.0, upper=5.0).compute(feature_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns


class TestWinsorizeTransformation:
    def test_winsorize_clips_extremes(self):
        df = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "FEAT": [float(i) for i in range(10)],  # 0..9
        })
        result = WinsorizeTransformation(lower_q=0.1, upper_q=0.9).compute(df)
        vals = result["FEAT"].to_list()
        assert min(vals) >= 0.9  # 10th percentile of 0..9 = 0.9
        assert max(vals) <= 8.1  # 90th percentile = 8.1


class TestRampUpTransformation:
    def test_ramp_up_zero_at_min(self):
        df = pl.DataFrame({
            "specid": ["A", "A", "A"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
            "FEAT": [1.0, 1.0, 1.0],
            "age": [0, 5, 10],
        })
        result = RampUpTransformation(age_col="age", min_age=0, max_age=10).compute(df)
        vals = result["FEAT"].to_list()
        assert vals[0] == pytest.approx(0.0)   # age=0 → scale=0
        assert vals[1] == pytest.approx(0.5)   # age=5 → scale=0.5
        assert vals[2] == pytest.approx(1.0)   # age=10 → scale=1.0


class TestRampDownTransformation:
    def test_ramp_down_one_at_min(self):
        df = pl.DataFrame({
            "specid": ["A", "A", "A"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
            "FEAT": [1.0, 1.0, 1.0],
            "age": [0, 5, 10],
        })
        result = RampDownTransformation(age_col="age", min_age=0, max_age=10).compute(df)
        vals = result["FEAT"].to_list()
        assert vals[0] == pytest.approx(1.0)   # age=0 → scale=1.0
        assert vals[1] == pytest.approx(0.5)   # age=5 → scale=0.5
        assert vals[2] == pytest.approx(0.0)   # age=10 → scale=0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_transformations.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement transformations**

Write `kpi_metrics/transformations.py`:
```python
from __future__ import annotations

import polars as pl

from kpi_metrics.params import Transformation


def _feature_cols(df: pl.DataFrame, non_feature: set[str]) -> list[str]:
    return [c for c in df.columns if c not in non_feature]


class ClipTransformation(Transformation):
    """Clip all feature columns to [lower, upper]."""

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = lower
        self.upper = upper

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        key_cols = {"specid", "DATADATE"}  # detect by convention
        feat_cols = _feature_cols(df, key_cols)
        return df.with_columns([
            pl.col(c).clip(self.lower, self.upper) for c in feat_cols
        ])


class WinsorizeTransformation(Transformation):
    """Clip feature columns to [lower_q, upper_q] quantiles computed across all rows."""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        key_cols = {"specid", "DATADATE"}
        feat_cols = _feature_cols(df, key_cols)
        exprs = []
        for c in feat_cols:
            lo = df[c].quantile(self.lower_q, interpolation="linear")
            hi = df[c].quantile(self.upper_q, interpolation="linear")
            exprs.append(pl.col(c).clip(lo, hi))
        return df.with_columns(exprs)


class RampUpTransformation(Transformation):
    """Scale feature columns from 0→1 based on entity age: scale = (age - min_age) / (max_age - min_age), clipped to [0,1]."""

    def __init__(self, age_col: str, min_age: int, max_age: int) -> None:
        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        key_cols = {"specid", "DATADATE", self.age_col}
        feat_cols = _feature_cols(df, key_cols)
        scale = (
            ((pl.col(self.age_col) - self.min_age) / (self.max_age - self.min_age))
            .clip(0.0, 1.0)
        )
        return df.with_columns([
            (pl.col(c) * scale).alias(c) for c in feat_cols
        ])


class RampDownTransformation(Transformation):
    """Scale feature columns from 1→0 based on entity age: scale = 1 - (age - min_age) / (max_age - min_age), clipped to [0,1]."""

    def __init__(self, age_col: str, min_age: int, max_age: int) -> None:
        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        key_cols = {"specid", "DATADATE", self.age_col}
        feat_cols = _feature_cols(df, key_cols)
        scale = (
            1.0 - ((pl.col(self.age_col) - self.min_age) / (self.max_age - self.min_age))
        ).clip(0.0, 1.0)
        return df.with_columns([
            (pl.col(c) * scale).alias(c) for c in feat_cols
        ])
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_transformations.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/transformations.py tests/test_transformations.py
git commit -m "feat: add Clip, Winsorize, RampUp, RampDown transformations"
```

---

## Task 12: KpiMetricsBuilder (`builder.py`)

**Files:**
- Create: `kpi_metrics/builder.py`
- Create: `tests/test_builder.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_builder.py`:
```python
import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.base_features import YOYGrowthBaseFeature
from kpi_metrics.benchmarks import ZeroBenchmark, LaggedYOYGrowthBenchmark
from kpi_metrics.comparisons import DifferenceComparison
from kpi_metrics.transformations import ClipTransformation
from kpi_metrics.builder import KpiMetricsBuilder


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def raw_df():
    n = 20
    return pl.DataFrame({
        "specid": ["A"] * n + ["B"] * n,
        "DATADATE": [date(2022, 1, i + 1) for i in range(n)] * 2,
        "SALES": [float(i + 1) for i in range(n)] * 2,
    })


class TestKpiMetricsBuilder:
    def test_compute_returns_dataframe(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert isinstance(result, pl.DataFrame)

    def test_output_contains_keys_and_date(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns

    def test_output_has_feature_columns(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        feat_cols = [c for c in result.columns if "VS" in c]
        assert len(feat_cols) > 0

    def test_no_raw_metrics_in_output(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert "SALES" not in result.columns

    def test_multiple_benchmarks_and_comparisons(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7),
            benchmarks=[
                ZeroBenchmark(params),
                LaggedYOYGrowthBenchmark(params, lags=[3]),
            ],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        feat_cols = [c for c in result.columns if "VS" in c]
        assert len(feat_cols) >= 2

    def test_transformation_applied(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
            transformations=[ClipTransformation(lower=-0.5, upper=0.5)],
        )
        result = builder.compute(raw_df)
        feat_cols = [c for c in result.columns if "VS" in c]
        for col in feat_cols:
            vals = [v for v in result[col].to_list() if v is not None]
            assert all(-0.5 <= v <= 0.5 for v in vals)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_builder.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement KpiMetricsBuilder**

Write `kpi_metrics/builder.py`:
```python
from __future__ import annotations

import polars as pl

from kpi_metrics.params import BaseFeature, Benchmark, Comparison, Param, Transformation


class KpiMetricsBuilder:
    """Orchestrates base feature → benchmarks → comparisons → transformations pipeline."""

    def __init__(
        self,
        params: Param,
        base_feature: BaseFeature,
        benchmarks: list[Benchmark],
        comparisons: list[Comparison],
        transformations: list[Transformation] | None = None,
    ) -> None:
        self.params = params
        self.base_feature = base_feature
        self.benchmarks = benchmarks
        self.comparisons = comparisons
        self.transformations = transformations or []

    def compute(
        self,
        df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Run the full pipeline.

        Args:
            df: raw daily input DataFrame, sorted by (keys, DATADATE).
            external_dfs: optional external DataFrames keyed by name (e.g. {"ibes": ...}).

        Returns:
            DataFrame with keys + DATADATE + all feature columns.
        """
        keys = self.params.keys
        date_col = self.params.date_col

        # Step 1: compute base features
        base_df = self.base_feature.compute(df)

        # Step 2 & 3: for each benchmark, compute and then apply each comparison
        all_feature_dfs: list[pl.DataFrame] = []

        for benchmark in self.benchmarks:
            bench_df = benchmark.compute(base_df, df, external_dfs=external_dfs)

            for comparison in self.comparisons:
                feat_df = comparison.compute(base_df, bench_df)
                all_feature_dfs.append(feat_df)

        if not all_feature_dfs:
            return base_df.select([*keys, date_col])

        # Step 4: merge all feature DataFrames
        result = all_feature_dfs[0]
        for feat_df in all_feature_dfs[1:]:
            new_cols = [c for c in feat_df.columns if c not in result.columns]
            if new_cols:
                result = result.join(
                    feat_df.select([*keys, date_col, *new_cols]),
                    on=[*keys, date_col],
                    how="left",
                )

        # Step 5: apply transformations
        for transformation in self.transformations:
            result = transformation.compute(result)

        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_builder.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kpi_metrics/builder.py tests/test_builder.py
git commit -m "feat: add KpiMetricsBuilder"
```

---

## Task 13: Presets & Public API

**Files:**
- Create: `kpi_metrics/presets.py`
- Create: `tests/test_presets.py`
- Modify: `kpi_metrics/__init__.py`

- [ ] **Step 1: Write failing tests**

Write `tests/test_presets.py`:
```python
import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.presets import (
    default_yoy_pipeline,
    default_pop_pipeline,
    default_rolling_sum_pipeline,
)
from kpi_metrics.builder import KpiMetricsBuilder


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def raw_df():
    n = 200
    return pl.DataFrame({
        "specid": ["A"] * n,
        "DATADATE": [date(2022, 1, 1).__class__(2022, 1, 1).__class__.fromordinal(
            date(2022, 1, 1).toordinal() + i
        ) for i in range(n)],
        "SALES": [float(i + 1) for i in range(n)],
    })


class TestPresets:
    def test_default_yoy_pipeline_returns_builder(self, params):
        builder = default_yoy_pipeline(params)
        assert isinstance(builder, KpiMetricsBuilder)

    def test_default_pop_pipeline_returns_builder(self, params):
        builder = default_pop_pipeline(params)
        assert isinstance(builder, KpiMetricsBuilder)

    def test_default_rolling_sum_pipeline_returns_builder(self, params):
        builder = default_rolling_sum_pipeline(params)
        assert isinstance(builder, KpiMetricsBuilder)

    def test_default_yoy_pipeline_computes(self, params, raw_df):
        result = default_yoy_pipeline(params).compute(raw_df)
        assert isinstance(result, pl.DataFrame)
        assert "specid" in result.columns
        feat_cols = [c for c in result.columns if "VS" in c]
        assert len(feat_cols) > 0

    def test_default_pop_pipeline_computes(self, params, raw_df):
        result = default_pop_pipeline(params).compute(raw_df)
        assert isinstance(result, pl.DataFrame)
        feat_cols = [c for c in result.columns if "VS" in c]
        assert len(feat_cols) > 0

    def test_default_rolling_sum_pipeline_computes(self, params, raw_df):
        result = default_rolling_sum_pipeline(params).compute(raw_df)
        assert isinstance(result, pl.DataFrame)
        feat_cols = [c for c in result.columns if "VS" in c]
        assert len(feat_cols) > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_presets.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement presets**

Write `kpi_metrics/presets.py`:
```python
from __future__ import annotations

from kpi_metrics.base_features import (
    POPGrowthBaseFeature,
    RollingSumBaseFeature,
    YOYGrowthBaseFeature,
)
from kpi_metrics.benchmarks import (
    LaggedPOPGrowthBenchmark,
    LaggedRollingSumBenchmark,
    LaggedYOYGrowthBenchmark,
    ZeroBenchmark,
)
from kpi_metrics.builder import KpiMetricsBuilder
from kpi_metrics.comparisons import DifferenceComparison, RankComparison, ZScoreComparison
from kpi_metrics.params import Param


def default_yoy_pipeline(params: Param) -> KpiMetricsBuilder:
    """YoY growth vs Zero and Lagged benchmarks, with Difference and ZScore comparisons."""
    return KpiMetricsBuilder(
        params=params,
        base_feature=YOYGrowthBaseFeature(params),
        benchmarks=[
            ZeroBenchmark(params),
            LaggedYOYGrowthBenchmark(params, lags=[7, 14, 28, 91]),
        ],
        comparisons=[
            DifferenceComparison(),
            ZScoreComparison(params, lookback_window=364),
        ],
    )


def default_pop_pipeline(params: Param) -> KpiMetricsBuilder:
    """PoP growth vs Zero and Lagged benchmarks, with Difference comparison."""
    return KpiMetricsBuilder(
        params=params,
        base_feature=POPGrowthBaseFeature(params),
        benchmarks=[
            ZeroBenchmark(params),
            LaggedPOPGrowthBenchmark(params, lags=[7, 14, 28, 91]),
        ],
        comparisons=[
            DifferenceComparison(),
        ],
    )


def default_rolling_sum_pipeline(params: Param) -> KpiMetricsBuilder:
    """Rolling sum vs Lagged benchmarks, with Difference and Rank comparisons."""
    return KpiMetricsBuilder(
        params=params,
        base_feature=RollingSumBaseFeature(params),
        benchmarks=[
            LaggedRollingSumBenchmark(params, lags=[7, 14, 28, 91]),
        ],
        comparisons=[
            DifferenceComparison(),
            RankComparison(params, lookback_window=364),
        ],
    )
```

- [ ] **Step 4: Write public API**

Write `kpi_metrics/__init__.py`:
```python
from kpi_metrics.params import Param
from kpi_metrics.base_features import (
    RollingSumBaseFeature,
    YOYGrowthBaseFeature,
    POPGrowthBaseFeature,
)
from kpi_metrics.benchmarks import (
    ZeroBenchmark,
    LaggedYOYGrowthBenchmark,
    LaggedPOPGrowthBenchmark,
    LaggedRollingSumBenchmark,
    LaggedYOYProjectedRollingSumBenchmark,
    IBESConsensusSalesGrowthBenchmark,
    CohortRollingSumBenchmark,
)
from kpi_metrics.comparisons import (
    DifferenceComparison,
    SignComparison,
    RatioComparison,
    ResidualComparison,
    ZScoreComparison,
    RankComparison,
)
from kpi_metrics.transformations import (
    ClipTransformation,
    WinsorizeTransformation,
    RampUpTransformation,
    RampDownTransformation,
)
from kpi_metrics.builder import KpiMetricsBuilder
from kpi_metrics.presets import (
    default_yoy_pipeline,
    default_pop_pipeline,
    default_rolling_sum_pipeline,
)

__all__ = [
    "Param",
    "RollingSumBaseFeature",
    "YOYGrowthBaseFeature",
    "POPGrowthBaseFeature",
    "ZeroBenchmark",
    "LaggedYOYGrowthBenchmark",
    "LaggedPOPGrowthBenchmark",
    "LaggedRollingSumBenchmark",
    "LaggedYOYProjectedRollingSumBenchmark",
    "IBESConsensusSalesGrowthBenchmark",
    "CohortRollingSumBenchmark",
    "DifferenceComparison",
    "SignComparison",
    "RatioComparison",
    "ResidualComparison",
    "ZScoreComparison",
    "RankComparison",
    "ClipTransformation",
    "WinsorizeTransformation",
    "RampUpTransformation",
    "RampDownTransformation",
    "KpiMetricsBuilder",
    "default_yoy_pipeline",
    "default_pop_pipeline",
    "default_rolling_sum_pipeline",
]
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add kpi_metrics/presets.py kpi_metrics/__init__.py tests/test_presets.py
git commit -m "feat: add presets and public API — kpi_metrics complete"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `Param` dataclass with `keys`, `metrics`, `date_col="DATADATE"` — Task 2
- [x] Abstract base classes with `.compute()` — Task 2
- [x] `RollingSumBaseFeature` with default windows — Task 3
- [x] `YOYGrowthBaseFeature` with 364-day YoY — Task 4
- [x] `POPGrowthBaseFeature` — Task 5
- [x] `ZeroBenchmark` — Task 6
- [x] `LaggedYOYGrowthBenchmark`, `LaggedPOPGrowthBenchmark` — Task 6
- [x] `LaggedRollingSumBenchmark`, `LaggedYOYProjectedRollingSumBenchmark` — Task 7
- [x] `IBESConsensusSalesGrowthBenchmark`, `CohortRollingSumBenchmark` — Task 7
- [x] `DifferenceComparison`, `SignComparison`, `RatioComparison` — Task 8
- [x] `ZScoreComparison`, `RankComparison` — Task 9
- [x] `ResidualComparison` — Task 10
- [x] `ClipTransformation`, `WinsorizeTransformation`, `RampUpTransformation`, `RampDownTransformation` — Task 11
- [x] `KpiMetricsBuilder` with full pipeline execution — Task 12
- [x] Preset pipelines: `default_yoy_pipeline`, `default_pop_pipeline`, `default_rolling_sum_pipeline` — Task 13
- [x] Output column naming: `{METRIC}__{BASE_FEATURE}__VS__{BENCHMARK}__{COMPARISON}` — Tasks 8-10
- [x] `external_dfs` plumbed through builder to benchmarks — Tasks 7, 12
- [x] Polars throughout — all tasks
- [x] `uv` + hatchling — Task 1
