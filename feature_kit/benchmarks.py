from __future__ import annotations

import polars as pl

from feature_kit.params import Benchmark, Param

_DEFAULT_PERIODS: list[tuple[str, int]] = [("week", 7), ("month", 28), ("quarter", 91)]


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
    """YoY growth feature values lagged by N multiples of named periods.

    Each entry in *periods* is a ``(name, days)`` pair.  For every period the
    benchmark is computed for lags 1 × days, 2 × days, …, num_lags × days,
    producing columns named ``{feature}__LAG{n}{NAME}``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags

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
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df
                    .select([*keys, date_col, *yoy_cols])
                    .with_columns(
                        (pl.col(date_col) + pl.duration(days=lag)).alias(date_col)
                    )
                    .rename({c: f"{c}__{suffix}" for c in yoy_cols})
                )
                bench_cols = [f"{c}__{suffix}" for c in yoy_cols]
                result = result.join(
                    lag_df.select([*keys, date_col, *bench_cols]),
                    on=[*keys, date_col],
                    how="left",
                )
        return result


class LaggedPOPGrowthBenchmark(Benchmark):
    """PoP growth feature values lagged by N multiples of named periods.

    Each entry in *periods* is a ``(name, days)`` pair.  For every period the
    benchmark is computed for lags 1 × days, 2 × days, …, num_lags × days,
    producing columns named ``{feature}__LAG{n}{NAME}``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags

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
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df
                    .select([*keys, date_col, *pop_cols])
                    .with_columns(
                        (pl.col(date_col) + pl.duration(days=lag)).alias(date_col)
                    )
                    .rename({c: f"{c}__{suffix}" for c in pop_cols})
                )
                bench_cols = [f"{c}__{suffix}" for c in pop_cols]
                result = result.join(
                    lag_df.select([*keys, date_col, *bench_cols]),
                    on=[*keys, date_col],
                    how="left",
                )
        return result


class LaggedRollingSumBenchmark(Benchmark):
    """Rolling sum values lagged by N multiples of named periods.

    Each entry in *periods* is a ``(name, days)`` pair.  For every period the
    benchmark is computed for lags 1 × days, 2 × days, …, num_lags × days,
    producing columns named ``{feature}__LAG{n}{NAME}``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags

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
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df.select([*keys, date_col, *roll_cols])
                    .with_columns((pl.col(date_col) + pl.duration(days=lag)).alias(date_col))
                    .rename({c: f"{c}__{suffix}" for c in roll_cols})
                )
                bench_cols = [f"{c}__{suffix}" for c in roll_cols]
                result = result.join(lag_df.select([*keys, date_col, *bench_cols]), on=[*keys, date_col], how="left")
        return result


class LaggedYOYProjectedRollingSumBenchmark(Benchmark):
    """Projected rolling sum: Sum_{yoy_days ago} * Sum_{lag ago} / Sum_{(yoy_days+lag) ago}.

    Each entry in *periods* is a ``(name, days)`` pair.  For every period the
    projection is computed for lags 1 × days, 2 × days, …, num_lags × days,
    producing columns named ``{feature}__LAG{n}{NAME}_YOY_PROJECTED``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
        yoy_days: int = 364,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags
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
        result = base_df.select([*keys, date_col])
        yoy_df = (
            base_df.select([*keys, date_col, *roll_cols])
            .with_columns((pl.col(date_col) + pl.duration(days=self.yoy_days)).alias(date_col))
            .rename({c: f"{c}__YOY" for c in roll_cols})
        )
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df.select([*keys, date_col, *roll_cols])
                    .with_columns((pl.col(date_col) + pl.duration(days=lag)).alias(date_col))
                    .rename({c: f"{c}__{suffix}" for c in roll_cols})
                )
                yoy_lag_df = (
                    base_df.select([*keys, date_col, *roll_cols])
                    .with_columns((pl.col(date_col) + pl.duration(days=self.yoy_days + lag)).alias(date_col))
                    .rename({c: f"{c}__YOY{suffix}" for c in roll_cols})
                )
                joined = (
                    base_df.select([*keys, date_col])
                    .join(yoy_df, on=[*keys, date_col], how="left")
                    .join(lag_df, on=[*keys, date_col], how="left")
                    .join(yoy_lag_df, on=[*keys, date_col], how="left")
                )
                proj_exprs = [
                    (pl.col(f"{c}__YOY") * pl.col(f"{c}__{suffix}") / pl.col(f"{c}__YOY{suffix}"))
                    .alias(f"{c}__{suffix}_YOY_PROJECTED")
                    for c in roll_cols
                ]
                proj_cols = [f"{c}__{suffix}_YOY_PROJECTED" for c in roll_cols]
                result = result.join(
                    joined.with_columns(proj_exprs).select([*keys, date_col, *proj_cols]),
                    on=[*keys, date_col],
                    how="left",
                )
        return result


class IBESConsensusSalesGrowthBenchmark(Benchmark):
    """External IBES consensus sales growth joined via join_asof on DATADATE per entity."""

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
        bench_col = self.COLUMN_MAP[self.freq]
        return (
            base_df.select([*keys, date_col]).sort(date_col)
            .join_asof(
                external_dfs["ibes"].select([*keys, date_col, bench_col]).sort(date_col),
                on=date_col,
                by=keys,
                strategy="backward",
            )
            .select([*keys, date_col, bench_col])
        )


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
        enriched = base_df.join(
            raw_df.select([*keys, date_col, *self.cohort_cols]).unique([*keys, date_col]),
            on=[*keys, date_col],
            how="left",
        )
        agg_fn = pl.mean if self.agg == "mean" else pl.median
        cohort_agg = enriched.group_by([*self.cohort_cols, date_col]).agg(
            [agg_fn(c).alias(f"{c}__COHORT_BENCHMARK") for c in roll_cols]
        )
        bench_cols = [f"{c}__COHORT_BENCHMARK" for c in roll_cols]
        return (
            enriched.join(cohort_agg, on=[*self.cohort_cols, date_col], how="left")
            .select([*keys, date_col, *bench_cols])
        )
