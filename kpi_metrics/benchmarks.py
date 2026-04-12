from __future__ import annotations
from itertools import product as iproduct

import polars as pl

from kpi_metrics.params import Benchmark, Param


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

    def __init__(self, params: Param, lags: list[int] | None = None) -> None:
        self.params = params
        self.lags = lags if lags is not None else [7, 14, 28, 91]

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

    def __init__(self, params: Param, lags: list[int] | None = None) -> None:
        self.params = params
        self.lags = lags if lags is not None else [7, 14, 28, 91]

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
