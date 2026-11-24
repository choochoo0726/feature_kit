from __future__ import annotations

import polars as pl

from feature_kit.params import BaseFeature, Benchmark, Comparison, Param, Transformation


class FeatureBuilder:
    """Orchestrate base_feature → benchmarks → comparisons → transformations."""

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
        keys = self.params.keys
        date_col = self.params.date_col

        base_df = self.base_feature.compute(df)

        all_comparison_dfs: list[pl.DataFrame] = []
        for benchmark in self.benchmarks:
            bench_df = benchmark.compute(base_df, df, external_dfs)
            for comparison in self.comparisons:
                comp_df = comparison.compute(base_df, bench_df)
                all_comparison_dfs.append(comp_df)

        # Merge all comparison DataFrames on shared keys + date_col
        shared = [*keys, date_col]
        result = all_comparison_dfs[0]
        for other in all_comparison_dfs[1:]:
            new_cols = [c for c in other.columns if c not in result.columns]
            if new_cols:
                result = result.join(other.select([*shared, *new_cols]), on=shared, how="left")

        # Apply transformations sequentially
        for transformation in self.transformations:
            result = transformation.compute(result)

        return result
