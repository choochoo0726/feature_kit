from __future__ import annotations

import polars as pl

from feature_kit.params import BaseFeature, Benchmark, Comparison, Param, Transformation


class FeatureBuilder:
    """Orchestrate base_feature → benchmarks → comparisons → transformations.

    The builder runs one base feature, then every benchmark, then every
    comparison against every benchmark (a full cross product), merges all
    comparison outputs into one wide frame, and finally applies the
    transformations in order.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    base_feature : BaseFeature
        The single base feature to compute (e.g. ``YOYGrowthBaseFeature``).
    benchmarks : list[Benchmark]
        One or more benchmarks. Must be non-empty.
    comparisons : list[Comparison]
        One or more comparisons, each applied to every benchmark. Must be
        non-empty.
    transformations : list[Transformation], optional
        Applied sequentially to the merged result. Defaults to none.

    Examples
    --------
    >>> from feature_kit import (
    ...     FeatureBuilder, Param, YOYGrowthBaseFeature, ZeroBenchmark,
    ...     DifferenceComparison, ClipTransformation,
    ... )
    >>> params = Param(keys=["specid"], metrics=["SALES"])
    >>> builder = FeatureBuilder(
    ...     params=params,
    ...     base_feature=YOYGrowthBaseFeature(params),
    ...     benchmarks=[ZeroBenchmark(params)],
    ...     comparisons=[DifferenceComparison(params)],
    ...     transformations=[ClipTransformation(-1.0, 1.0)],
    ... )
    >>> signals = builder.compute(raw_df)  # doctest: +SKIP
    """

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
        """Run the full pipeline on a raw daily panel.

        Parameters
        ----------
        df : pl.DataFrame
            Raw input with ``params.keys``, ``params.date_col`` and the
            ``params.metrics`` columns (plus any extra columns that
            benchmarks or transformations need, e.g. cohort or age columns).
        external_dfs : dict[str, pl.DataFrame], optional
            External inputs passed through to every benchmark, e.g.
            ``{"ibes": ibes_df}`` for ``IBESConsensusSalesGrowthBenchmark``.

        Returns
        -------
        pl.DataFrame
            ``keys + date_col`` plus one column per
            (base feature, benchmark column, comparison) combination, named
            ``{base_col}__VS__{bench_col}__{COMPARISON}``.
        """
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
