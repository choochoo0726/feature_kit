from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl


@dataclass
class Param:
    """Shared configuration describing the layout of the input panel.

    Every pipeline component takes the same ``Param`` so that they agree on
    which columns identify an entity, which column holds the date, and which
    columns are the raw metrics to process.

    Parameters
    ----------
    keys : list[str]
        Entity identifier column(s), e.g. ``["specid"]``. Rolling windows,
        lags and z-scores are all computed within each group of ``keys``.
    metrics : list[str]
        Raw numeric metric columns to turn into features, e.g.
        ``["SALES", "TXN"]``. Each metric name becomes the prefix of every
        column derived from it (``SALES__...``).
    date_col : str, default "DATADATE"
        Observation date column. Must be a Polars ``Date`` (lags are applied
        by adding a ``pl.duration(days=...)``).

    Examples
    --------
    >>> from feature_kit import Param
    >>> params = Param(keys=["specid"], metrics=["SALES"])
    >>> params.date_col
    'DATADATE'
    """

    keys: list[str]
    metrics: list[str]
    date_col: str = "DATADATE"


class BaseFeature(ABC):
    """Stage 1: turn raw daily metrics into base feature columns.

    Subclass this to add a new kind of base feature. Output columns should
    follow the ``{METRIC}__{FEATURE}__{WINDOW}`` naming convention so that
    benchmarks can find them by substring (e.g. ``"__YOY_GROWTH__"``).
    """

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute base features from raw daily DataFrame.

        Args:
            df: sorted by (keys, date_col), contains raw metric columns.

        Returns:
            DataFrame with keys + date_col + feature columns only.
        """


class Benchmark(ABC):
    """Stage 2: produce reference values to compare base features against.

    A benchmark returns one row per ``(keys, date_col)`` row of ``base_df``.
    Benchmark columns derived from a metric should start with that metric's
    name (``SALES__...``) so comparisons only pair them with the same metric;
    columns with no metric prefix (e.g. ``ZERO_BENCHMARK``) pair with every
    feature.
    """

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
    """Stage 3: combine each base feature with each matching benchmark column."""

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
    """Stage 4: post-process comparison columns (clip, winsorize, scale)."""

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply transformation to feature columns, returning a DataFrame with the same structure.

        Args:
            df: DataFrame with keys + date_col + feature columns.

        Returns:
            DataFrame with same structure, feature values transformed.
        """
