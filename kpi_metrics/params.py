from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
