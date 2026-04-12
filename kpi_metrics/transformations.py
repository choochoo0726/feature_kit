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
        feat_cols = _feature_cols(df, {"specid", "DATADATE"})
        return df.with_columns([pl.col(c).clip(self.lower, self.upper) for c in feat_cols])


class WinsorizeTransformation(Transformation):
    """Clip feature columns to [lower_q, upper_q] quantiles computed across all rows."""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, {"specid", "DATADATE"})
        exprs = []
        for c in feat_cols:
            lo = df[c].quantile(self.lower_q, interpolation="linear")
            hi = df[c].quantile(self.upper_q, interpolation="linear")
            exprs.append(pl.col(c).clip(lo, hi))
        return df.with_columns(exprs)


class RampUpTransformation(Transformation):
    """Scale feature columns 0→1 based on entity age: (age - min_age) / (max_age - min_age), clipped [0,1]."""

    def __init__(self, age_col: str, min_age: int, max_age: int) -> None:
        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, {"specid", "DATADATE", self.age_col})
        scale = (
            ((pl.col(self.age_col) - self.min_age) / (self.max_age - self.min_age))
            .clip(0.0, 1.0)
        )
        return df.with_columns([(pl.col(c) * scale).alias(c) for c in feat_cols])


class RampDownTransformation(Transformation):
    """Scale feature columns 1→0 based on entity age: 1 - (age - min_age) / (max_age - min_age), clipped [0,1]."""

    def __init__(self, age_col: str, min_age: int, max_age: int) -> None:
        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, {"specid", "DATADATE", self.age_col})
        scale = (
            1.0 - ((pl.col(self.age_col) - self.min_age) / (self.max_age - self.min_age))
        ).clip(0.0, 1.0)
        return df.with_columns([(pl.col(c) * scale).alias(c) for c in feat_cols])
