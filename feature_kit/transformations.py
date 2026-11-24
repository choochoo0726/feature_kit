from __future__ import annotations

import polars as pl

from feature_kit.params import Transformation


def _feature_cols(df: pl.DataFrame, non_feature: set[str]) -> list[str]:
    return [c for c in df.columns if c not in non_feature]


class ClipTransformation(Transformation):
    """Clip every feature column to the fixed range ``[lower, upper]``.

    Parameters
    ----------
    lower : float
        Lower bound.
    upper : float
        Upper bound.

    Notes
    -----
    Every column except ``specid`` and ``DATADATE`` is treated as a feature
    (these names are hardcoded; see *Known limitations* in the docs).
    """

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = lower
        self.upper = upper

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, {"specid", "DATADATE"})
        return df.with_columns([pl.col(c).clip(self.lower, self.upper) for c in feat_cols])


class WinsorizeTransformation(Transformation):
    """Clip each feature column to its ``[lower_q, upper_q]`` quantiles.

    Quantiles are computed once over **all rows of the input**, pooled
    across every entity and every date. In a backtest this uses future
    data: a value in 2022 is clipped with bounds that depend on 2024 data.
    For point-in-time signals, winsorize per date or on an expanding window
    downstream instead.

    Parameters
    ----------
    lower_q : float, default 0.01
        Lower quantile (linear interpolation).
    upper_q : float, default 0.99
        Upper quantile (linear interpolation).

    Notes
    -----
    Every column except ``specid`` and ``DATADATE`` is treated as a feature.
    """

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
    """Fade features in as an entity ages: multiply by a 0→1 ramp.

    $$
    s = \\operatorname{clip}\\left(\\frac{\\text{age} - \\text{min\\_age}}
        {\\text{max\\_age} - \\text{min\\_age}},\\ 0,\\ 1\\right),
    \\qquad f' = f \\cdot s
    $$

    Use it to down-weight signals from newly covered entities whose data is
    still noisy.

    Parameters
    ----------
    age_col : str
        Column in the input holding entity age (e.g. days since coverage
        start). It is left unscaled.
    min_age : int
        Age at or below which features are zeroed.
    max_age : int
        Age at or above which features are kept at full weight.

    Notes
    -----
    ``age_col`` must be present in the DataFrame passed to ``compute``.
    ``FeatureBuilder`` output keeps only keys, date and comparison columns,
    so call this transformation on its own after joining the age column,
    not inside the builder's ``transformations`` list.
    """

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
    """Fade features out as an entity ages: multiply by a 1→0 ramp.

    $$
    s = \\operatorname{clip}\\left(1 - \\frac{\\text{age} - \\text{min\\_age}}
        {\\text{max\\_age} - \\text{min\\_age}},\\ 0,\\ 1\\right),
    \\qquad f' = f \\cdot s
    $$

    The mirror of ``RampUpTransformation``, e.g. for signals that only
    matter shortly after an event.

    Parameters
    ----------
    age_col : str
        Column in the input holding entity age. It is left unscaled.
    min_age : int
        Age at or below which features are kept at full weight.
    max_age : int
        Age at or above which features are zeroed.

    Notes
    -----
    Same ``age_col`` requirement as ``RampUpTransformation``: apply it on
    its own, not inside ``FeatureBuilder``.
    """

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
