from __future__ import annotations

from typing import Literal

import polars as pl

from feature_kit.params import Param, Transformation

_DEFAULT_PARAMS = Param(keys=["specid"], metrics=[], date_col="DATADATE")


def _feature_cols(
    df: pl.DataFrame,
    params: Param | None,
    extra: list[str] | None = None,
) -> list[str]:
    """All columns except keys, date and *extra* (e.g. an age column)."""
    p = params or _DEFAULT_PARAMS
    non_feature = {*p.keys, p.date_col, *(extra or [])}
    return [c for c in df.columns if c not in non_feature]


class ClipTransformation(Transformation):
    """Clip every feature column to the fixed range ``[lower, upper]``.

    Point-in-time safe: each value is clipped using only itself.

    Parameters
    ----------
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    params : Param, optional
        Identifies the key and date columns to leave untouched. Defaults to
        ``specid`` / ``DATADATE``; ``FeatureBuilder`` fills it in
        automatically.
    """

    def __init__(self, lower: float, upper: float, params: Param | None = None) -> None:
        self.lower = lower
        self.upper = upper
        self.params = params

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, self.params)
        return df.with_columns([pl.col(c).clip(self.lower, self.upper) for c in feat_cols])


class WinsorizeTransformation(Transformation):
    """Clip each feature column to its ``[lower_q, upper_q]`` quantiles.

    With the default ``scope="date"``, quantiles are computed **per date,
    across entities**. Each row is clipped using only data from its own
    date, so the result is point-in-time safe for backtests.

    ``scope="pooled"`` computes one set of quantiles over all rows of the
    input, across every date. This uses future data (a 2022 value is
    clipped with bounds that depend on 2024 data), so use it only for
    exploratory, non-backtest work.

    Parameters
    ----------
    lower_q : float, default 0.01
        Lower quantile (linear interpolation).
    upper_q : float, default 0.99
        Upper quantile (linear interpolation).
    scope : {"date", "pooled"}, default "date"
        Which rows the quantiles are computed over.
    params : Param, optional
        Identifies the key and date columns. Defaults to ``specid`` /
        ``DATADATE``; ``FeatureBuilder`` fills it in automatically.

    Notes
    -----
    Per-date quantiles need enough entities per date to be meaningful. With
    only a handful of entities, the 1%/99% quantiles are close to the
    min/max, so little is clipped.
    """

    def __init__(
        self,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
        scope: Literal["date", "pooled"] = "date",
        params: Param | None = None,
    ) -> None:
        if scope not in ("date", "pooled"):
            raise ValueError(f"scope must be 'date' or 'pooled', got {scope!r}")
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.scope = scope
        self.params = params

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, self.params)
        if self.scope == "pooled":
            exprs = []
            for c in feat_cols:
                lo = df[c].quantile(self.lower_q, interpolation="linear")
                hi = df[c].quantile(self.upper_q, interpolation="linear")
                exprs.append(pl.col(c).clip(lo, hi))
            return df.with_columns(exprs)

        date_col = (self.params or _DEFAULT_PARAMS).date_col
        return df.with_columns(
            pl.col(c).clip(
                pl.col(c).quantile(self.lower_q, interpolation="linear").over(date_col),
                pl.col(c).quantile(self.upper_q, interpolation="linear").over(date_col),
            )
            for c in feat_cols
        )


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
        Column holding entity age (e.g. days since coverage start). It is
        left unscaled.
    min_age : int
        Age at or below which features are zeroed.
    max_age : int
        Age at or above which features are kept at full weight.
    params : Param, optional
        Identifies the key and date columns. Defaults to ``specid`` /
        ``DATADATE``; ``FeatureBuilder`` fills it in automatically.

    Notes
    -----
    Standalone, ``age_col`` must be in the DataFrame passed to ``compute``.
    Inside ``FeatureBuilder`` it is read from the raw input automatically
    (via ``required_columns``) and dropped from the output.
    """

    def __init__(
        self, age_col: str, min_age: int, max_age: int, params: Param | None = None
    ) -> None:
        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age
        self.params = params

    @property
    def required_columns(self) -> list[str]:
        return [self.age_col]

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, self.params, [self.age_col])
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
        Column holding entity age. It is left unscaled.
    min_age : int
        Age at or below which features are kept at full weight.
    max_age : int
        Age at or above which features are zeroed.
    params : Param, optional
        Identifies the key and date columns. Defaults to ``specid`` /
        ``DATADATE``; ``FeatureBuilder`` fills it in automatically.

    Notes
    -----
    Same ``age_col`` handling as ``RampUpTransformation``.
    """

    def __init__(
        self, age_col: str, min_age: int, max_age: int, params: Param | None = None
    ) -> None:
        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age
        self.params = params

    @property
    def required_columns(self) -> list[str]:
        return [self.age_col]

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feat_cols = _feature_cols(df, self.params, [self.age_col])
        scale = (
            1.0 - ((pl.col(self.age_col) - self.min_age) / (self.max_age - self.min_age))
        ).clip(0.0, 1.0)
        return df.with_columns([(pl.col(c) * scale).alias(c) for c in feat_cols])
