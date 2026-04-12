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
    windows: list[tuple[str, int]],
) -> pl.DataFrame:
    """Compute group-wise rolling sums. Assumes df is sorted by (keys, date_col).

    Window size is row-count-based, not calendar-day-based. For daily data with
    no gaps, these are equivalent. Callers are responsible for ensuring the input
    has no missing dates if calendar-day semantics are required.
    """
    exprs = [
        pl.col(metric)
        .rolling_sum(window_size=days, min_samples=1)
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
        windows: list[tuple[str, int]] | None = None,
    ) -> None:
        self.params = params
        self.windows = windows if windows is not None else _DEFAULT_ROLLING_WINDOWS

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = [
            f"{metric}__ROLLING_SUM__{name.upper()}"
            for metric, (name, _) in iproduct(self.params.metrics, self.windows)
        ]
        result = _rolling_sum(
            df.sort([*self.params.keys, self.params.date_col]),
            metrics=self.params.metrics,
            keys=self.params.keys,
            windows=self.windows,
        )
        return result.select([*self.params.keys, self.params.date_col, *feature_cols])


class YOYGrowthBaseFeature(BaseFeature):
    """YoY growth of N-day rolling sum: (sum_t - sum_{t-yoy_days}) / abs(sum_{t-yoy_days})."""

    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] | None = None,
        yoy_days: int = 364,
    ) -> None:
        self.params = params
        self.windows = windows if windows is not None else _DEFAULT_YOY_WINDOWS
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
