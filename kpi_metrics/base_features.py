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
    date_col: str,
    windows: list[tuple[str, int]],
) -> pl.DataFrame:
    """Compute group-wise rolling sums. Assumes df is sorted by (keys, date_col)."""
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
        windows: list[tuple[str, int]] = _DEFAULT_ROLLING_WINDOWS,
    ) -> None:
        self.params = params
        self.windows = windows

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        feature_cols = [
            f"{metric}__ROLLING_SUM__{name.upper()}"
            for metric, (name, _) in iproduct(self.params.metrics, self.windows)
        ]
        result = _rolling_sum(
            df.sort([*self.params.keys, self.params.date_col]),
            metrics=self.params.metrics,
            keys=self.params.keys,
            date_col=self.params.date_col,
            windows=self.windows,
        )
        return result.select([*self.params.keys, self.params.date_col, *feature_cols])
