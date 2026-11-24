from __future__ import annotations
from itertools import product as iproduct

import polars as pl

from feature_kit.params import BaseFeature, Param

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
    """N-day rolling sum of each metric, grouped by entity keys.

    $$
    S^{(w)}_t = \\sum_{i=0}^{w-1} x_{t-i}
    $$

    The window counts **rows**, not calendar days, and uses
    ``min_samples=1``: the first ``w - 1`` rows of each entity hold partial
    sums rather than nulls.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    windows : list[tuple[str, int]], optional
        ``(name, size)`` pairs. ``name`` is upper-cased into the column name.
        Defaults to ``[("daily", 1), ("weekly", 7), ("quadweekly", 28),
        ("quarterly", 91)]``.

    Notes
    -----
    Output columns: ``{METRIC}__ROLLING_SUM__{NAME}``, e.g.
    ``SALES__ROLLING_SUM__WEEKLY``.

    Examples
    --------
    >>> base = RollingSumBaseFeature(params, windows=[("weekly", 7)])
    >>> base.compute(raw_df).columns  # doctest: +SKIP
    ['specid', 'DATADATE', 'SALES__ROLLING_SUM__WEEKLY']
    """

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
    """Year-over-year growth of an N-day rolling sum.

    $$
    g_t = \\frac{S^{(w)}_t - S^{(w)}_{t - \\text{yoy\\_days}}}
               {\\left|S^{(w)}_{t - \\text{yoy\\_days}}\\right|}
    $$

    The rolling sum ``S`` is row-based (see ``RollingSumBaseFeature``), but
    the prior-year value is matched by **exact calendar date**
    (``date - yoy_days``). If that date is missing for the entity, the growth
    is null. A zero prior-year sum gives ``inf``/``NaN``.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    windows : list[tuple[str, int]], optional
        ``(name, size)`` rolling-sum windows. Defaults to
        ``[("quadweekly", 28), ("quarterly", 91)]``.
    yoy_days : int, default 364
        Calendar-day offset of the comparison period. 364 (52 weeks) keeps
        the day-of-week aligned.

    Notes
    -----
    Output columns: ``{METRIC}__YOY_GROWTH__{NAME}``, e.g.
    ``SALES__YOY_GROWTH__QUADWEEKLY``.
    """

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


class POPGrowthBaseFeature(BaseFeature):
    """Period-over-period growth of an N-day rolling sum.

    $$
    g_t = \\frac{S^{(w)}_t - S^{(w)}_{t-w}}{\\left|S^{(w)}_{t-w}\\right|}
    $$

    The lag equals the window size, so consecutive non-overlapping periods
    are compared (e.g. the last 28 days vs. the 28 days before). The prior
    value is matched by exact calendar date (``date - w`` days); a missing
    prior date gives null.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    windows : list[tuple[str, int]], optional
        ``(name, size)`` windows. Defaults to
        ``[("quadweekly", 28), ("quarterly", 91)]``.

    Notes
    -----
    Output columns: ``{METRIC}__POP_GROWTH__{NAME}``, e.g.
    ``SALES__POP_GROWTH__QUARTERLY``.
    """

    def __init__(
        self,
        params: Param,
        windows: list[tuple[str, int]] | None = None,
    ) -> None:
        self.params = params
        self.windows = windows if windows is not None else _DEFAULT_POP_WINDOWS

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        metrics = self.params.metrics

        roll_df = _rolling_sum(
            df.sort([*keys, date_col]),
            metrics=metrics,
            keys=keys,
            windows=self.windows,
        )

        feature_exprs = []
        feature_cols = []

        for name, window_days in self.windows:
            cols_for_window = [f"{m}__ROLLING_SUM__{name.upper()}" for m in metrics]

            prior_df = (
                roll_df
                .select([*keys, date_col, *cols_for_window])
                .with_columns(
                    (pl.col(date_col) + pl.duration(days=window_days)).alias(date_col)
                )
                .rename({c: f"{c}__PRIOR" for c in cols_for_window})
            )

            roll_df = roll_df.join(prior_df, on=[*keys, date_col], how="left")

            for m in metrics:
                curr = f"{m}__ROLLING_SUM__{name.upper()}"
                prior = f"{curr}__PRIOR"
                feat = f"{m}__POP_GROWTH__{name.upper()}"
                feature_exprs.append(
                    ((pl.col(curr) - pl.col(prior)) / pl.col(prior).abs()).alias(feat)
                )
                feature_cols.append(feat)

        return (
            roll_df
            .with_columns(feature_exprs)
            .select([*keys, date_col, *feature_cols])
        )
