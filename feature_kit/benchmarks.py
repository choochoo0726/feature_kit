from __future__ import annotations

import polars as pl

from feature_kit.params import Benchmark, Param, _safe_div

_DEFAULT_PERIODS: list[tuple[str, int]] = [("week", 7), ("month", 28), ("quarter", 91)]


class ZeroBenchmark(Benchmark):
    """Constant ``0.0`` benchmark aligned to ``base_df``.

    Answers "is the feature positive?". Works with any base feature. Because
    the column has no metric prefix, comparisons pair it with every feature
    column.

    Parameters
    ----------
    params : Param
        Shared column configuration.

    Notes
    -----
    Output column: ``ZERO_BENCHMARK``.
    """

    def __init__(self, params: Param) -> None:
        self.params = params

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        return base_df.select([*keys, date_col]).with_columns(
            pl.lit(0.0).alias("ZERO_BENCHMARK")
        )


class LaggedYOYGrowthBenchmark(Benchmark):
    """YoY growth feature values lagged by N multiples of named periods.

    Answers "has YoY growth accelerated since a week / month / quarter ago?".
    For each ``(name, days)`` period and each ``n`` in ``1..num_lags``, the
    benchmark at date ``t`` is the feature's value at calendar date
    ``t - n * days`` (null if that date is missing).

    Requires a base feature with ``__YOY_GROWTH__`` columns
    (``YOYGrowthBaseFeature``); other columns are ignored.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    periods : list[tuple[str, int]], optional
        ``(name, days)`` pairs. Defaults to
        ``[("week", 7), ("month", 28), ("quarter", 91)]``.
    num_lags : int, default 1
        Number of multiples of each period to produce.

    Notes
    -----
    Output columns: ``{feature}__LAG{n}{NAME}``, e.g.
    ``SALES__YOY_GROWTH__QUADWEEKLY__LAG1WEEK``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        yoy_cols = [c for c in base_df.columns if "__YOY_GROWTH__" in c]

        result = base_df.select([*keys, date_col])
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df
                    .select([*keys, date_col, *yoy_cols])
                    .with_columns(
                        (pl.col(date_col) + pl.duration(days=lag)).alias(date_col)
                    )
                    .rename({c: f"{c}__{suffix}" for c in yoy_cols})
                )
                bench_cols = [f"{c}__{suffix}" for c in yoy_cols]
                result = result.join(
                    lag_df.select([*keys, date_col, *bench_cols]),
                    on=[*keys, date_col],
                    how="left",
                )
        return result


class LaggedPOPGrowthBenchmark(Benchmark):
    """PoP growth feature values lagged by N multiples of named periods.

    Same mechanics as ``LaggedYOYGrowthBenchmark`` but for
    ``__POP_GROWTH__`` columns (``POPGrowthBaseFeature``): the benchmark at
    date ``t`` is the feature's value at calendar date ``t - n * days``.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    periods : list[tuple[str, int]], optional
        ``(name, days)`` pairs. Defaults to
        ``[("week", 7), ("month", 28), ("quarter", 91)]``.
    num_lags : int, default 1
        Number of multiples of each period to produce.

    Notes
    -----
    Output columns: ``{feature}__LAG{n}{NAME}``, e.g.
    ``SALES__POP_GROWTH__QUADWEEKLY__LAG2MONTH``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        pop_cols = [c for c in base_df.columns if "__POP_GROWTH__" in c]

        result = base_df.select([*keys, date_col])
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df
                    .select([*keys, date_col, *pop_cols])
                    .with_columns(
                        (pl.col(date_col) + pl.duration(days=lag)).alias(date_col)
                    )
                    .rename({c: f"{c}__{suffix}" for c in pop_cols})
                )
                bench_cols = [f"{c}__{suffix}" for c in pop_cols]
                result = result.join(
                    lag_df.select([*keys, date_col, *bench_cols]),
                    on=[*keys, date_col],
                    how="left",
                )
        return result


class LaggedRollingSumBenchmark(Benchmark):
    """Rolling sum values lagged by N multiples of named periods.

    For ``__ROLLING_SUM__`` columns (``RollingSumBaseFeature``): the
    benchmark at date ``t`` is the rolling sum at calendar date
    ``t - n * days``. Pair with ``RatioComparison`` to get a growth rate
    versus that earlier period.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    periods : list[tuple[str, int]], optional
        ``(name, days)`` pairs. Defaults to
        ``[("week", 7), ("month", 28), ("quarter", 91)]``.
    num_lags : int, default 1
        Number of multiples of each period to produce.

    Notes
    -----
    Output columns: ``{feature}__LAG{n}{NAME}``, e.g.
    ``SALES__ROLLING_SUM__WEEKLY__LAG1MONTH``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        roll_cols = [c for c in base_df.columns if "__ROLLING_SUM__" in c]
        result = base_df.select([*keys, date_col])
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df.select([*keys, date_col, *roll_cols])
                    .with_columns((pl.col(date_col) + pl.duration(days=lag)).alias(date_col))
                    .rename({c: f"{c}__{suffix}" for c in roll_cols})
                )
                bench_cols = [f"{c}__{suffix}" for c in roll_cols]
                result = result.join(lag_df.select([*keys, date_col, *bench_cols]), on=[*keys, date_col], how="left")
        return result


class LaggedYOYProjectedRollingSumBenchmark(Benchmark):
    """Seasonally projected rolling sum.

    Projects today's rolling sum by applying the YoY growth observed ``L``
    days ago to last year's value:

    $$
    \\hat S_t = S_{t-Y} \\cdot \\frac{S_{t-L}}{S_{t-Y-L}}
    $$

    where ``Y = yoy_days`` and ``L = n * days`` for each period and each
    ``n`` in ``1..num_lags``. All three terms are matched by exact calendar
    date; any missing term, or a zero denominator, gives null. Answers "is the current period
    running ahead of what the recent YoY trend implies?".

    Requires ``__ROLLING_SUM__`` columns (``RollingSumBaseFeature``).

    Parameters
    ----------
    params : Param
        Shared column configuration.
    periods : list[tuple[str, int]], optional
        ``(name, days)`` pairs. Defaults to
        ``[("week", 7), ("month", 28), ("quarter", 91)]``.
    num_lags : int, default 1
        Number of multiples of each period to produce.
    yoy_days : int, default 364
        Calendar-day offset of the prior year.

    Notes
    -----
    Output columns: ``{feature}__LAG{n}{NAME}_YOY_PROJECTED``, e.g.
    ``SALES__ROLLING_SUM__WEEKLY__LAG1MONTH_YOY_PROJECTED``.
    """

    def __init__(
        self,
        params: Param,
        periods: list[tuple[str, int]] | None = None,
        num_lags: int = 1,
        yoy_days: int = 364,
    ) -> None:
        self.params = params
        self.periods = periods if periods is not None else _DEFAULT_PERIODS
        self.num_lags = num_lags
        self.yoy_days = yoy_days

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        roll_cols = [c for c in base_df.columns if "__ROLLING_SUM__" in c]
        result = base_df.select([*keys, date_col])
        yoy_df = (
            base_df.select([*keys, date_col, *roll_cols])
            .with_columns((pl.col(date_col) + pl.duration(days=self.yoy_days)).alias(date_col))
            .rename({c: f"{c}__YOY" for c in roll_cols})
        )
        for period_name, period_days in self.periods:
            for n in range(1, self.num_lags + 1):
                lag = n * period_days
                suffix = f"LAG{n}{period_name.upper()}"
                lag_df = (
                    base_df.select([*keys, date_col, *roll_cols])
                    .with_columns((pl.col(date_col) + pl.duration(days=lag)).alias(date_col))
                    .rename({c: f"{c}__{suffix}" for c in roll_cols})
                )
                yoy_lag_df = (
                    base_df.select([*keys, date_col, *roll_cols])
                    .with_columns((pl.col(date_col) + pl.duration(days=self.yoy_days + lag)).alias(date_col))
                    .rename({c: f"{c}__YOY{suffix}" for c in roll_cols})
                )
                joined = (
                    base_df.select([*keys, date_col])
                    .join(yoy_df, on=[*keys, date_col], how="left")
                    .join(lag_df, on=[*keys, date_col], how="left")
                    .join(yoy_lag_df, on=[*keys, date_col], how="left")
                )
                proj_exprs = [
                    _safe_div(pl.col(f"{c}__YOY") * pl.col(f"{c}__{suffix}"), pl.col(f"{c}__YOY{suffix}"))
                    .alias(f"{c}__{suffix}_YOY_PROJECTED")
                    for c in roll_cols
                ]
                proj_cols = [f"{c}__{suffix}_YOY_PROJECTED" for c in roll_cols]
                result = result.join(
                    joined.with_columns(proj_exprs).select([*keys, date_col, *proj_cols]),
                    on=[*keys, date_col],
                    how="left",
                )
        return result


class IBESConsensusSalesGrowthBenchmark(Benchmark):
    """External IBES consensus sales growth, as-of joined per entity.

    For each ``(keys, date)`` row in ``base_df``, takes the most recent IBES
    row for the same entity dated **on or before** that date (backward
    ``join_asof``), so no future consensus leaks in. Output rows are in the
    same order as ``base_df``.

    Parameters
    ----------
    params : Param
        Shared column configuration.
    freq : {"quarterly", "semiannual"}, default "quarterly"
        Which consensus column to read.
    metric : str, optional
        The metric the consensus describes, e.g. ``"SALES"``. The output
        column is prefixed with it, so comparisons pair it only with that
        metric's features. Defaults to the only metric in
        ``params.metrics``; required when there are several.

    Raises
    ------
    ValueError
        If ``freq`` is invalid, ``metric`` is missing or not in
        ``params.metrics``, or ``external_dfs["ibes"]`` is not provided to
        ``compute``.

    Notes
    -----
    ``external_dfs["ibes"]`` must contain ``params.keys``,
    ``params.date_col`` and the input column for ``freq``:

    | freq | Input column | Output column |
    |------|--------------|---------------|
    | ``"quarterly"`` | ``IBES_QUARTERLY_SALES_GROWTH_BENCHMARK`` | ``{metric}__IBES_QUARTERLY_SALES_GROWTH_BENCHMARK`` |
    | ``"semiannual"`` | ``IBES_SEMIANNUAL_SALES_GROWTH_BENCHMARK`` | ``{metric}__IBES_SEMIANNUAL_SALES_GROWTH_BENCHMARK`` |

    Use it with ``YOYGrowthBaseFeature`` so growth is compared with growth.
    """

    COLUMN_MAP = {
        "quarterly": "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK",
        "semiannual": "IBES_SEMIANNUAL_SALES_GROWTH_BENCHMARK",
    }

    def __init__(self, params: Param, freq: str = "quarterly", metric: str | None = None) -> None:
        if freq not in self.COLUMN_MAP:
            raise ValueError(f"freq must be 'quarterly' or 'semiannual', got {freq!r}")
        if metric is None:
            if len(params.metrics) != 1:
                raise ValueError(
                    "IBESConsensusSalesGrowthBenchmark needs metric=... when params has "
                    f"several metrics {params.metrics}"
                )
            metric = params.metrics[0]
        elif metric not in params.metrics:
            raise ValueError(f"metric {metric!r} is not in params.metrics {params.metrics}")
        self.params = params
        self.freq = freq
        self.metric = metric

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if external_dfs is None or "ibes" not in external_dfs:
            raise ValueError("IBESConsensusSalesGrowthBenchmark requires external_dfs['ibes']")
        keys = self.params.keys
        date_col = self.params.date_col
        in_col = self.COLUMN_MAP[self.freq]
        out_col = f"{self.metric}__{in_col}"
        order = "__row_order__"
        # join_asof needs both sides sorted by date within each entity; sorting by
        # (keys, date) satisfies that, so the sortedness check can be skipped.
        right = (
            external_dfs["ibes"].select([*keys, date_col, in_col])
            .sort([*keys, date_col])
            .rename({in_col: out_col})
        )
        return (
            base_df.select([*keys, date_col]).with_row_index(order)
            .sort([*keys, date_col])
            .join_asof(right, on=date_col, by=keys, strategy="backward", check_sortedness=False)
            .sort(order)
            .select([*keys, date_col, out_col])
        )


class CohortRollingSumBenchmark(Benchmark):
    """Cross-sectional mean or median of rolling sums within a cohort, per date.

    Answers "is this entity doing better than its peers today?". For each
    date, groups entities by ``cohort_cols`` (read from ``raw_df``) and
    aggregates each ``__ROLLING_SUM__`` column. By default the entity itself
    is included in its own cohort's aggregate; set ``exclude_self=True`` for
    a leave-one-out benchmark (peers only).

    Parameters
    ----------
    params : Param
        Shared column configuration.
    cohort_cols : list[str]
        Columns in ``raw_df`` defining the peer group, e.g. ``["SECTOR"]``.
    agg : {"mean", "median"}, default "mean"
        Aggregation across the cohort. Null values are ignored.
    exclude_self : bool, default False
        Leave the entity's own value out of its cohort's aggregate. A cohort
        whose only non-null value is the entity's own gives null.

    Raises
    ------
    ValueError
        If ``agg`` is not ``"mean"`` or ``"median"``.

    Notes
    -----
    Output columns: ``{feature}__COHORT_BENCHMARK``, e.g.
    ``SALES__ROLLING_SUM__WEEKLY__COHORT_BENCHMARK``.
    """

    def __init__(
        self,
        params: Param,
        cohort_cols: list[str],
        agg: str = "mean",
        exclude_self: bool = False,
    ) -> None:
        if agg not in ("mean", "median"):
            raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
        self.params = params
        self.cohort_cols = cohort_cols
        self.agg = agg
        self.exclude_self = exclude_self

    def compute(
        self,
        base_df: pl.DataFrame,
        raw_df: pl.DataFrame,
        external_dfs: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        roll_cols = [c for c in base_df.columns if "__ROLLING_SUM__" in c]
        enriched = base_df.join(
            raw_df.select([*keys, date_col, *self.cohort_cols]).unique([*keys, date_col]),
            on=[*keys, date_col],
            how="left",
        )
        group = [*self.cohort_cols, date_col]
        bench_cols = [f"{c}__COHORT_BENCHMARK" for c in roll_cols]

        if not self.exclude_self:
            agg_fn = pl.mean if self.agg == "mean" else pl.median
            cohort_agg = enriched.group_by(group).agg(
                [agg_fn(c).alias(f"{c}__COHORT_BENCHMARK") for c in roll_cols]
            )
            return (
                enriched.join(cohort_agg, on=group, how="left")
                .select([*keys, date_col, *bench_cols])
            )

        exprs = [
            (_loo_mean(c, group) if self.agg == "mean" else _loo_median(c, group))
            .alias(f"{c}__COHORT_BENCHMARK")
            for c in roll_cols
        ]
        return enriched.with_columns(exprs).select([*keys, date_col, *bench_cols])


def _loo_mean(col: str, group: list[str]) -> pl.Expr:
    """Mean of the group's non-null values, leaving out this row's own value."""
    x = pl.col(col)
    total = x.sum().over(group)
    n = x.count().over(group)  # non-null count
    own_n = x.is_not_null().cast(pl.Int64)
    rest_n = n - own_n
    rest_sum = total - x.fill_null(0.0)
    return pl.when(rest_n > 0).then(rest_sum / rest_n).otherwise(None)


def _loo_median(col: str, group: list[str]) -> pl.Expr:
    """Median of the group's non-null values, leaving out this row's own value.

    Uses the group's sorted values ``s`` and this row's position ``p`` in that
    order (ordinal rank, so ties get distinct positions). Removing index ``p``
    gives ``r[k] = s[k]`` for ``k < p`` and ``s[k + 1]`` otherwise; the median
    of ``r`` (length ``m``) is the mean of ``r[(m - 1) // 2]`` and ``r[m // 2]``.
    A row with a null value is not in ``s``, so it gets the full-group median.
    """
    x = pl.col(col)
    s = x.drop_nulls().sort().implode().over(group)
    p = (x.rank("ordinal").over(group) - 1).cast(pl.Int64)
    m = x.count().over(group).cast(pl.Int64) - 1

    def r(k: pl.Expr) -> pl.Expr:
        return s.list.get(pl.when(k < p).then(k).otherwise(k + 1), null_on_oob=True)

    loo = (r((m - 1) // 2) + r(m // 2)) / 2
    return (
        pl.when(x.is_null()).then(s.list.median())
        .when(m > 0).then(loo)
        .otherwise(None)
    )
