from __future__ import annotations

import numpy as np
import polars as pl

from feature_kit.params import Comparison, Param


def _shared_and_feature_cols(
    base_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Return (shared_cols, feature_cols, bench_cols) from the two DataFrames."""
    shared = [c for c in base_df.columns if c in benchmark_df.columns]
    feature_cols = [c for c in base_df.columns if c not in shared]
    bench_cols = [c for c in benchmark_df.columns if c not in shared]
    return shared, feature_cols, bench_cols


def _pair_cols(
    feature_cols: list[str],
    bench_cols: list[str],
    metrics: list[str] | None,
) -> list[tuple[str, str]]:
    """Return valid (feature_col, bench_col) pairs.

    Rules, applied to each benchmark column in order:

    1. A benchmark column derived from a specific feature column (its name
       starts with ``{feature_col}__``, e.g. ``SALES__YOY_GROWTH__QUARTERLY__LAG1WEEK``)
       is paired **only** with that feature. This prevents cross-window and
       cross-metric pairs.
    2. Otherwise, when *metrics* is provided, a benchmark column with a known
       metric prefix is paired only with feature columns of that same metric.
    3. Otherwise (e.g. ``ZERO_BENCHMARK``, IBES columns) the benchmark column
       is paired with every feature column.
    """
    # The feature each benchmark column was derived from, if any (longest match wins).
    source = {
        bc: max((fc for fc in feature_cols if bc.startswith(fc + "__")), key=len, default=None)
        for bc in bench_cols
    }
    pairs = []
    for fc in feature_cols:
        for bc in bench_cols:
            if source[bc] is not None:
                if source[bc] != fc:
                    continue
            elif metrics:
                fc_metric = next((m for m in metrics if fc.startswith(m + "__")), None)
                bc_metric = next((m for m in metrics if bc.startswith(m + "__")), None)
                if fc_metric and bc_metric and fc_metric != bc_metric:
                    continue
            pairs.append((fc, bc))
    return pairs


class DifferenceComparison(Comparison):
    """Difference ``base - benchmark`` for each matching column pair.

    Parameters
    ----------
    params : Param, optional
        Used only for pairing. A benchmark column derived from a feature
        (``{feature}__...``) is always paired only with that feature. If
        ``params`` is given, other metric-prefixed benchmark columns are
        paired only with features of the same metric; if omitted, they are
        paired with every feature. Unprefixed columns (``ZERO_BENCHMARK``)
        pair with every feature either way.

    Notes
    -----
    Output columns: ``{base_col}__VS__{bench_col}__DIFFERENCE``.
    """

    def __init__(self, params: Param | None = None) -> None:
        self.params = params

    def compute(self, base_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> pl.DataFrame:
        shared, feature_cols, bench_cols = _shared_and_feature_cols(base_df, benchmark_df)
        metrics = self.params.metrics if self.params else None
        joined = base_df.join(benchmark_df, on=shared, how="left")
        exprs, result_cols = [], []
        for fc, bc in _pair_cols(feature_cols, bench_cols, metrics):
            name = f"{fc}__VS__{bc}__DIFFERENCE"
            exprs.append((pl.col(fc) - pl.col(bc)).alias(name))
            result_cols.append(name)
        return joined.with_columns(exprs).select([*shared, *result_cols])


class SignComparison(Comparison):
    """Sign of ``base - benchmark``: ``-1.0``, ``0.0`` or ``1.0`` (Float64).

    Parameters
    ----------
    params : Param, optional
        Used only for same-metric pairing; see ``DifferenceComparison``.

    Notes
    -----
    Output columns: ``{base_col}__VS__{bench_col}__SIGN``.
    """

    def __init__(self, params: Param | None = None) -> None:
        self.params = params

    def compute(self, base_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> pl.DataFrame:
        shared, feature_cols, bench_cols = _shared_and_feature_cols(base_df, benchmark_df)
        metrics = self.params.metrics if self.params else None
        joined = base_df.join(benchmark_df, on=shared, how="left")
        exprs, result_cols = [], []
        for fc, bc in _pair_cols(feature_cols, bench_cols, metrics):
            name = f"{fc}__VS__{bc}__SIGN"
            exprs.append((pl.col(fc) - pl.col(bc)).sign().cast(pl.Float64).alias(name))
            result_cols.append(name)
        return joined.with_columns(exprs).select([*shared, *result_cols])


class RatioComparison(Comparison):
    """Relative difference ``(base - benchmark) / benchmark``.

    Divides by the signed benchmark, not its absolute value, so the sign
    flips when the benchmark is negative. A zero benchmark gives
    ``inf``/``NaN``, so don't use it with ``ZeroBenchmark``.

    Parameters
    ----------
    params : Param, optional
        Used only for same-metric pairing; see ``DifferenceComparison``.

    Notes
    -----
    Output columns: ``{base_col}__VS__{bench_col}__RATIO``.
    """

    def __init__(self, params: Param | None = None) -> None:
        self.params = params

    def compute(self, base_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> pl.DataFrame:
        shared, feature_cols, bench_cols = _shared_and_feature_cols(base_df, benchmark_df)
        metrics = self.params.metrics if self.params else None
        joined = base_df.join(benchmark_df, on=shared, how="left")
        exprs, result_cols = [], []
        for fc, bc in _pair_cols(feature_cols, bench_cols, metrics):
            name = f"{fc}__VS__{bc}__RATIO"
            exprs.append(((pl.col(fc) - pl.col(bc)) / pl.col(bc)).alias(name))
            result_cols.append(name)
        return joined.with_columns(exprs).select([*shared, *result_cols])


class ZScoreComparison(Comparison):
    """Rolling time-series z-score of ``d = base - benchmark``, per entity.

    $$
    z_t = \\frac{d_t - \\operatorname{mean}(d_{t-W+1..t})}
               {\\operatorname{std}(d_{t-W+1..t})}
    $$

    The window is the last ``W`` rows of the entity **including** row
    ``t``. The first ``W - 1`` rows, and any window containing a null, give
    null.

    Parameters
    ----------
    params : Param
        Shared column configuration (required: provides ``keys`` for
        grouping and ``metrics`` for pairing).
    lookback_window : int
        ``W``, the number of rows in the rolling window.

    Notes
    -----
    Output columns: ``{base_col}__VS__{bench_col}__ZSCORE``.
    """

    def __init__(self, params: Param, lookback_window: int) -> None:
        self.params = params
        self.lookback_window = lookback_window

    def compute(self, base_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        shared, feature_cols, bench_cols = _shared_and_feature_cols(base_df, benchmark_df)
        joined = base_df.join(benchmark_df, on=shared, how="left").sort([*keys, date_col])

        diff_exprs, diff_cols = [], []
        for fc, bc in _pair_cols(feature_cols, bench_cols, self.params.metrics):
            diff_col = f"__diff__{fc}__{bc}"
            diff_exprs.append((pl.col(fc) - pl.col(bc)).alias(diff_col))
            diff_cols.append((diff_col, fc, bc))

        joined = joined.with_columns(diff_exprs)
        W = self.lookback_window
        zscore_exprs, result_cols = [], []
        for diff_col, fc, bc in diff_cols:
            name = f"{fc}__VS__{bc}__ZSCORE"
            mean_expr = pl.col(diff_col).rolling_mean(window_size=W, min_samples=W).over(keys)
            std_expr = pl.col(diff_col).rolling_std(window_size=W, min_samples=W).over(keys)
            zscore_exprs.append(((pl.col(diff_col) - mean_expr) / std_expr).alias(name))
            result_cols.append(name)

        return joined.with_columns(zscore_exprs).select([*shared, *result_cols])


class RankComparison(Comparison):
    """Rolling time-series percentile rank of ``d = base - benchmark``, per entity.

    Ranks today's ``d_t`` among the entity's own last ``W`` values
    (including today) and rescales it:

    $$
    r_t = 2 \\cdot \\frac{\\operatorname{rank}(d_t)}{W} - 1
    $$

    Ties use the average rank. The result lies in ``[2/W - 1, 1]``: ``1``
    means today is the highest value in the window. This is a time-series
    rank within each entity, **not** a cross-sectional rank across
    entities. The first ``W - 1`` rows give null.

    Parameters
    ----------
    params : Param
        Shared column configuration (required).
    lookback_window : int
        ``W``, the number of rows in the rolling window.

    Notes
    -----
    Output columns: ``{base_col}__VS__{bench_col}__RANK``. Uses
    ``rolling_map`` with a Python callback, so it is much slower than the
    other comparisons on large panels.
    """

    def __init__(self, params: Param, lookback_window: int) -> None:
        self.params = params
        self.lookback_window = lookback_window

    def compute(self, base_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        shared, feature_cols, bench_cols = _shared_and_feature_cols(base_df, benchmark_df)
        joined = base_df.join(benchmark_df, on=shared, how="left").sort([*keys, date_col])

        diff_exprs, diff_cols = [], []
        for fc, bc in _pair_cols(feature_cols, bench_cols, self.params.metrics):
            diff_col = f"__diff__{fc}__{bc}"
            diff_exprs.append((pl.col(fc) - pl.col(bc)).alias(diff_col))
            diff_cols.append((diff_col, fc, bc))

        joined = joined.with_columns(diff_exprs)
        W = self.lookback_window
        rank_exprs, result_cols = [], []
        for diff_col, fc, bc in diff_cols:
            name = f"{fc}__VS__{bc}__RANK"
            # rolling percentile rank scaled to [-1, 1]
            rank_expr = pl.col(diff_col).rolling_map(
                function=lambda s: (s.rank(method="average") / len(s) * 2 - 1).tail(1),
                window_size=W,
                min_samples=W,
            ).over(keys)
            rank_exprs.append(rank_expr.alias(name))
            result_cols.append(name)

        return joined.with_columns(rank_exprs).select([*shared, *result_cols])


class ResidualComparison(Comparison):
    """Rolling OLS residual of ``base ~ benchmark + const``, per entity.

    For each row ``t``, fits ``y = beta * x + alpha`` by least squares on
    the entity's last ``W`` rows (including ``t``) and returns

    $$
    e_t = y_t - (\\hat\\beta\\, x_t + \\hat\\alpha)
    $$

    where ``y`` is the base feature and ``x`` the benchmark. This is the
    part of the feature not explained by its recent linear relationship to
    the benchmark. Windows containing a null, and the first ``W - 1`` rows,
    give null.

    Parameters
    ----------
    params : Param
        Shared column configuration (required).
    lookback_window : int
        ``W``, the number of rows in the regression window.

    Notes
    -----
    Output columns: ``{base_col}__VS__{bench_col}__RESIDUAL``. Implemented
    as a Python/NumPy loop over entities and rows, so cost is roughly
    ``O(rows x W x pairs)``. It is the slowest component; pair it with
    a short list of features.

    A constant benchmark (e.g. ``ZeroBenchmark``) makes the regression
    degenerate: the residual reduces to ``y_t`` minus its window mean.
    """

    def __init__(self, params: Param, lookback_window: int) -> None:
        self.params = params
        self.lookback_window = lookback_window

    def compute(self, base_df: pl.DataFrame, benchmark_df: pl.DataFrame) -> pl.DataFrame:
        keys = self.params.keys
        date_col = self.params.date_col
        shared, feature_cols, bench_cols = _shared_and_feature_cols(base_df, benchmark_df)
        joined = base_df.join(benchmark_df, on=shared, how="left").sort([*keys, date_col])

        W = self.lookback_window
        result_cols = []

        for fc, bc in _pair_cols(feature_cols, bench_cols, self.params.metrics):
            out_col = f"{fc}__VS__{bc}__RESIDUAL"
            residuals = []

            for _, group_df in joined.group_by(keys, maintain_order=True):
                y = group_df[fc].to_numpy().astype(float)
                x = group_df[bc].to_numpy().astype(float)
                n = len(y)
                resid = np.full(n, np.nan)
                for i in range(W - 1, n):
                    y_win = y[i - W + 1: i + 1]
                    x_win = x[i - W + 1: i + 1]
                    if np.any(np.isnan(y_win)) or np.any(np.isnan(x_win)):
                        continue
                    X = np.column_stack([x_win, np.ones(W)])
                    try:
                        beta, *_ = np.linalg.lstsq(X, y_win, rcond=None)
                        resid[i] = y[i] - (x[i] * beta[0] + beta[1])
                    except np.linalg.LinAlgError:
                        pass
                residuals.extend(resid.tolist())

            joined = joined.with_columns(
                pl.Series(out_col, [None if np.isnan(v) else v for v in residuals])
            )
            result_cols.append(out_col)

        return joined.select([*shared, *result_cols])
