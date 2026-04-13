from __future__ import annotations

import numpy as np
import polars as pl

from kpi_metrics.params import Comparison, Param


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

    When *metrics* is provided, a benchmark column whose name starts with a
    known metric prefix is only paired with feature columns from that same
    metric.  Benchmark columns with no metric prefix (e.g. ``ZERO_BENCHMARK``)
    are paired with every feature column as before.
    """
    pairs = []
    for fc in feature_cols:
        for bc in bench_cols:
            if metrics:
                fc_metric = next((m for m in metrics if fc.startswith(m + "__")), None)
                bc_metric = next((m for m in metrics if bc.startswith(m + "__")), None)
                if fc_metric and bc_metric and fc_metric != bc_metric:
                    continue
            pairs.append((fc, bc))
    return pairs


class DifferenceComparison(Comparison):
    """base - benchmark for each same-metric (base_col, bench_col) pair."""

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
    """sign(base - benchmark) for each same-metric (base_col, bench_col) pair."""

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
    """(base - benchmark) / benchmark for each same-metric (base_col, bench_col) pair."""

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
    """Rolling z-score of (base - benchmark), grouped by params.keys."""

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
    """Cross-sectional [-1, 1] percentile rank of (base - benchmark), grouped by params.keys."""

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
    """OLS residual of (base ~ benchmark + const) over a rolling window, grouped by params.keys."""

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
