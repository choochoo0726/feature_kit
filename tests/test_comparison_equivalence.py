"""Pin the vectorised RankComparison / ResidualComparison to the original implementations.

The reference functions below are copies of the pre-vectorisation code
(Python callback for rank, NumPy least-squares loop for residual).
"""
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from feature_kit.comparisons import RankComparison, ResidualComparison
from feature_kit.params import Param

FEAT = "SALES__YOY_GROWTH__Q"
BENCH = "SALES__YOY_GROWTH__Q__LAG1WEEK"


def reference_rank(joined: pl.DataFrame, W: int) -> np.ndarray:
    diff = (pl.col(FEAT) - pl.col(BENCH)).alias("d")
    return (
        joined.with_columns(diff)
        .with_columns(
            r=pl.col("d").rolling_map(
                function=lambda s: (s.rank(method="average") / len(s) * 2 - 1).tail(1),
                window_size=W,
                min_samples=W,
            ).over("specid")
        )["r"].to_numpy()
    )


def reference_residual(joined: pl.DataFrame, W: int) -> np.ndarray:
    residuals = []
    for _, g in joined.group_by(["specid"], maintain_order=True):
        y = g[FEAT].to_numpy().astype(float)
        x = g[BENCH].to_numpy().astype(float)
        n = len(y)
        resid = np.full(n, np.nan)
        for i in range(W - 1, n):
            y_win, x_win = y[i - W + 1: i + 1], x[i - W + 1: i + 1]
            if np.any(np.isnan(y_win)) or np.any(np.isnan(x_win)):
                continue
            X = np.column_stack([x_win, np.ones(W)])
            beta, *_ = np.linalg.lstsq(X, y_win, rcond=None)
            resid[i] = y[i] - (x[i] * beta[0] + beta[1])
        residuals.extend(resid.tolist())
    return np.array(residuals)


def make_panel(n_ent: int, n_days: int, seed: int, *, ties=False, level=0.0, null_frac=0.03):
    rng = np.random.default_rng(seed)
    rows = n_ent * n_days
    x = rng.normal(size=rows) + level
    y = 0.5 * (x - level) + rng.normal(size=rows)
    if ties:
        x, y = np.round(x), np.round(y)
    x[np.arange(0, rows, 53)] = level + 2.0            # repeated values -> ties / constant stretches
    x_s = pl.Series(x).scatter(rng.choice(rows, int(rows * null_frac), replace=False), None)
    y_s = pl.Series(y).scatter(rng.choice(rows, int(rows * null_frac), replace=False), None)
    start = date(2023, 1, 1)
    return pl.DataFrame({
        "specid": np.repeat([f"E{i:03d}" for i in range(n_ent)], n_days),
        "DATADATE": [start + timedelta(days=d) for d in range(n_days)] * n_ent,
        FEAT: y_s,
        BENCH: x_s,
    })


def split(panel: pl.DataFrame):
    return panel.select("specid", "DATADATE", FEAT), panel.select("specid", "DATADATE", BENCH)


def assert_same(got: np.ndarray, want: np.ndarray, rtol: float, atol: float = 1e-9):
    assert np.array_equal(np.isnan(got), np.isnan(want)), "null pattern differs"
    ok = ~np.isnan(want)
    np.testing.assert_allclose(got[ok], want[ok], rtol=rtol, atol=atol)


PARAMS = Param(keys=["specid"], metrics=["SALES"])


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("W", [5, 20])
def test_rank_matches_reference(ties, W):
    panel = make_panel(6, 120, seed=1, ties=ties)
    base, bench = split(panel)
    got = RankComparison(PARAMS, W).compute(base, bench)[f"{FEAT}__VS__{BENCH}__RANK"].to_numpy()
    assert_same(got, reference_rank(panel, W), rtol=0, atol=0)


@pytest.mark.parametrize("level", [0.0, 1e6])
@pytest.mark.parametrize("W", [5, 20])
def test_residual_matches_reference(level, W):
    panel = make_panel(6, 150, seed=2, level=level)
    base, bench = split(panel)
    got = ResidualComparison(PARAMS, W).compute(base, bench)[f"{FEAT}__VS__{BENCH}__RESIDUAL"].to_numpy()
    assert_same(got, reference_residual(panel, W), rtol=1e-6)


def test_residual_constant_benchmark_window():
    panel = pl.DataFrame({
        "specid": ["A"] * 6,
        "DATADATE": [date(2023, 1, i + 1) for i in range(6)],
        FEAT: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        BENCH: [2.0] * 6,
    })
    base, bench = split(panel)
    got = ResidualComparison(PARAMS, 3).compute(base, bench)[f"{FEAT}__VS__{BENCH}__RESIDUAL"].to_list()
    assert got[:2] == [None, None]
    assert got[2:] == pytest.approx([1.0, 1.0, 1.0, 1.0])   # y_t minus window mean


def test_residual_faster_than_reference():
    # null-free data: the reference loop skips any window containing a null,
    # which would make it look artificially fast
    panel = make_panel(50, 1000, seed=3, null_frac=0.0)
    base, bench = split(panel)
    ResidualComparison(PARAMS, 63).compute(base.head(100), bench.head(100))  # warm-up
    t0 = time.perf_counter()
    ResidualComparison(PARAMS, 63).compute(base, bench)
    new = time.perf_counter() - t0
    t0 = time.perf_counter()
    reference_residual(panel, 63)
    old = time.perf_counter() - t0
    assert new < old
