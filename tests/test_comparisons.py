import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.comparisons import (
    DifferenceComparison,
    SignComparison,
    RatioComparison,
    ZScoreComparison,
    RankComparison,
    ResidualComparison,
)


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def aligned_dfs():
    base_df = pl.DataFrame({
        "specid": ["A", "A", "A"],
        "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
        "SALES__YOY_GROWTH__DAILY": [0.1, 0.2, -0.1],
    })
    bench_df = pl.DataFrame({
        "specid": ["A", "A", "A"],
        "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
        "ZERO_BENCHMARK": [0.0, 0.0, 0.0],
    })
    return base_df, bench_df


@pytest.fixture
def rolling_dfs():
    base_df = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES__YOY_GROWTH__DAILY": [float(i) for i in range(10)],
    })
    bench_df = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "ZERO_BENCHMARK": [0.0] * 10,
    })
    return base_df, bench_df


class TestDifferenceComparison:
    def test_difference_values(self, aligned_dfs):
        base_df, bench_df = aligned_dfs
        result = DifferenceComparison().compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__DIFFERENCE"
        assert col in result.columns
        assert result[col].to_list() == pytest.approx([0.1, 0.2, -0.1])

    def test_no_raw_metrics_in_output(self, aligned_dfs):
        base_df, bench_df = aligned_dfs
        result = DifferenceComparison().compute(base_df, bench_df)
        assert "SALES__YOY_GROWTH__DAILY" not in result.columns
        assert "specid" in result.columns
        assert "DATADATE" in result.columns


class TestSignComparison:
    def test_sign_values(self, aligned_dfs):
        base_df, bench_df = aligned_dfs
        result = SignComparison().compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__SIGN"
        assert result[col].to_list() == pytest.approx([1.0, 1.0, -1.0])


class TestRatioComparison:
    def test_ratio_values(self):
        base_df = pl.DataFrame({
            "specid": ["A", "A"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 2)],
            "SALES__YOY_GROWTH__DAILY": [0.3, 0.6],
        })
        bench_df = pl.DataFrame({
            "specid": ["A", "A"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 2)],
            "LAGGED_BENCHMARK": [0.1, 0.2],
        })
        result = RatioComparison().compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__LAGGED_BENCHMARK__RATIO"
        assert result[col].to_list() == pytest.approx([2.0, 2.0])


class TestZScoreComparison:
    def test_output_columns(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = ZScoreComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__ZSCORE"
        assert col in result.columns

    def test_nulls_before_window(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = ZScoreComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__ZSCORE"
        vals = result[col].to_list()
        assert all(v is None for v in vals[:4])

    def test_non_null_after_window(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = ZScoreComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__ZSCORE"
        assert result[col][-1] is not None


class TestRankComparison:
    def test_output_columns(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = RankComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RANK"
        assert col in result.columns

    def test_rank_range(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = RankComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RANK"
        vals = [v for v in result[col].to_list() if v is not None]
        assert all(-1.0 <= v <= 1.0 for v in vals)


class TestResidualComparison:
    def test_output_columns(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = ResidualComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RESIDUAL"
        assert col in result.columns

    def test_nulls_before_window(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = ResidualComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RESIDUAL"
        vals = result[col].to_list()
        assert all(v is None for v in vals[:4])

    def test_non_null_after_window(self, params, rolling_dfs):
        base_df, bench_df = rolling_dfs
        result = ResidualComparison(params, lookback_window=5).compute(base_df, bench_df)
        col = "SALES__YOY_GROWTH__DAILY__VS__ZERO_BENCHMARK__RESIDUAL"
        assert result[col][-1] is not None
