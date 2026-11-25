import polars as pl
import pytest
from datetime import date
from feature_kit.params import Param
from feature_kit.comparisons import (
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

    def test_no_cross_metric_pairs(self):
        """SALES feature must not be compared against a COUNT benchmark and vice versa."""
        params = Param(keys=["specid"], metrics=["SALES", "COUNT"])
        base_df = pl.DataFrame({
            "specid": ["A", "A"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 2)],
            "SALES__YOY_GROWTH__DAILY": [0.1, 0.2],
            "COUNT__YOY_GROWTH__DAILY": [0.3, 0.4],
        })
        bench_df = pl.DataFrame({
            "specid": ["A", "A"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 2)],
            "SALES__YOY_GROWTH__DAILY__LAG1MONTH": [0.05, 0.15],
            "COUNT__YOY_GROWTH__DAILY__LAG1MONTH": [0.25, 0.35],
        })
        result = DifferenceComparison(params).compute(base_df, bench_df)
        cols = result.columns
        # Same-metric pairs should exist
        assert "SALES__YOY_GROWTH__DAILY__VS__SALES__YOY_GROWTH__DAILY__LAG1MONTH__DIFFERENCE" in cols
        assert "COUNT__YOY_GROWTH__DAILY__VS__COUNT__YOY_GROWTH__DAILY__LAG1MONTH__DIFFERENCE" in cols
        # Cross-metric pairs must not exist
        assert "SALES__YOY_GROWTH__DAILY__VS__COUNT__YOY_GROWTH__DAILY__LAG1MONTH__DIFFERENCE" not in cols
        assert "COUNT__YOY_GROWTH__DAILY__VS__SALES__YOY_GROWTH__DAILY__LAG1MONTH__DIFFERENCE" not in cols

    def test_no_cross_window_pairs(self, params):
        """A feature is only compared with benchmarks derived from that same feature (window)."""
        base_df = pl.DataFrame({
            "specid": ["A"],
            "DATADATE": [date(2023, 1, 1)],
            "SALES__YOY_GROWTH__QUADWEEKLY": [0.1],
            "SALES__YOY_GROWTH__QUARTERLY": [0.2],
        })
        bench_df = pl.DataFrame({
            "specid": ["A"],
            "DATADATE": [date(2023, 1, 1)],
            "SALES__YOY_GROWTH__QUADWEEKLY__LAG1WEEK": [0.05],
            "SALES__YOY_GROWTH__QUARTERLY__LAG1WEEK": [0.15],
        })
        result = DifferenceComparison(params).compute(base_df, bench_df)
        signal_cols = [c for c in result.columns if "__VS__" in c]
        assert signal_cols == [
            "SALES__YOY_GROWTH__QUADWEEKLY__VS__SALES__YOY_GROWTH__QUADWEEKLY__LAG1WEEK__DIFFERENCE",
            "SALES__YOY_GROWTH__QUARTERLY__VS__SALES__YOY_GROWTH__QUARTERLY__LAG1WEEK__DIFFERENCE",
        ]
        assert result[signal_cols[1]].to_list() == pytest.approx([0.05])

    def test_derived_benchmark_pairs_with_source_even_without_params(self):
        """Without params, a feature-derived benchmark still pairs only with its source feature."""
        base_df = pl.DataFrame({
            "specid": ["A"],
            "DATADATE": [date(2023, 1, 1)],
            "SALES__YOY_GROWTH__DAILY": [0.1],
            "COUNT__YOY_GROWTH__DAILY": [0.3],
        })
        bench_df = pl.DataFrame({
            "specid": ["A"],
            "DATADATE": [date(2023, 1, 1)],
            "SALES__YOY_GROWTH__DAILY__LAG1MONTH": [0.05],
        })
        result = DifferenceComparison().compute(base_df, bench_df)
        assert [c for c in result.columns if "__VS__" in c] == [
            "SALES__YOY_GROWTH__DAILY__VS__SALES__YOY_GROWTH__DAILY__LAG1MONTH__DIFFERENCE",
        ]

    def test_unprefixed_benchmark_pairs_with_every_feature(self, params):
        base_df = pl.DataFrame({
            "specid": ["A"],
            "DATADATE": [date(2023, 1, 1)],
            "SALES__YOY_GROWTH__QUADWEEKLY": [0.1],
            "SALES__YOY_GROWTH__QUARTERLY": [0.2],
        })
        bench_df = pl.DataFrame({
            "specid": ["A"],
            "DATADATE": [date(2023, 1, 1)],
            "ZERO_BENCHMARK": [0.0],
        })
        result = DifferenceComparison(params).compute(base_df, bench_df)
        assert len([c for c in result.columns if "__VS__" in c]) == 2


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


class TestPairingWithoutParams:
    def test_metric_prefixed_benchmark_stays_within_metric(self):
        """Without params, a SALES-prefixed (non-derived) benchmark must not pair with TXN features."""
        base_df = pl.DataFrame({
            "specid": ["A"], "DATADATE": [date(2023, 1, 1)],
            "SALES__YOY_GROWTH__Q": [0.1], "TXN__YOY_GROWTH__Q": [0.2],
        })
        bench_df = pl.DataFrame({
            "specid": ["A"], "DATADATE": [date(2023, 1, 1)],
            "SALES__CUSTOM_BENCHMARK": [0.0], "ZERO_BENCHMARK": [0.0],
        })
        cols = [c for c in DifferenceComparison().compute(base_df, bench_df).columns if "__VS__" in c]
        assert cols == [
            "SALES__YOY_GROWTH__Q__VS__SALES__CUSTOM_BENCHMARK__DIFFERENCE",
            "SALES__YOY_GROWTH__Q__VS__ZERO_BENCHMARK__DIFFERENCE",
            "TXN__YOY_GROWTH__Q__VS__ZERO_BENCHMARK__DIFFERENCE",
        ]
