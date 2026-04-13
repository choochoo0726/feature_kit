import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.base_features import YOYGrowthBaseFeature, POPGrowthBaseFeature
from kpi_metrics.benchmarks import ZeroBenchmark, LaggedYOYGrowthBenchmark, LaggedPOPGrowthBenchmark


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def yoy_base(params):
    raw = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES": [float(i + 1) for i in range(10)],
    })
    base_df = YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=3).compute(raw)
    return base_df, raw


class TestZeroBenchmark:
    def test_output_columns(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = ZeroBenchmark(params).compute(base_df, raw_df)
        assert "ZERO_BENCHMARK" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "ZERO_BENCHMARK"}

    def test_all_zeros(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = ZeroBenchmark(params).compute(base_df, raw_df)
        assert result["ZERO_BENCHMARK"].to_list() == [0.0] * len(base_df)

    def test_aligned_to_base_df(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = ZeroBenchmark(params).compute(base_df, raw_df)
        assert len(result) == len(base_df)


class TestLaggedYOYGrowthBenchmark:
    def test_output_columns(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = LaggedYOYGrowthBenchmark(params, periods=[("daily", 1)], num_lags=2).compute(base_df, raw_df)
        assert "SALES__YOY_GROWTH__DAILY__LAG1DAILY" in result.columns
        assert "SALES__YOY_GROWTH__DAILY__LAG2DAILY" in result.columns

    def test_lagged_values_shift(self, params, yoy_base):
        base_df, raw_df = yoy_base
        # 1 lag of 1-day period: benchmark at date D = base value at D-1
        result = LaggedYOYGrowthBenchmark(params, periods=[("daily", 1)], num_lags=1).compute(base_df, raw_df)
        col = "SALES__YOY_GROWTH__DAILY__LAG1DAILY"
        base_vals = base_df["SALES__YOY_GROWTH__DAILY"].to_list()
        lag_vals = result[col].to_list()
        # first row: 1 day ago doesn't exist → null
        assert lag_vals[0] is None
        # row 4 (day 5): benchmark = base_df value at day 4
        if base_vals[3] is not None:
            assert lag_vals[4] == pytest.approx(base_vals[3])

    def test_default_periods(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = LaggedYOYGrowthBenchmark(params).compute(base_df, raw_df)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__YOY_GROWTH__DAILY__LAG1{name}" in result.columns

    def test_multiple_lags(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = LaggedYOYGrowthBenchmark(params, periods=[("daily", 1)], num_lags=3).compute(base_df, raw_df)
        for n in [1, 2, 3]:
            assert f"SALES__YOY_GROWTH__DAILY__LAG{n}DAILY" in result.columns


class TestLaggedPOPGrowthBenchmark:
    def test_output_columns(self, params):
        raw = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [float(i + 1) for i in range(10)],
        })
        base_df = POPGrowthBaseFeature(params, windows=[("daily", 1)]).compute(raw)
        result = LaggedPOPGrowthBenchmark(params, periods=[("daily", 1)], num_lags=2).compute(base_df, raw)
        assert "SALES__POP_GROWTH__DAILY__LAG1DAILY" in result.columns
        assert "SALES__POP_GROWTH__DAILY__LAG2DAILY" in result.columns

    def test_default_periods(self, params):
        raw = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [float(i + 1) for i in range(10)],
        })
        base_df = POPGrowthBaseFeature(params, windows=[("daily", 1)]).compute(raw)
        result = LaggedPOPGrowthBenchmark(params).compute(base_df, raw)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__POP_GROWTH__DAILY__LAG1{name}" in result.columns

from kpi_metrics.base_features import RollingSumBaseFeature
from kpi_metrics.benchmarks import (
    LaggedRollingSumBenchmark,
    LaggedYOYProjectedRollingSumBenchmark,
    IBESConsensusSalesGrowthBenchmark,
    CohortRollingSumBenchmark,
)


@pytest.fixture
def rolling_base(params):
    raw = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES": [float(i + 1) for i in range(10)],
    })
    base_df = RollingSumBaseFeature(params, windows=[("daily", 1)]).compute(raw)
    return base_df, raw


class TestLaggedRollingSumBenchmark:
    def test_output_columns(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params, periods=[("daily", 1)], num_lags=2).compute(base_df, raw_df)
        assert "SALES__ROLLING_SUM__DAILY__LAG1DAILY" in result.columns
        assert "SALES__ROLLING_SUM__DAILY__LAG2DAILY" in result.columns

    def test_lagged_values(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params, periods=[("daily", 1)], num_lags=1).compute(base_df, raw_df)
        col = "SALES__ROLLING_SUM__DAILY__LAG1DAILY"
        vals = result[col].to_list()
        assert vals[0] is None
        assert vals[1] == pytest.approx(1.0)

    def test_default_periods(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params).compute(base_df, raw_df)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__ROLLING_SUM__DAILY__LAG1{name}" in result.columns

    def test_multiple_lags(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params, periods=[("daily", 1)], num_lags=3).compute(base_df, raw_df)
        for n in [1, 2, 3]:
            assert f"SALES__ROLLING_SUM__DAILY__LAG{n}DAILY" in result.columns


class TestLaggedYOYProjectedRollingSumBenchmark:
    def test_output_columns(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedYOYProjectedRollingSumBenchmark(
            params, periods=[("daily", 1)], num_lags=2, yoy_days=3
        ).compute(base_df, raw_df)
        assert "SALES__ROLLING_SUM__DAILY__LAG1DAILY_YOY_PROJECTED" in result.columns
        assert "SALES__ROLLING_SUM__DAILY__LAG2DAILY_YOY_PROJECTED" in result.columns

    def test_has_keys_and_date(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedYOYProjectedRollingSumBenchmark(
            params, periods=[("daily", 1)], num_lags=1, yoy_days=3
        ).compute(base_df, raw_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns

    def test_default_periods(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedYOYProjectedRollingSumBenchmark(params, yoy_days=3).compute(base_df, raw_df)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__ROLLING_SUM__DAILY__LAG1{name}_YOY_PROJECTED" in result.columns


class TestIBESConsensusSalesGrowthBenchmark:
    def test_output_columns(self, params, yoy_base):
        base_df, raw_df = yoy_base
        ibes_df = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK": [0.05 * (i + 1) for i in range(10)],
        })
        result = IBESConsensusSalesGrowthBenchmark(params).compute(base_df, raw_df, external_dfs={"ibes": ibes_df})
        assert "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK" in result.columns

    def test_raises_without_external_dfs(self, params, yoy_base):
        base_df, raw_df = yoy_base
        with pytest.raises(ValueError, match="requires external_dfs"):
            IBESConsensusSalesGrowthBenchmark(params).compute(base_df, raw_df)

    def test_invalid_freq_raises(self, params):
        with pytest.raises(ValueError, match="freq must be"):
            IBESConsensusSalesGrowthBenchmark(params, freq="annual")


class TestCohortRollingSumBenchmark:
    def test_output_columns(self, params, rolling_base):
        base_df, raw_df = rolling_base
        raw_with_sector = raw_df.with_columns(pl.lit("tech").alias("sector"))
        result = CohortRollingSumBenchmark(params, cohort_cols=["sector"]).compute(base_df, raw_with_sector)
        assert "SALES__ROLLING_SUM__DAILY__COHORT_BENCHMARK" in result.columns

    def test_invalid_agg_raises(self, params):
        with pytest.raises(ValueError, match="agg must be"):
            CohortRollingSumBenchmark(params, cohort_cols=["sector"], agg="sum")
