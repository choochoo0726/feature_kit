import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.base_features import RollingSumBaseFeature


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def small_df():
    return pl.DataFrame({
        "specid": ["A"] * 5,
        "DATADATE": [date(2023, 1, i + 1) for i in range(5)],
        "SALES": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


class TestRollingSumBaseFeature:
    def test_output_columns(self, params, small_df):
        feat = RollingSumBaseFeature(params, windows=[("weekly", 3)])
        result = feat.compute(small_df)
        assert "SALES__ROLLING_SUM__WEEKLY" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "SALES__ROLLING_SUM__WEEKLY"}

    def test_rolling_sum_values(self, params, small_df):
        feat = RollingSumBaseFeature(params, windows=[("weekly", 3)])
        result = feat.compute(small_df)
        vals = result["SALES__ROLLING_SUM__WEEKLY"].to_list()
        assert vals == pytest.approx([1.0, 3.0, 6.0, 9.0, 12.0])

    def test_multiple_windows(self, params, small_df):
        feat = RollingSumBaseFeature(params, windows=[("daily", 1), ("weekly", 3)])
        result = feat.compute(small_df)
        assert "SALES__ROLLING_SUM__DAILY" in result.columns
        assert "SALES__ROLLING_SUM__WEEKLY" in result.columns

    def test_multiple_companies(self, params):
        df = pl.DataFrame({
            "specid": ["A", "A", "A", "B", "B", "B"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(3)] * 2,
            "SALES": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        feat = RollingSumBaseFeature(params, windows=[("weekly", 3)])
        result = feat.compute(df)
        a_vals = result.filter(pl.col("specid") == "A")["SALES__ROLLING_SUM__WEEKLY"].to_list()
        b_vals = result.filter(pl.col("specid") == "B")["SALES__ROLLING_SUM__WEEKLY"].to_list()
        assert a_vals == pytest.approx([1.0, 3.0, 6.0])
        assert b_vals == pytest.approx([10.0, 30.0, 60.0])

    def test_default_windows(self, params, small_df):
        feat = RollingSumBaseFeature(params)
        result = feat.compute(small_df)
        for name in ["DAILY", "WEEKLY", "QUADWEEKLY", "QUARTERLY"]:
            assert f"SALES__ROLLING_SUM__{name}" in result.columns
