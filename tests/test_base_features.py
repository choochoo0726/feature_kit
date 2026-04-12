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


from kpi_metrics.base_features import YOYGrowthBaseFeature


class TestYOYGrowthBaseFeature:
    def test_output_columns(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = YOYGrowthBaseFeature(params, windows=[("quadweekly", 4)], yoy_days=7)
        # Build 14-day df so yoy (7 days lag) is computable
        df = pl.DataFrame({
            "specid": ["A"] * 14,
            "DATADATE": [date(2023, 1, i + 1) for i in range(14)],
            "SALES": [float(i + 1) for i in range(14)],
        })
        result = feat.compute(df)
        assert "SALES__YOY_GROWTH__QUADWEEKLY" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "SALES__YOY_GROWTH__QUADWEEKLY"}

    def test_yoy_growth_values(self):
        # Window=1 (daily), yoy_days=7: growth at day 8 = (val8 - val1) / abs(val1)
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=7)
        df = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        })
        result = feat.compute(df)
        # Days 1-7: prior is null (< yoy_days ago), growth = null
        # Day 8: (20 - 10) / 10 = 1.0
        vals = result["SALES__YOY_GROWTH__DAILY"].to_list()
        assert vals[:7] == [None] * 7
        assert vals[7] == pytest.approx(1.0)

    def test_multiple_companies_independent(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=3)
        df = pl.DataFrame({
            "specid": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(4)] * 2,
            "SALES": [10.0, 10.0, 10.0, 20.0, 5.0, 5.0, 5.0, 15.0],
        })
        result = feat.compute(df)
        a_vals = result.filter(pl.col("specid") == "A")["SALES__YOY_GROWTH__DAILY"].to_list()
        b_vals = result.filter(pl.col("specid") == "B")["SALES__YOY_GROWTH__DAILY"].to_list()
        # Day 4: (20 - 10) / 10 = 1.0 for A; (15 - 5) / 5 = 2.0 for B
        assert a_vals[-1] == pytest.approx(1.0)
        assert b_vals[-1] == pytest.approx(2.0)


from kpi_metrics.base_features import POPGrowthBaseFeature


class TestPOPGrowthBaseFeature:
    def test_output_columns(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = POPGrowthBaseFeature(params, windows=[("quadweekly", 4)])
        df = pl.DataFrame({
            "specid": ["A"] * 8,
            "DATADATE": [date(2023, 1, i + 1) for i in range(8)],
            "SALES": [float(i + 1) for i in range(8)],
        })
        result = feat.compute(df)
        assert "SALES__POP_GROWTH__QUADWEEKLY" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "SALES__POP_GROWTH__QUADWEEKLY"}

    def test_pop_growth_values(self):
        # window=("period3", 3), pop lag = 3 days
        # rolling_sum(3) at day 4 = 2+3+4=9, at day 1 = 1+...clipped=1
        # easier: window=("daily", 1) means rolling_sum=value itself, lag=1 day
        # day 2: (2-1)/1=1.0, day 3: (3-2)/2=0.5, ...
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = POPGrowthBaseFeature(params, windows=[("daily", 1)])
        df = pl.DataFrame({
            "specid": ["A"] * 5,
            "DATADATE": [date(2023, 1, i + 1) for i in range(5)],
            "SALES": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = feat.compute(df)
        vals = result["SALES__POP_GROWTH__DAILY"].to_list()
        # Day 1: no prior (1 day ago doesn't exist) → null
        assert vals[0] is None
        # Day 2: (2-1)/1 = 1.0
        assert vals[1] == pytest.approx(1.0)
        # Day 3: (3-2)/2 = 0.5
        assert vals[2] == pytest.approx(0.5)

    def test_default_windows(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = POPGrowthBaseFeature(params)
        df = pl.DataFrame({
            "specid": ["A"] * 5,
            "DATADATE": [date(2023, 1, i + 1) for i in range(5)],
            "SALES": [float(i + 1) for i in range(5)],
        })
        result = feat.compute(df)
        assert "SALES__POP_GROWTH__QUADWEEKLY" in result.columns
        assert "SALES__POP_GROWTH__QUARTERLY" in result.columns

    def test_multiple_companies_independent(self):
        params = Param(keys=["specid"], metrics=["SALES"])
        feat = POPGrowthBaseFeature(params, windows=[("daily", 1)])
        df = pl.DataFrame({
            "specid": ["A", "A", "A", "B", "B", "B"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(3)] * 2,
            "SALES": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        result = feat.compute(df)
        a_vals = result.filter(pl.col("specid") == "A")["SALES__POP_GROWTH__DAILY"].to_list()
        b_vals = result.filter(pl.col("specid") == "B")["SALES__POP_GROWTH__DAILY"].to_list()
        # Day 2: A=(2-1)/1=1.0, B=(20-10)/10=1.0
        assert a_vals[1] == pytest.approx(1.0)
        assert b_vals[1] == pytest.approx(1.0)
