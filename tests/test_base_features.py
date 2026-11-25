import polars as pl
import pytest
from datetime import date, timedelta
from feature_kit.params import Param
from feature_kit.base_features import (
    RollingSumBaseFeature,
    YOYGrowthBaseFeature,
    POPGrowthBaseFeature,
)


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
        # first w - 1 = 2 days have less than a full window of history -> null
        assert vals[:2] == [None, None]
        assert vals[2:] == pytest.approx([6.0, 9.0, 12.0])

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
        assert a_vals[:2] == [None, None] and a_vals[2] == pytest.approx(6.0)
        assert b_vals[:2] == [None, None] and b_vals[2] == pytest.approx(60.0)

    def test_entity_history_counted_per_entity(self, params):
        """B starts later than A; B's first w - 1 days are null even though A has history."""
        df = pl.DataFrame({
            "specid": ["A"] * 4 + ["B"] * 2,
            "DATADATE": [date(2023, 1, i + 1) for i in range(4)] + [date(2023, 1, 3), date(2023, 1, 4)],
            "SALES": [1.0] * 6,
        })
        result = RollingSumBaseFeature(params, windows=[("w", 2)]).compute(df)
        b_vals = result.filter(pl.col("specid") == "B")["SALES__ROLLING_SUM__W"].to_list()
        assert b_vals == [None, pytest.approx(2.0)]

    def test_default_windows(self, params, small_df):
        feat = RollingSumBaseFeature(params)
        result = feat.compute(small_df)
        for name in ["DAILY", "WEEKLY", "QUADWEEKLY", "QUARTERLY"]:
            assert f"SALES__ROLLING_SUM__{name}" in result.columns


from feature_kit.base_features import YOYGrowthBaseFeature


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


from feature_kit.base_features import POPGrowthBaseFeature


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


def _gappy_df(present_days: list[int], value: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame({
        "specid": ["A"] * len(present_days),
        "DATADATE": [date(2023, 1, 1) + timedelta(days=d) for d in present_days],
        "SALES": [value] * len(present_days),
    })


class TestCalendarWindowsAndCoverage:
    def test_gap_sums_observed_days_only(self, params):
        # 10-day window ending on day 9; days 3 and 4 missing -> 8 of 10 observed
        df = _gappy_df([0, 1, 2, 5, 6, 7, 8, 9])
        result = RollingSumBaseFeature(params, windows=[("w", 10)], min_coverage=0.8).compute(df)
        assert result["SALES__ROLLING_SUM__W"].to_list()[-1] == pytest.approx(8.0)

    def test_row_based_window_would_have_stretched(self, params):
        # 3-day window on day 5 with days 3-4 missing: only day 5 is inside the window
        df = _gappy_df([0, 1, 2, 5])
        result = RollingSumBaseFeature(params, windows=[("w", 3)], min_coverage=0.3).compute(df)
        assert result["SALES__ROLLING_SUM__W"].to_list()[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize("min_coverage, expected", [(0.7, 7.0), (0.8, None), (0.9, None)])
    def test_coverage_threshold(self, params, min_coverage, expected):
        # 10-day window ending on day 9 with 7 observed days -> coverage 0.7
        df = _gappy_df([0, 1, 2, 3, 4, 8, 9])
        result = RollingSumBaseFeature(params, windows=[("w", 10)], min_coverage=min_coverage).compute(df)
        last = result["SALES__ROLLING_SUM__W"].to_list()[-1]
        assert last == (pytest.approx(expected) if expected is not None else None)

    def test_exact_threshold_is_valid(self, params):
        df = _gappy_df([0, 1, 2, 3, 4, 5, 6, 9])  # 8 of 10 days
        result = RollingSumBaseFeature(params, windows=[("w", 10)], min_coverage=0.8).compute(df)
        assert result["SALES__ROLLING_SUM__W"].to_list()[-1] == pytest.approx(8.0)

    def test_null_values_count_as_missing(self, params):
        df = pl.DataFrame({
            "specid": ["A"] * 4,
            "DATADATE": [date(2023, 1, i + 1) for i in range(4)],
            "SALES": [1.0, None, None, 1.0],
        })
        strict = RollingSumBaseFeature(params, windows=[("w", 4)], min_coverage=0.75).compute(df)
        loose = RollingSumBaseFeature(params, windows=[("w", 4)], min_coverage=0.5).compute(df)
        assert strict["SALES__ROLLING_SUM__W"].to_list()[-1] is None
        assert loose["SALES__ROLLING_SUM__W"].to_list()[-1] == pytest.approx(2.0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.01])
    def test_min_coverage_validation(self, params, bad):
        with pytest.raises(ValueError, match="min_coverage"):
            RollingSumBaseFeature(params, min_coverage=bad)
        with pytest.raises(ValueError, match="min_coverage"):
            YOYGrowthBaseFeature(params, min_coverage=bad)
        with pytest.raises(ValueError, match="min_coverage"):
            POPGrowthBaseFeature(params, min_coverage=bad)

    def test_yoy_null_when_prior_year_window_has_low_coverage(self, params):
        # 2 years of daily data, then drop 5 of the 7 days in the prior-year window
        n = 420
        days = [d for d in range(n) if d not in {10, 11, 12, 13, 14}]
        df = _gappy_df(days)
        result = YOYGrowthBaseFeature(params, windows=[("w", 7)], yoy_days=364).compute(df)
        by_day = dict(zip(result["DATADATE"].to_list(), result["SALES__YOY_GROWTH__W"].to_list()))
        affected = date(2023, 1, 1) + timedelta(days=364 + 14)   # prior window = days 8..14
        clean = date(2023, 1, 1) + timedelta(days=364 + 30)
        assert by_day[affected] is None
        assert by_day[clean] == pytest.approx(0.0)


class TestZeroDenominator:
    def test_yoy_zero_prior_sum_is_null(self, params):
        # SALES is 0 for the first 8 days, then 1. Window w=2, yoy_days=364.
        days = list(range(380))
        df = pl.DataFrame({
            "specid": ["A"] * len(days),
            "DATADATE": [date(2023, 1, 1) + timedelta(days=d) for d in days],
            "SALES": [0.0 if d < 8 else 1.0 for d in days],
        })
        result = YOYGrowthBaseFeature(params, windows=[("w", 2)]).compute(df)
        vals = dict(zip(result["DATADATE"].to_list(), result["SALES__YOY_GROWTH__W"].to_list()))
        day = lambda d: date(2023, 1, 1) + timedelta(days=d)
        assert vals[day(369)] is None                      # current 2, prior (days 4-5) 0 -> null, not inf
        assert vals[day(372)] == pytest.approx(1.0)        # current 2, prior (days 7-8) 1 -> 1.0

    @pytest.mark.parametrize("values", [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    def test_pop_zero_prior_sum_is_null(self, params, values):
        # x/0 and 0/0 both give null
        df = pl.DataFrame({
            "specid": ["A"] * 4,
            "DATADATE": [date(2023, 1, i + 1) for i in range(4)],
            "SALES": values,
        })
        result = POPGrowthBaseFeature(params, windows=[("w", 2)]).compute(df)
        assert result["SALES__POP_GROWTH__W"].to_list()[-1] is None
