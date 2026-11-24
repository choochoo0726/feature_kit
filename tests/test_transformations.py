import polars as pl
import pytest
from datetime import date
from feature_kit.params import Param
from feature_kit.transformations import (
    ClipTransformation,
    WinsorizeTransformation,
    RampUpTransformation,
    RampDownTransformation,
)


@pytest.fixture
def feature_df():
    return pl.DataFrame({
        "specid": ["A", "A", "A", "B", "B", "B"],
        "DATADATE": [date(2023, 1, i + 1) for i in range(3)] * 2,
        "FEAT": [-5.0, 0.5, 10.0, -3.0, 1.0, 8.0],
    })


class TestClipTransformation:
    def test_clip_values(self, feature_df):
        result = ClipTransformation(lower=-2.0, upper=5.0).compute(feature_df)
        assert result["FEAT"].to_list() == pytest.approx([-2.0, 0.5, 5.0, -2.0, 1.0, 5.0])

    def test_preserves_keys_and_date(self, feature_df):
        result = ClipTransformation(lower=-2.0, upper=5.0).compute(feature_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns


class TestWinsorizeTransformation:
    def test_pooled_clips_extremes(self):
        df = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "FEAT": [float(i) for i in range(10)],
        })
        result = WinsorizeTransformation(lower_q=0.1, upper_q=0.9, scope="pooled").compute(df)
        vals = result["FEAT"].to_list()
        assert min(vals) >= 0.9 - 1e-6
        assert max(vals) <= 8.1 + 1e-6


    def test_per_date_is_default_and_uses_only_same_date(self):
        # 2 dates x 11 entities; date 2 has a huge outlier that must not affect date 1.
        n = 11
        df = pl.DataFrame({
            "specid": [f"E{i}" for i in range(n)] * 2,
            "DATADATE": [date(2023, 1, 1)] * n + [date(2023, 1, 2)] * n,
            "FEAT": [float(i) for i in range(n)] + [float(i) for i in range(n - 1)] + [1e9],
        })
        result = WinsorizeTransformation(lower_q=0.1, upper_q=0.9).compute(df)
        day1 = result.filter(pl.col("DATADATE") == date(2023, 1, 1))["FEAT"].to_list()
        day2 = result.filter(pl.col("DATADATE") == date(2023, 1, 2))["FEAT"].to_list()
        # day 1: quantiles of 0..10 are 1 and 9
        assert min(day1) == pytest.approx(1.0)
        assert max(day1) == pytest.approx(9.0)
        # day 2: the outlier is clipped to that date's 90th percentile
        assert max(day2) < 1e9

    def test_past_rows_unchanged_by_future_data(self):
        n = 11
        base = pl.DataFrame({
            "specid": [f"E{i}" for i in range(n)],
            "DATADATE": [date(2023, 1, 1)] * n,
            "FEAT": [float(i) for i in range(n)],
        })
        future = base.with_columns(
            pl.lit(date(2023, 1, 2)).alias("DATADATE"), (pl.col("FEAT") * 1000).alias("FEAT")
        )
        w = WinsorizeTransformation(lower_q=0.1, upper_q=0.9)
        alone = w.compute(base)["FEAT"].to_list()
        with_future = w.compute(pl.concat([base, future])).filter(
            pl.col("DATADATE") == date(2023, 1, 1)
        )["FEAT"].to_list()
        assert alone == pytest.approx(with_future)

    def test_invalid_scope(self):
        with pytest.raises(ValueError, match="scope"):
            WinsorizeTransformation(scope="expanding")


class TestCustomColumnNames:
    @pytest.fixture
    def custom_df(self):
        return pl.DataFrame({
            "ticker": ["A", "A", "B", "B"],
            "dt": [date(2023, 1, 1), date(2023, 1, 2)] * 2,
            "FEAT": [-5.0, 0.5, 10.0, 1.0],
        })

    @pytest.fixture
    def custom_params(self):
        return Param(keys=["ticker"], metrics=["X"], date_col="dt")

    def test_clip_with_custom_keys(self, custom_df, custom_params):
        result = ClipTransformation(-1.0, 1.0, params=custom_params).compute(custom_df)
        assert result["ticker"].to_list() == ["A", "A", "B", "B"]
        assert result["FEAT"].to_list() == pytest.approx([-1.0, 0.5, 1.0, 1.0])

    def test_numeric_key_not_modified(self, custom_params):
        df = pl.DataFrame({"ticker": [100, 200], "dt": [date(2023, 1, 1)] * 2, "FEAT": [5.0, -5.0]})
        result = ClipTransformation(-1.0, 1.0, params=custom_params).compute(df)
        assert result["ticker"].to_list() == [100, 200]

    def test_winsorize_with_custom_date_col(self, custom_df, custom_params):
        result = WinsorizeTransformation(params=custom_params).compute(custom_df)
        assert result.columns == custom_df.columns

    def test_ramp_with_custom_keys(self, custom_params):
        df = pl.DataFrame({"ticker": ["A"], "dt": [date(2023, 1, 1)], "FEAT": [2.0], "age": [5]})
        result = RampUpTransformation("age", 0, 10, params=custom_params).compute(df)
        assert result["FEAT"].to_list() == pytest.approx([1.0])
        assert result["age"].to_list() == [5]


class TestRampUpTransformation:
    def test_scale_values(self):
        df = pl.DataFrame({
            "specid": ["A", "A", "A"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
            "FEAT": [1.0, 1.0, 1.0],
            "age": [0, 5, 10],
        })
        result = RampUpTransformation(age_col="age", min_age=0, max_age=10).compute(df)
        vals = result["FEAT"].to_list()
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(0.5)
        assert vals[2] == pytest.approx(1.0)

    def test_age_col_not_in_feature_cols(self):
        df = pl.DataFrame({
            "specid": ["A"],
            "DATADATE": [date(2023, 1, 1)],
            "FEAT": [2.0],
            "age": [5],
        })
        result = RampUpTransformation(age_col="age", min_age=0, max_age=10).compute(df)
        assert result["age"].to_list() == [5]  # age column unchanged


class TestRampDownTransformation:
    def test_scale_values(self):
        df = pl.DataFrame({
            "specid": ["A", "A", "A"],
            "DATADATE": [date(2023, 1, i + 1) for i in range(3)],
            "FEAT": [1.0, 1.0, 1.0],
            "age": [0, 5, 10],
        })
        result = RampDownTransformation(age_col="age", min_age=0, max_age=10).compute(df)
        vals = result["FEAT"].to_list()
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(0.5)
        assert vals[2] == pytest.approx(0.0)
