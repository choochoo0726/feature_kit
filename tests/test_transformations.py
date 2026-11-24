import polars as pl
import pytest
from datetime import date
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
    def test_clips_extremes(self):
        df = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "FEAT": [float(i) for i in range(10)],
        })
        result = WinsorizeTransformation(lower_q=0.1, upper_q=0.9).compute(df)
        vals = result["FEAT"].to_list()
        assert min(vals) >= 0.9 - 1e-6
        assert max(vals) <= 8.1 + 1e-6


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
