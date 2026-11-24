import polars as pl
import pytest
from datetime import date
from feature_kit.params import Param
from feature_kit.base_features import RollingSumBaseFeature
from feature_kit.benchmarks import ZeroBenchmark
from feature_kit.comparisons import DifferenceComparison
from feature_kit.transformations import ClipTransformation, RampUpTransformation
from feature_kit.builder import FeatureBuilder


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def raw_df():
    return pl.DataFrame({
        "specid": ["A", "A", "A", "B", "B", "B"],
        "DATADATE": [date(2023, 1, i + 1) for i in range(3)] * 2,
        "SALES": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0],
    })


class TestFeatureBuilderBasic:
    def test_returns_dataframe(self, params, raw_df):
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert isinstance(result, pl.DataFrame)

    def test_output_contains_keys_and_date(self, params, raw_df):
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns

    def test_output_has_comparison_columns(self, params, raw_df):
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        comp_cols = [c for c in result.columns if "__VS__" in c]
        assert len(comp_cols) > 0

    def test_no_raw_metric_in_output(self, params, raw_df):
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert "SALES" not in result.columns


class TestFeatureBuilderTransformations:
    def test_transformation_applied(self, params, raw_df):
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
            transformations=[ClipTransformation(lower=0.0, upper=50.0)],
        )
        result = builder.compute(raw_df)
        comp_cols = [c for c in result.columns if "__VS__" in c]
        for col in comp_cols:
            vals = [v for v in result[col].to_list() if v is not None]
            assert all(0.0 <= v <= 50.0 for v in vals)


class TestFeatureBuilderMultipleBenchmarks:
    def test_multiple_comparisons_produce_multiple_columns(self, params, raw_df):
        from feature_kit.comparisons import SignComparison
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison(), SignComparison()],
        )
        result = builder.compute(raw_df)
        diff_cols = [c for c in result.columns if "__DIFFERENCE" in c]
        sign_cols = [c for c in result.columns if "__SIGN" in c]
        assert len(diff_cols) >= 1
        assert len(sign_cols) >= 1


class TestFeatureBuilderTransformationColumns:
    def test_ramp_inside_builder(self, params, raw_df):
        df = raw_df.with_columns(pl.int_range(pl.len()).over("specid").alias("AGE"))  # 0, 1, 2
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison(params)],
            transformations=[RampUpTransformation("AGE", min_age=0, max_age=2)],
        )
        result = builder.compute(df).sort(["specid", "DATADATE"])
        col = "SALES__ROLLING_SUM__DAILY__VS__ZERO_BENCHMARK__DIFFERENCE"
        assert "AGE" not in result.columns
        assert result[col].to_list() == pytest.approx([0.0, 55.0, 120.0, 0.0, 105.0, 220.0])

    def test_age_column_not_clipped_by_earlier_transformation(self, params, raw_df):
        df = raw_df.with_columns(pl.lit(100).alias("AGE"))
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison(params)],
            transformations=[
                RampUpTransformation("AGE", min_age=0, max_age=100),
                ClipTransformation(-1000.0, 1000.0),
            ],
        )
        result = builder.compute(df)
        assert result.columns == ["specid", "DATADATE",
                                  "SALES__ROLLING_SUM__DAILY__VS__ZERO_BENCHMARK__DIFFERENCE"]

    def test_missing_required_column_raises(self, params, raw_df):
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison(params)],
            transformations=[RampUpTransformation("AGE", min_age=0, max_age=2)],
        )
        with pytest.raises(ValueError, match="AGE"):
            builder.compute(raw_df)

    def test_custom_keys_fill_transformation_params(self):
        params = Param(keys=["ticker"], metrics=["SALES"], date_col="dt")
        df = pl.DataFrame({
            "ticker": ["A", "A"],
            "dt": [date(2023, 1, 1), date(2023, 1, 2)],
            "SALES": [5.0, -5.0],
        })
        clip = ClipTransformation(-1.0, 1.0)
        builder = FeatureBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison(params)],
            transformations=[clip],
        )
        result = builder.compute(df).sort("dt")
        assert result["ticker"].to_list() == ["A", "A"]
        assert result["SALES__ROLLING_SUM__DAILY__VS__ZERO_BENCHMARK__DIFFERENCE"].to_list() == pytest.approx([1.0, -1.0])
        assert clip.params is None  # the user's object is not mutated
