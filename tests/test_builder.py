import polars as pl
import pytest
from datetime import date
from feature_kit.params import Param
from feature_kit.base_features import RollingSumBaseFeature
from feature_kit.benchmarks import ZeroBenchmark
from feature_kit.comparisons import DifferenceComparison
from feature_kit.transformations import ClipTransformation
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
