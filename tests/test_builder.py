import polars as pl
import pytest
from datetime import date
from kpi_metrics.params import Param
from kpi_metrics.base_features import RollingSumBaseFeature
from kpi_metrics.benchmarks import ZeroBenchmark
from kpi_metrics.comparisons import DifferenceComparison
from kpi_metrics.transformations import ClipTransformation
from kpi_metrics.builder import KpiMetricsBuilder


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


class TestKpiMetricsBuilderBasic:
    def test_returns_dataframe(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert isinstance(result, pl.DataFrame)

    def test_output_contains_keys_and_date(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns

    def test_output_has_comparison_columns(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        comp_cols = [c for c in result.columns if "__VS__" in c]
        assert len(comp_cols) > 0

    def test_no_raw_metric_in_output(self, params, raw_df):
        builder = KpiMetricsBuilder(
            params=params,
            base_feature=RollingSumBaseFeature(params, windows=[("daily", 1)]),
            benchmarks=[ZeroBenchmark(params)],
            comparisons=[DifferenceComparison()],
        )
        result = builder.compute(raw_df)
        assert "SALES" not in result.columns


class TestKpiMetricsBuilderTransformations:
    def test_transformation_applied(self, params, raw_df):
        builder = KpiMetricsBuilder(
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


class TestKpiMetricsBuilderMultipleBenchmarks:
    def test_multiple_comparisons_produce_multiple_columns(self, params, raw_df):
        from kpi_metrics.comparisons import SignComparison
        builder = KpiMetricsBuilder(
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
