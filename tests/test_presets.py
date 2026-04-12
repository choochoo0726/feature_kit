import polars as pl
import pytest
from datetime import date
from kpi_metrics import KpiMetricsBuilder, Param
from kpi_metrics.presets import sales_yoy_vs_zero, sales_rolling_vs_zero


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def raw_df():
    return pl.DataFrame({
        "specid": ["A"] * 10 + ["B"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)] * 2,
        "SALES": [float(100 + i * 10) for i in range(10)] * 2,
    })


class TestPresets:
    def test_sales_yoy_vs_zero_returns_builder(self, params):
        builder = sales_yoy_vs_zero(params)
        assert isinstance(builder, KpiMetricsBuilder)

    def test_sales_rolling_vs_zero_returns_builder(self, params):
        builder = sales_rolling_vs_zero(params)
        assert isinstance(builder, KpiMetricsBuilder)

    def test_sales_yoy_vs_zero_compute_runs(self, params, raw_df):
        builder = sales_yoy_vs_zero(params)
        result = builder.compute(raw_df)
        assert isinstance(result, pl.DataFrame)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns

    def test_sales_rolling_vs_zero_compute_runs(self, params, raw_df):
        builder = sales_rolling_vs_zero(params)
        result = builder.compute(raw_df)
        assert isinstance(result, pl.DataFrame)
        comp_cols = [c for c in result.columns if "__VS__" in c]
        assert len(comp_cols) > 0


class TestPublicAPI:
    def test_import_param(self):
        from kpi_metrics import Param
        p = Param(keys=["specid"], metrics=["SALES"])
        assert p.date_col == "DATADATE"

    def test_import_builder(self):
        from kpi_metrics import KpiMetricsBuilder
        assert KpiMetricsBuilder is not None

    def test_import_base_features(self):
        from kpi_metrics import (
            RollingSumBaseFeature,
            YOYGrowthBaseFeature,
            POPGrowthBaseFeature,
        )
        assert RollingSumBaseFeature is not None

    def test_import_benchmarks(self):
        from kpi_metrics import ZeroBenchmark, LaggedYOYGrowthBenchmark
        assert ZeroBenchmark is not None

    def test_import_comparisons(self):
        from kpi_metrics import DifferenceComparison, ZScoreComparison
        assert DifferenceComparison is not None

    def test_import_transformations(self):
        from kpi_metrics import ClipTransformation, WinsorizeTransformation
        assert ClipTransformation is not None
