import polars as pl
import pytest
from kpi_metrics.params import Param, BaseFeature, Benchmark, Comparison, Transformation


def test_param_defaults():
    p = Param(keys=["specid"], metrics=["SALES"])
    assert p.date_col == "DATADATE"
    assert p.keys == ["specid"]
    assert p.metrics == ["SALES"]


def test_param_custom_date_col():
    p = Param(keys=["cusip"], metrics=["REVENUE"], date_col="DATE")
    assert p.date_col == "DATE"


def test_base_feature_is_abstract():
    with pytest.raises(TypeError):
        BaseFeature()


def test_benchmark_is_abstract():
    with pytest.raises(TypeError):
        Benchmark()


def test_comparison_is_abstract():
    with pytest.raises(TypeError):
        Comparison()


def test_transformation_is_abstract():
    with pytest.raises(TypeError):
        Transformation()
