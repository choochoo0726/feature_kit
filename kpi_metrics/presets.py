from __future__ import annotations

from kpi_metrics.base_features import YOYGrowthBaseFeature, RollingSumBaseFeature
from kpi_metrics.benchmarks import ZeroBenchmark
from kpi_metrics.builder import KpiMetricsBuilder
from kpi_metrics.comparisons import DifferenceComparison, SignComparison
from kpi_metrics.params import Param


def sales_yoy_vs_zero(params: Param) -> KpiMetricsBuilder:
    """YoY growth features compared against zero benchmark (difference + sign)."""
    return KpiMetricsBuilder(
        params=params,
        base_feature=YOYGrowthBaseFeature(params),
        benchmarks=[ZeroBenchmark(params)],
        comparisons=[DifferenceComparison(), SignComparison()],
    )


def sales_rolling_vs_zero(params: Param) -> KpiMetricsBuilder:
    """Rolling sum features compared against zero benchmark (difference + sign)."""
    return KpiMetricsBuilder(
        params=params,
        base_feature=RollingSumBaseFeature(params),
        benchmarks=[ZeroBenchmark(params)],
        comparisons=[DifferenceComparison(), SignComparison()],
    )
