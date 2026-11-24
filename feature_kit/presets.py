from __future__ import annotations

from feature_kit.base_features import YOYGrowthBaseFeature, RollingSumBaseFeature
from feature_kit.benchmarks import ZeroBenchmark
from feature_kit.builder import FeatureBuilder
from feature_kit.comparisons import DifferenceComparison, SignComparison
from feature_kit.params import Param


def sales_yoy_vs_zero(params: Param) -> FeatureBuilder:
    """YoY growth features compared against zero benchmark (difference + sign)."""
    return FeatureBuilder(
        params=params,
        base_feature=YOYGrowthBaseFeature(params),
        benchmarks=[ZeroBenchmark(params)],
        comparisons=[DifferenceComparison(), SignComparison()],
    )


def sales_rolling_vs_zero(params: Param) -> FeatureBuilder:
    """Rolling sum features compared against zero benchmark (difference + sign)."""
    return FeatureBuilder(
        params=params,
        base_feature=RollingSumBaseFeature(params),
        benchmarks=[ZeroBenchmark(params)],
        comparisons=[DifferenceComparison(), SignComparison()],
    )
