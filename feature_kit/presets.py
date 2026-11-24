"""Ready-made ``FeatureBuilder`` configurations."""

from __future__ import annotations

from feature_kit.base_features import YOYGrowthBaseFeature, RollingSumBaseFeature
from feature_kit.benchmarks import ZeroBenchmark
from feature_kit.builder import FeatureBuilder
from feature_kit.comparisons import DifferenceComparison, SignComparison
from feature_kit.params import Param


def sales_yoy_vs_zero(params: Param) -> FeatureBuilder:
    """YoY growth vs. zero: "is growth positive, and by how much?".

    Builds ``YOYGrowthBaseFeature`` (default windows: quadweekly 28,
    quarterly 91) → ``ZeroBenchmark`` → ``DifferenceComparison`` +
    ``SignComparison``, with no transformations.

    Parameters
    ----------
    params : Param
        Shared column configuration. Despite the name, works for any metric
        in ``params.metrics``.

    Returns
    -------
    FeatureBuilder
        Call ``.compute(raw_df)`` on it. Output columns look like
        ``SALES__YOY_GROWTH__QUADWEEKLY__VS__ZERO_BENCHMARK__DIFFERENCE``.
    """
    return FeatureBuilder(
        params=params,
        base_feature=YOYGrowthBaseFeature(params),
        benchmarks=[ZeroBenchmark(params)],
        comparisons=[DifferenceComparison(), SignComparison()],
    )


def sales_rolling_vs_zero(params: Param) -> FeatureBuilder:
    """Rolling sums vs. zero (difference + sign).

    Builds ``RollingSumBaseFeature`` (default windows: daily 1, weekly 7,
    quadweekly 28, quarterly 91) → ``ZeroBenchmark`` →
    ``DifferenceComparison`` + ``SignComparison``. The difference simply
    equals the rolling sum, so this preset mostly serves as a template.

    Parameters
    ----------
    params : Param
        Shared column configuration.

    Returns
    -------
    FeatureBuilder
        Output columns look like
        ``SALES__ROLLING_SUM__WEEKLY__VS__ZERO_BENCHMARK__SIGN``.
    """
    return FeatureBuilder(
        params=params,
        base_feature=RollingSumBaseFeature(params),
        benchmarks=[ZeroBenchmark(params)],
        comparisons=[DifferenceComparison(), SignComparison()],
    )
