# API reference

Generated from the source docstrings. Everything below is importable from the top-level package (`from feature_kit import ...`), except presets, which live in `feature_kit.presets`.

| Module | Contents |
|---|---|
| [`feature_kit.params`](params.md) | `Param`, and the abstract bases `BaseFeature`, `Benchmark`, `Comparison`, `Transformation` |
| [`feature_kit.base_features`](base_features.md) | `RollingSumBaseFeature`, `YOYGrowthBaseFeature`, `POPGrowthBaseFeature` |
| [`feature_kit.benchmarks`](benchmarks.md) | `ZeroBenchmark`, `LaggedYOYGrowthBenchmark`, `LaggedPOPGrowthBenchmark`, `LaggedRollingSumBenchmark`, `LaggedYOYProjectedRollingSumBenchmark`, `CohortRollingSumBenchmark`, `IBESConsensusSalesGrowthBenchmark` |
| [`feature_kit.comparisons`](comparisons.md) | `DifferenceComparison`, `SignComparison`, `RatioComparison`, `ZScoreComparison`, `RankComparison`, `ResidualComparison` |
| [`feature_kit.transformations`](transformations.md) | `ClipTransformation`, `WinsorizeTransformation`, `RampUpTransformation`, `RampDownTransformation` |
| [`feature_kit.builder`](builder.md) | `FeatureBuilder` |
| [`feature_kit.presets`](presets.md) | `sales_yoy_vs_zero`, `sales_rolling_vs_zero` |
