import polars as pl
import pytest
from datetime import date
from feature_kit.params import Param
from feature_kit.base_features import YOYGrowthBaseFeature, POPGrowthBaseFeature
from feature_kit.benchmarks import ZeroBenchmark, LaggedYOYGrowthBenchmark, LaggedPOPGrowthBenchmark


@pytest.fixture
def params():
    return Param(keys=["specid"], metrics=["SALES"])


@pytest.fixture
def yoy_base(params):
    raw = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES": [float(i + 1) for i in range(10)],
    })
    base_df = YOYGrowthBaseFeature(params, windows=[("daily", 1)], yoy_days=3).compute(raw)
    return base_df, raw


class TestZeroBenchmark:
    def test_output_columns(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = ZeroBenchmark(params).compute(base_df, raw_df)
        assert "ZERO_BENCHMARK" in result.columns
        assert set(result.columns) == {"specid", "DATADATE", "ZERO_BENCHMARK"}

    def test_all_zeros(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = ZeroBenchmark(params).compute(base_df, raw_df)
        assert result["ZERO_BENCHMARK"].to_list() == [0.0] * len(base_df)

    def test_aligned_to_base_df(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = ZeroBenchmark(params).compute(base_df, raw_df)
        assert len(result) == len(base_df)


class TestLaggedYOYGrowthBenchmark:
    def test_output_columns(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = LaggedYOYGrowthBenchmark(params, periods=[("daily", 1)], num_lags=2).compute(base_df, raw_df)
        assert "SALES__YOY_GROWTH__DAILY__LAG1DAILY" in result.columns
        assert "SALES__YOY_GROWTH__DAILY__LAG2DAILY" in result.columns

    def test_lagged_values_shift(self, params, yoy_base):
        base_df, raw_df = yoy_base
        # 1 lag of 1-day period: benchmark at date D = base value at D-1
        result = LaggedYOYGrowthBenchmark(params, periods=[("daily", 1)], num_lags=1).compute(base_df, raw_df)
        col = "SALES__YOY_GROWTH__DAILY__LAG1DAILY"
        base_vals = base_df["SALES__YOY_GROWTH__DAILY"].to_list()
        lag_vals = result[col].to_list()
        # first row: 1 day ago doesn't exist → null
        assert lag_vals[0] is None
        # row 4 (day 5): benchmark = base_df value at day 4
        if base_vals[3] is not None:
            assert lag_vals[4] == pytest.approx(base_vals[3])

    def test_default_periods(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = LaggedYOYGrowthBenchmark(params).compute(base_df, raw_df)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__YOY_GROWTH__DAILY__LAG1{name}" in result.columns

    def test_multiple_lags(self, params, yoy_base):
        base_df, raw_df = yoy_base
        result = LaggedYOYGrowthBenchmark(params, periods=[("daily", 1)], num_lags=3).compute(base_df, raw_df)
        for n in [1, 2, 3]:
            assert f"SALES__YOY_GROWTH__DAILY__LAG{n}DAILY" in result.columns


class TestLaggedPOPGrowthBenchmark:
    def test_output_columns(self, params):
        raw = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [float(i + 1) for i in range(10)],
        })
        base_df = POPGrowthBaseFeature(params, windows=[("daily", 1)]).compute(raw)
        result = LaggedPOPGrowthBenchmark(params, periods=[("daily", 1)], num_lags=2).compute(base_df, raw)
        assert "SALES__POP_GROWTH__DAILY__LAG1DAILY" in result.columns
        assert "SALES__POP_GROWTH__DAILY__LAG2DAILY" in result.columns

    def test_default_periods(self, params):
        raw = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "SALES": [float(i + 1) for i in range(10)],
        })
        base_df = POPGrowthBaseFeature(params, windows=[("daily", 1)]).compute(raw)
        result = LaggedPOPGrowthBenchmark(params).compute(base_df, raw)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__POP_GROWTH__DAILY__LAG1{name}" in result.columns

from feature_kit.base_features import RollingSumBaseFeature
from feature_kit.benchmarks import (
    LaggedRollingSumBenchmark,
    LaggedYOYProjectedRollingSumBenchmark,
    IBESConsensusSalesGrowthBenchmark,
    CohortRollingSumBenchmark,
)


@pytest.fixture
def rolling_base(params):
    raw = pl.DataFrame({
        "specid": ["A"] * 10,
        "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
        "SALES": [float(i + 1) for i in range(10)],
    })
    base_df = RollingSumBaseFeature(params, windows=[("daily", 1)]).compute(raw)
    return base_df, raw


class TestLaggedRollingSumBenchmark:
    def test_output_columns(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params, periods=[("daily", 1)], num_lags=2).compute(base_df, raw_df)
        assert "SALES__ROLLING_SUM__DAILY__LAG1DAILY" in result.columns
        assert "SALES__ROLLING_SUM__DAILY__LAG2DAILY" in result.columns

    def test_lagged_values(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params, periods=[("daily", 1)], num_lags=1).compute(base_df, raw_df)
        col = "SALES__ROLLING_SUM__DAILY__LAG1DAILY"
        vals = result[col].to_list()
        assert vals[0] is None
        assert vals[1] == pytest.approx(1.0)

    def test_default_periods(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params).compute(base_df, raw_df)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__ROLLING_SUM__DAILY__LAG1{name}" in result.columns

    def test_multiple_lags(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedRollingSumBenchmark(params, periods=[("daily", 1)], num_lags=3).compute(base_df, raw_df)
        for n in [1, 2, 3]:
            assert f"SALES__ROLLING_SUM__DAILY__LAG{n}DAILY" in result.columns


class TestLaggedYOYProjectedRollingSumBenchmark:
    def test_output_columns(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedYOYProjectedRollingSumBenchmark(
            params, periods=[("daily", 1)], num_lags=2, yoy_days=3
        ).compute(base_df, raw_df)
        assert "SALES__ROLLING_SUM__DAILY__LAG1DAILY_YOY_PROJECTED" in result.columns
        assert "SALES__ROLLING_SUM__DAILY__LAG2DAILY_YOY_PROJECTED" in result.columns

    def test_has_keys_and_date(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedYOYProjectedRollingSumBenchmark(
            params, periods=[("daily", 1)], num_lags=1, yoy_days=3
        ).compute(base_df, raw_df)
        assert "specid" in result.columns
        assert "DATADATE" in result.columns

    def test_default_periods(self, params, rolling_base):
        base_df, raw_df = rolling_base
        result = LaggedYOYProjectedRollingSumBenchmark(params, yoy_days=3).compute(base_df, raw_df)
        # default: periods=[("weekly",7),("monthly",28),("quarterly",91)], num_lags=1
        for name in ["WEEK", "MONTH", "QUARTER"]:
            assert f"SALES__ROLLING_SUM__DAILY__LAG1{name}_YOY_PROJECTED" in result.columns


class TestIBESConsensusSalesGrowthBenchmark:
    def test_output_columns(self, params, yoy_base):
        base_df, raw_df = yoy_base
        ibes_df = pl.DataFrame({
            "specid": ["A"] * 10,
            "DATADATE": [date(2023, 1, i + 1) for i in range(10)],
            "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK": [0.05 * (i + 1) for i in range(10)],
        })
        result = IBESConsensusSalesGrowthBenchmark(params).compute(base_df, raw_df, external_dfs={"ibes": ibes_df})
        assert result.columns == ["specid", "DATADATE", "SALES__IBES_QUARTERLY_SALES_GROWTH_BENCHMARK"]

    def test_raises_without_external_dfs(self, params, yoy_base):
        base_df, raw_df = yoy_base
        with pytest.raises(ValueError, match="requires external_dfs"):
            IBESConsensusSalesGrowthBenchmark(params).compute(base_df, raw_df)

    def test_invalid_freq_raises(self, params):
        with pytest.raises(ValueError, match="freq must be"):
            IBESConsensusSalesGrowthBenchmark(params, freq="annual")

    def test_metric_required_with_several_metrics(self):
        params = Param(keys=["specid"], metrics=["SALES", "TXN"])
        with pytest.raises(ValueError, match="needs metric"):
            IBESConsensusSalesGrowthBenchmark(params)
        with pytest.raises(ValueError, match="not in params.metrics"):
            IBESConsensusSalesGrowthBenchmark(params, metric="REVENUE")
        assert IBESConsensusSalesGrowthBenchmark(params, metric="TXN").metric == "TXN"

    def test_backward_asof_order_and_no_warning(self, recwarn):
        params = Param(keys=["specid"], metrics=["SALES"])
        # base_df deliberately not sorted by entity
        base_df = pl.DataFrame({
            "specid": ["B", "A", "B", "A"],
            "DATADATE": [date(2023, 1, 5), date(2023, 1, 5), date(2023, 1, 1), date(2023, 1, 2)],
            "SALES__YOY_GROWTH__Q": [0.0] * 4,
        })
        ibes_df = pl.DataFrame({
            "specid": ["A", "A", "B"],
            "DATADATE": [date(2023, 1, 1), date(2023, 1, 4), date(2023, 1, 3)],
            "IBES_QUARTERLY_SALES_GROWTH_BENCHMARK": [0.1, 0.2, 0.3],
        })
        result = IBESConsensusSalesGrowthBenchmark(params).compute(base_df, base_df, {"ibes": ibes_df})
        assert result.select("specid", "DATADATE").equals(base_df.select("specid", "DATADATE"))
        # B@1/5 -> 0.3 (1/3), A@1/5 -> 0.2 (1/4), B@1/1 -> none yet, A@1/2 -> 0.1 (1/1)
        assert result["SALES__IBES_QUARTERLY_SALES_GROWTH_BENCHMARK"].to_list() == [0.3, 0.2, None, 0.1]
        assert not [w for w in recwarn if "Sortedness" in str(w.message)]

    def test_pairs_only_with_its_metric(self):
        from feature_kit.comparisons import DifferenceComparison
        base_df = pl.DataFrame({
            "specid": ["A"], "DATADATE": [date(2023, 1, 1)],
            "SALES__YOY_GROWTH__Q": [0.1], "TXN__YOY_GROWTH__Q": [0.2],
        })
        bench_df = pl.DataFrame({
            "specid": ["A"], "DATADATE": [date(2023, 1, 1)],
            "SALES__IBES_QUARTERLY_SALES_GROWTH_BENCHMARK": [0.05],
        })
        cols = [c for c in DifferenceComparison().compute(base_df, bench_df).columns if "__VS__" in c]
        assert cols == ["SALES__YOY_GROWTH__Q__VS__SALES__IBES_QUARTERLY_SALES_GROWTH_BENCHMARK__DIFFERENCE"]


class TestCohortRollingSumBenchmark:
    def test_output_columns(self, params, rolling_base):
        base_df, raw_df = rolling_base
        raw_with_sector = raw_df.with_columns(pl.lit("tech").alias("sector"))
        result = CohortRollingSumBenchmark(params, cohort_cols=["sector"]).compute(base_df, raw_with_sector)
        assert "SALES__ROLLING_SUM__DAILY__COHORT_BENCHMARK" in result.columns

    def test_invalid_agg_raises(self, params):
        with pytest.raises(ValueError, match="agg must be"):
            CohortRollingSumBenchmark(params, cohort_cols=["sector"], agg="sum")


def _brute_force_cohort(df, value_col, cohort_col, agg, exclude_self):
    """Reference implementation: loop over rows in plain Python."""
    import statistics
    rows = df.to_dicts()
    out = []
    for r in rows:
        peers = [
            q[value_col] for q in rows
            if q[cohort_col] == r[cohort_col] and q["DATADATE"] == r["DATADATE"]
            and q[value_col] is not None
            and not (exclude_self and q["specid"] == r["specid"])
        ]
        if not peers:
            out.append(None)
        else:
            out.append(statistics.fmean(peers) if agg == "mean" else statistics.median(peers))
    return out


class TestCohortLeaveOneOut:
    @pytest.fixture
    def cohort_data(self):
        import random
        rng = random.Random(7)
        rows = []
        for d in range(4):
            for i in range(9):
                v = None if rng.random() < 0.15 else float(rng.choice([1, 2, 2, 3, 5, 8, 8, 13]))  # ties + nulls
                rows.append({
                    "specid": f"E{i}",
                    "DATADATE": date(2023, 1, d + 1),
                    "SECTOR": ["x", "y", "z"][i % 3] if i < 8 else "solo",
                    "SALES__ROLLING_SUM__W": v,
                })
        df = pl.DataFrame(rows)
        return df, df.select("specid", "DATADATE", "SECTOR")

    @pytest.mark.parametrize("agg", ["mean", "median"])
    @pytest.mark.parametrize("exclude_self", [False, True])
    def test_matches_brute_force(self, cohort_data, agg, exclude_self):
        base_df, raw_df = cohort_data
        params = Param(keys=["specid"], metrics=["SALES"])
        bench = CohortRollingSumBenchmark(params, ["SECTOR"], agg=agg, exclude_self=exclude_self)
        result = bench.compute(base_df.drop("SECTOR"), raw_df)
        got = result["SALES__ROLLING_SUM__W__COHORT_BENCHMARK"].to_list()
        want = _brute_force_cohort(base_df, "SALES__ROLLING_SUM__W", "SECTOR", agg, exclude_self)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert (g is None and w is None) or g == pytest.approx(w)

    def test_cohort_of_one_is_null(self, cohort_data):
        base_df, raw_df = cohort_data
        params = Param(keys=["specid"], metrics=["SALES"])
        result = CohortRollingSumBenchmark(params, ["SECTOR"], exclude_self=True).compute(base_df.drop("SECTOR"), raw_df)
        solo = result.filter(pl.col("specid") == "E8")["SALES__ROLLING_SUM__W__COHORT_BENCHMARK"]
        assert solo.null_count() == len(solo)

    def test_default_includes_self(self, params):
        assert CohortRollingSumBenchmark(params, cohort_cols=["sector"]).exclude_self is False


class TestProjectedZeroDenominator:
    def test_zero_denominator_gives_null(self):
        from feature_kit.benchmarks import LaggedYOYProjectedRollingSumBenchmark
        params = Param(keys=["specid"], metrics=["SALES"])
        # projection at t = S(t-4) * S(t-1) / S(t-5)   (yoy_days=4, lag=1)
        base_df = pl.DataFrame({
            "specid": ["A"] * 7,
            "DATADATE": [date(2023, 1, i + 1) for i in range(7)],
            "SALES__ROLLING_SUM__W": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        })
        bench = LaggedYOYProjectedRollingSumBenchmark(params, periods=[("d", 1)], yoy_days=4)
        vals = bench.compute(base_df, base_df)["SALES__ROLLING_SUM__W__LAG1D_YOY_PROJECTED"].to_list()
        assert vals[5] is None                              # S(1/1) = 0 -> null, not inf
        assert vals[6] == pytest.approx(3.0 * 6.0 / 2.0)    # S(1/3) * S(1/6) / S(1/2)
        assert vals[:5] == [None] * 5                       # not enough history
