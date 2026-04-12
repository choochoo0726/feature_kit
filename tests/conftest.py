import pytest
import polars as pl
from datetime import date, timedelta


def make_daily_df(
    companies: list[str],
    start: date,
    n_days: int,
    metrics: dict[str, list[float]],
) -> pl.DataFrame:
    """Build a synthetic daily DataFrame sorted by (specid, DATADATE)."""
    rows = []
    dates = [start + timedelta(days=i) for i in range(n_days)]
    for company in companies:
        for i, d in enumerate(dates):
            row = {"specid": company, "DATADATE": d}
            for metric, values in metrics.items():
                row[metric] = float(values[i % len(values)])
            rows.append(row)
    return pl.DataFrame(rows).sort(["specid", "DATADATE"])


@pytest.fixture
def small_df():
    """10 days, 2 companies, 1 metric (SALES = 1..10)."""
    return make_daily_df(
        companies=["A", "B"],
        start=date(2023, 1, 1),
        n_days=10,
        metrics={"SALES": list(range(1, 11))},
    )


@pytest.fixture
def yoy_df():
    """730 days (2 years), 2 companies. SALES = day index + 1."""
    n = 730
    return make_daily_df(
        companies=["A", "B"],
        start=date(2022, 1, 1),
        n_days=n,
        metrics={"SALES": list(range(1, n + 1))},
    )
