import datetime as dt

import pyarrow as pa
import sympy as sp

from dqx import data
from dqx.api import VerificationSuite, check
from dqx.common import Context, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.display import print_analysis_report, print_assertion_results, print_metrics_by_execution_id
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


@check(name="Simple Checks", datasets=["ds1"])
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("delivered")).where(name="Delivered null count is less than 100").is_leq(100)
    ctx.assert_that(mp.minimum("quantity")).where(name="Minimum quantity check").is_leq(2.5)
    ctx.assert_that(mp.average("price")).where(name="Average price check").is_geq(10.0)
    ctx.assert_that(mp.ext.day_over_day(mp.average("tax"))).where(name="Tax day-over-day check").is_geq(0.5)
    ctx.assert_that(mp.duplicate_count(["name"], dataset="ds1")).where(name="No duplicates on name").is_eq(0)
    ctx.assert_that(mp.minimum("quantity", dataset="ds1")).where(
        name="Quantity minimum is between 1 and 5",
    ).is_between(1, 5.0)
    ctx.assert_that(mp.count_values("name", "np", dataset="ds1")).where(name="NP never buys here").is_eq(0)


@check(name="Delivered null percentage", datasets=["ds1"])
def null_percentage(mp: MetricProvider, ctx: Context) -> None:
    null_count = mp.null_count("delivered", dataset="ds1")
    nr = mp.num_rows()
    ctx.assert_that(null_count / nr).where(name="null percentage is less than 40%").is_leq(0.4)


@check(name="Manual Day Over Day", datasets=["ds1"])
def manual_day_over_day(mp: MetricProvider, ctx: Context) -> None:
    tax_avg = mp.average("tax")
    tax_avg_lag = mp.average("tax", key=ctx.key.lag(1))
    ctx.assert_that(tax_avg / tax_avg_lag).where(name="Tax average day-over-day equals 1.0").is_eq(1.0, tol=0.01)


@check(name="Rate of change", datasets=["ds2"])
def rate_of_change(mp: MetricProvider, ctx: Context) -> None:
    tax_dod = mp.ext.day_over_day(mp.maximum("tax"))
    tax_wow = mp.ext.week_over_week(mp.average("tax"))
    rate = sp.Abs(tax_dod - 1.0)
    ctx.assert_that(rate).where(name="Maximum tax rate change is less than 20%").is_leq(0.2)
    ctx.assert_that(tax_wow).where(name="Average tax week-over-week change is less than 30%").is_leq(0.3)


@check(name="Cross Dataset Check", datasets=["ds1", "ds2"])
def cross_dataset_check(mp: MetricProvider, ctx: Context) -> None:
    tax_avg_1 = mp.average("tax", dataset="ds1")
    tax_avg_2 = mp.average("tax", dataset="ds2")

    ctx.assert_that(sp.Abs(tax_avg_1 / tax_avg_2 - 1)).where(name="Tax average ratio between datasets").is_lt(
        0.2, tol=0.01
    )
    ctx.assert_that(mp.first("tax", dataset="ds1")).where(name="random tax value").noop()


def create_ground_truth(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> dict[str, float | str]:
    """Create ground truth values for all symbols x_1 through x_22.

    Args:
        commerce_data_c1: Dataset 1 (seed 1050)
        commerce_data_c2: Dataset 2 (seed 2100)

    Returns:
        Dictionary mapping symbol names to expected values
    """
    import pyarrow.compute as pc

    ground_truth: dict[str, float | str] = {}

    # x_1: null_count(delivered) on ds1
    delivered_col = commerce_data_c1["delivered"]
    ground_truth["x_1"] = float(pc.sum(pc.is_null(delivered_col)).as_py())

    # x_2: minimum(quantity) on ds1
    ground_truth["x_2"] = float(pc.min(commerce_data_c1["quantity"]).as_py())

    # x_3: average(price) on ds1
    ground_truth["x_3"] = float(pc.mean(commerce_data_c1["price"]).as_py())

    # x_4: average(tax) on ds1
    ground_truth["x_4"] = float(pc.mean(commerce_data_c1["tax"]).as_py())

    # x_5: day_over_day(average(tax)) on ds1
    # Since historical data is identical, this equals 1.0
    ground_truth["x_5"] = 1.0

    # x_6: lag(1)(x_4) on ds1 for 2025-01-14
    # Since historical data is identical, this equals x_4
    ground_truth["x_6"] = ground_truth["x_4"]

    # x_7: duplicate_count(name) on ds1
    # Count duplicates by getting unique count and subtracting from total
    name_col = commerce_data_c1["name"]
    unique_names = pc.unique(name_col)
    total_rows = len(commerce_data_c1)
    unique_count = len(unique_names)
    ground_truth["x_7"] = float(total_rows - unique_count)

    # x_8: minimum(quantity) on ds1 (duplicate of x_2)
    ground_truth["x_8"] = ground_truth["x_2"]

    # x_9: count_values(name, "np") on ds1
    # Count occurrences of "np" in name column
    name_values = commerce_data_c1["name"].to_pylist()
    ground_truth["x_9"] = float(sum(1 for name in name_values if name == "np"))

    # x_10: average(tax) on ds1 (duplicate of x_4)
    ground_truth["x_10"] = ground_truth["x_4"]

    # x_11: average(tax) on ds1 for 2025-01-14
    # Since historical data is identical, this equals x_4
    ground_truth["x_11"] = ground_truth["x_4"]

    # x_12: maximum(tax) on ds2
    ground_truth["x_12"] = float(pc.max(commerce_data_c2["tax"]).as_py())

    # x_13: day_over_day(maximum(tax)) on ds2
    # Since historical data is identical, this equals 1.0
    ground_truth["x_13"] = 1.0

    # x_14: lag(1)(x_12) on ds2 for 2025-01-14
    # Since historical data is identical, this equals x_12
    ground_truth["x_14"] = ground_truth["x_12"]

    # x_15: average(tax) on ds2
    ground_truth["x_15"] = float(pc.mean(commerce_data_c2["tax"]).as_py())

    # x_16: week_over_week(average(tax)) on ds2
    # Since historical data is identical, this equals 1.0
    ground_truth["x_16"] = 1.0

    # x_17: lag(7)(x_15) on ds2 for 2025-01-08
    # Since historical data is identical, this equals x_15
    ground_truth["x_17"] = ground_truth["x_15"]

    # x_18: null_count(delivered) on ds1 (duplicate of x_1)
    ground_truth["x_18"] = ground_truth["x_1"]

    # x_19: num_rows() on ds1
    ground_truth["x_19"] = float(len(commerce_data_c1))

    # x_20: average(tax) on ds1 (duplicate of x_4)
    ground_truth["x_20"] = ground_truth["x_4"]

    # x_21: average(tax) on ds2 (duplicate of x_15)
    ground_truth["x_21"] = ground_truth["x_15"]

    # x_22: first(tax) on ds1
    ground_truth["x_22"] = float(commerce_data_c1["tax"][0].as_py())

    return ground_truth


def print_ground_truth(ground_truth: dict[str, float | str]) -> None:
    """Display ground truth values in a formatted table.

    Args:
        ground_truth: Dictionary mapping symbol names to expected values
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Ground Truth Values for Symbols")

    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Expected Value", style="green")
    table.add_column("Description", style="yellow")

    descriptions = {
        "x_1": "null_count(delivered) on ds1",
        "x_2": "minimum(quantity) on ds1",
        "x_3": "average(price) on ds1",
        "x_4": "average(tax) on ds1",
        "x_5": "day_over_day(average(tax)) on ds1",
        "x_6": "lag(1)(x_4) on ds1",
        "x_7": "duplicate_count(name) on ds1",
        "x_8": "minimum(quantity) on ds1 (duplicate)",
        "x_9": "count_values(name, 'np') on ds1",
        "x_10": "average(tax) on ds1 (duplicate)",
        "x_11": "average(tax) on ds1 for 2025-01-14",
        "x_12": "maximum(tax) on ds2",
        "x_13": "day_over_day(maximum(tax)) on ds2",
        "x_14": "lag(1)(x_12) on ds2",
        "x_15": "average(tax) on ds2",
        "x_16": "week_over_week(average(tax)) on ds2",
        "x_17": "lag(7)(x_15) on ds2",
        "x_18": "null_count(delivered) on ds1 (duplicate)",
        "x_19": "num_rows() on ds1",
        "x_20": "average(tax) on ds1 (duplicate)",
        "x_21": "average(tax) on ds2 (duplicate)",
        "x_22": "first(tax) on ds1",
    }

    # Sort symbols by numeric index (x_1, x_2, ..., x_22)
    sorted_symbols = sorted(ground_truth.keys(), key=lambda x: int(x.split("_")[1]))

    for symbol in sorted_symbols:
        value = ground_truth[symbol]
        description = descriptions.get(symbol, "Unknown")

        if isinstance(value, float):
            table.add_row(symbol, f"{value:.10f}", description)
        else:
            table.add_row(symbol, str(value), description)

    console.print(table)


def test_e2e_suite(commerce_data_c1: pa.Table, commerce_data_c2: pa.Table) -> None:
    db = InMemoryMetricDB()
    ds1 = DuckRelationDataSource.from_arrow(commerce_data_c1, "ds1")
    ds2 = DuckRelationDataSource.from_arrow(commerce_data_c2, "ds2")

    key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})
    checks = [simple_checks, manual_day_over_day, rate_of_change, null_percentage, cross_dataset_check]

    # Run for today
    suite = VerificationSuite(checks, db, name="Simple test suite")

    suite.run([ds1, ds2], key)
    suite.graph.print_tree()

    print_assertion_results(suite.collect_results())
    suite.provider.print_symbols(key)

    # Create and display ground truth
    print("\n" + "=" * 80 + "\n")
    ground_truth = create_ground_truth(commerce_data_c1, commerce_data_c2)
    print_ground_truth(ground_truth)

    print_metrics_by_execution_id(data.metrics_by_execution_id(db, suite.execution_id), suite.execution_id)
    print_analysis_report(suite._analysis_reports)
