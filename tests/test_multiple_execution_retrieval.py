"""Integration test for metric retrieval after multiple suite executions."""

import time
from datetime import date
from typing import Any, Dict, List, Tuple

import pyarrow as pa
from returns.maybe import Some

from dqx import data, specs
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.orm.repositories import Metric as DBMetric


def _create_test_data() -> Tuple[pa.Table, pa.Table, pa.Table]:
    """Create test datasets with different values for 3 runs."""
    run1_data = pa.table(
        {
            "price": [10.0, 20.0, 30.0],  # avg = 20.0
            "quantity": [1.0, 2.0, 3.0],  # sum = 6.0
        }
    )

    run2_data = pa.table(
        {
            "price": [15.0, 25.0, 35.0],  # avg = 25.0
            "quantity": [2.0, 3.0, 4.0],  # sum = 9.0
        }
    )

    run3_data = pa.table(
        {
            "price": [20.0, 30.0, 40.0],  # avg = 30.0
            "quantity": [3.0, 4.0, 5.0],  # sum = 12.0
        }
    )

    return run1_data, run2_data, run3_data


def _run_suite_executions(
    test_check: Any, db: InMemoryMetricDB, key: ResultKey, datasets: Tuple[pa.Table, pa.Table, pa.Table]
) -> Tuple[List[str], List[VerificationSuite]]:
    """Run 3 suite executions and return execution IDs and all suites."""
    execution_ids = []
    suites = []
    run1_data, run2_data, run3_data = datasets

    # Run 1
    ds1 = DuckRelationDataSource.from_arrow(run1_data, "test_data")
    suite1 = VerificationSuite([test_check], db, "Test Suite")
    suite1.run([ds1], key)
    execution_ids.append(suite1.execution_id)
    suites.append(suite1)

    time.sleep(0.01)  # Small delay to ensure different timestamps

    # Run 2
    ds2 = DuckRelationDataSource.from_arrow(run2_data, "test_data")
    suite2 = VerificationSuite([test_check], db, "Test Suite")
    suite2.run([ds2], key)
    execution_ids.append(suite2.execution_id)
    suites.append(suite2)

    time.sleep(0.01)

    # Run 3 (latest)
    ds3 = DuckRelationDataSource.from_arrow(run3_data, "test_data")
    suite3 = VerificationSuite([test_check], db, "Test Suite")
    suite3.run([ds3], key)
    execution_ids.append(suite3.execution_id)
    suites.append(suite3)

    return execution_ids, suites


def _get_metric_trace_values(suite: VerificationSuite, db: InMemoryMetricDB) -> Dict[str, float]:
    """Extract metric values from trace."""
    trace = suite.metric_trace(db)
    trace_dict = trace.to_pydict()

    values = {}
    for i in range(len(trace_dict["metric"])):
        metric_name = trace_dict["metric"][i]
        value_db = trace_dict["value_db"][i]

        # Match the actual lowercase metric names
        if metric_name == "average(price)":
            values["avg_price"] = value_db
        elif metric_name == "sum(quantity)":
            values["sum_quantity"] = value_db

    # Ensure we found both metrics
    if "avg_price" not in values or "sum_quantity" not in values:
        print(f"Available metrics in trace: {trace_dict['metric']}")
        if "avg_price" not in values:
            values["avg_price"] = None
        if "sum_quantity" not in values:
            values["sum_quantity"] = None

    return values


def _retrieve_metric_values(
    db: InMemoryMetricDB, avg_spec: specs.Average, sum_spec: specs.Sum, key: ResultKey
) -> Tuple[float, float]:
    """Retrieve metric values using get_metric_value."""
    avg_value = db.get_metric_value(avg_spec, key, "test_data")
    sum_value = db.get_metric_value(sum_spec, key, "test_data")

    assert isinstance(avg_value, Some), "Average metric not found"
    assert isinstance(sum_value, Some), "Sum metric not found"

    return avg_value.unwrap(), sum_value.unwrap()


def _retrieve_metric_windows(
    db: InMemoryMetricDB, avg_spec: specs.Average, sum_spec: specs.Sum, key: ResultKey
) -> Tuple[float, float]:
    """Retrieve metric values using get_metric_window."""
    avg_window = db.get_metric_window(avg_spec, key, lag=0, window=1, dataset="test_data")
    sum_window = db.get_metric_window(sum_spec, key, lag=0, window=1, dataset="test_data")

    assert isinstance(avg_window, Some), "Average window not found"
    assert isinstance(sum_window, Some), "Sum window not found"

    avg_ts = avg_window.unwrap()
    sum_ts = sum_window.unwrap()

    assert len(avg_ts) == 1, f"Expected 1 entry in window, got {len(avg_ts)}"
    assert len(sum_ts) == 1, f"Expected 1 entry in window, got {len(sum_ts)}"
    assert key.yyyy_mm_dd in avg_ts, "Date not found in average window"
    assert key.yyyy_mm_dd in sum_ts, "Date not found in sum window"

    return avg_ts[key.yyyy_mm_dd], sum_ts[key.yyyy_mm_dd]


def _get_execution_metrics(db: InMemoryMetricDB, execution_id: str) -> List[Metric]:
    """Get all metrics for a specific execution ID."""
    return list(data.metrics_by_execution_id(db, execution_id))


def _query_all_metrics_directly(db: InMemoryMetricDB, key: ResultKey) -> List[DBMetric]:
    """Query all average metrics directly from DB to show ordering issue."""
    session = db.new_session()
    return (
        session.query(DBMetric)
        .filter(
            DBMetric.metric_type == "Average",
            DBMetric.yyyy_mm_dd == key.yyyy_mm_dd,
            DBMetric.tags == key.tags,
            DBMetric.dataset == "test_data",
        )
        .all()
    )


def test_multiple_execution_metric_retrieval() -> None:
    """Test metric retrieval after multiple suite executions with different execution IDs."""

    # Create a simple check with 2 metrics
    @check(name="Test Metrics Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        avg_price = mp.average("price")
        sum_quantity = mp.sum("quantity")
        ctx.assert_that(avg_price).where(name="Average price check").is_positive()
        ctx.assert_that(sum_quantity).where(name="Sum quantity check").is_positive()

    # Setup
    db = InMemoryMetricDB()
    key = ResultKey(date(2024, 1, 15), {"env": "test", "version": "v1"})
    datasets = _create_test_data()

    # Run all executions
    execution_ids, suites = _run_suite_executions(test_check, db, key, datasets)

    # Assertion 1: Verify all execution IDs are unique
    assert len(set(execution_ids)) == 3, "All execution IDs should be unique"

    # Display metric trace after each run
    from dqx.display import print_metric_trace

    print("\n=== Metric Trace After Run 1 ===")
    trace1 = suites[0].metric_trace(db)
    print_metric_trace(trace1, execution_ids[0])

    print("\n=== Metric Trace After Run 2 ===")
    trace2 = suites[1].metric_trace(db)
    print_metric_trace(trace2, execution_ids[1])

    print("\n=== Metric Trace After Run 3 ===")
    trace3 = suites[2].metric_trace(db)
    print_metric_trace(trace3, execution_ids[2])

    # Get trace values (should be from latest execution)
    print("\n=== Testing metric_trace ===")
    trace_values = _get_metric_trace_values(suites[2], db)

    # Assertion 2: Metric trace should contain only latest values
    assert trace_values["avg_price"] == 30.0, (
        f"Trace Average(price) should be 30.0 (latest), got {trace_values['avg_price']}"
    )
    assert trace_values["sum_quantity"] == 12.0, (
        f"Trace Sum(quantity) should be 12.0 (latest), got {trace_values['sum_quantity']}"
    )
    print("✓ metric_trace correctly returns latest values")

    # Get metric specs
    avg_spec = specs.Average("price")
    sum_spec = specs.Sum("quantity")

    # Test get_metric_value retrieval
    print("\n=== Testing get_metric_value ===")
    avg_retrieved, sum_retrieved = _retrieve_metric_values(db, avg_spec, sum_spec, key)
    print(f"get_metric_value - Average(price): {avg_retrieved} (expected 30.0)")
    print(f"get_metric_value - Sum(quantity): {sum_retrieved} (expected 12.0)")

    # Assertion 3: These should fail, demonstrating the bug
    assert avg_retrieved == 30.0, f"get_metric_value should return latest average 30.0, got {avg_retrieved}"
    assert sum_retrieved == 12.0, f"get_metric_value should return latest sum 12.0, got {sum_retrieved}"

    # Test get_metric_window retrieval
    print("\n=== Testing get_metric_window ===")
    avg_window_value, sum_window_value = _retrieve_metric_windows(db, avg_spec, sum_spec, key)
    print(f"get_metric_window - Average(price): {avg_window_value} (expected 30.0)")
    print(f"get_metric_window - Sum(quantity): {sum_window_value} (expected 12.0)")

    # Assertion 4: Window retrieval should also show latest values
    assert avg_window_value == 30.0, f"get_metric_window should return latest average 30.0, got {avg_window_value}"
    assert sum_window_value == 12.0, f"get_metric_window should return latest sum 12.0, got {sum_window_value}"

    # Verify metadata storage
    print("\n=== Verifying metadata storage ===")
    run3_metrics = _get_execution_metrics(db, execution_ids[2])

    # Assertion 5: Metadata should be properly stored
    assert len(run3_metrics) >= 2, f"Expected at least 2 metrics for run 3, got {len(run3_metrics)}"
    for metric in run3_metrics:
        assert metric.metadata is not None, "Metric should have metadata"
        assert metric.metadata.execution_id == execution_ids[2], "Metric should have correct execution_id"
    print(f"✓ All {len(run3_metrics)} metrics from run 3 have correct execution_id")

    # Demonstrate the issue with direct query
    print("\n=== Demonstrating the issue ===")
    all_avg_metrics = _query_all_metrics_directly(db, key)

    # Assertion 6: Verify all 3 metrics exist in DB
    assert len(all_avg_metrics) == 3, f"Should have 3 Average metrics in DB, got {len(all_avg_metrics)}"
    metric_values = sorted(m.value for m in all_avg_metrics)
    assert metric_values == [20.0, 25.0, 30.0], f"Should have all 3 metric values, got {metric_values}"

    print(f"\nFound {len(all_avg_metrics)} Average(price) metrics in DB:")
    for i, db_metric in enumerate(sorted(all_avg_metrics, key=lambda m: m.value)):
        exec_id = "None"
        if db_metric.meta and db_metric.meta.execution_id:
            exec_id = db_metric.meta.execution_id[:8] + "..."
        print(f"  Run {i + 1}: value={db_metric.value}, created={db_metric.created}, execution_id={exec_id}")
