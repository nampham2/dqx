"""Integration test for metric retrieval after multiple suite executions."""

from datetime import date
from typing import Any, List, Tuple

import pyarrow as pa
from returns.maybe import Some

from dqx import data, specs
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.orm.repositories import Metric as DBMetric


def _create_test_datasets_with_expected_values() -> List[Tuple[pa.Table, float, float]]:
    """Create test datasets with their expected metric values."""
    datasets = [
        (
            pa.table({"price": [10.0, 20.0, 30.0], "quantity": [1.0, 2.0, 3.0]}),
            20.0,  # avg price
            6.0,  # sum quantity
        ),
        (
            pa.table({"price": [15.0, 25.0, 35.0], "quantity": [2.0, 3.0, 4.0]}),
            25.0,  # avg price
            9.0,  # sum quantity
        ),
        (
            pa.table({"price": [20.0, 30.0, 40.0], "quantity": [3.0, 4.0, 5.0]}),
            30.0,  # avg price
            12.0,  # sum quantity
        ),
    ]
    return datasets


def _run_single_execution_with_trace(
    test_check: Any,
    db: InMemoryMetricDB,
    key: ResultKey,
    data: pa.Table,
    run_number: int,
    expected_avg: float,
    expected_sum: float,
) -> Tuple[str, VerificationSuite, pa.Table]:
    """Run a single suite execution and immediately collect its trace."""
    from dqx.display import print_metric_trace

    # Create datasource and suite
    ds = DuckRelationDataSource.from_arrow(data, "test_data")
    suite = VerificationSuite([test_check], db, "Test Suite")

    # Run suite
    suite.run([ds], key)
    execution_id = suite.execution_id

    # Get trace immediately after execution
    trace = suite.metric_trace(db)

    # Display trace
    print(f"\n=== Metric Trace After Run {run_number} ===")
    print_metric_trace(trace, execution_id)

    # Assert trace consistency
    _assert_trace_values_consistent(trace, run_number)

    # Verify expected values
    _verify_trace_has_expected_values(trace, expected_avg, expected_sum, run_number)

    return execution_id, suite, trace


def _verify_trace_has_expected_values(
    trace: pa.Table, expected_avg: float, expected_sum: float, run_number: int
) -> None:
    """Verify that the trace contains the expected values."""
    trace_dict = trace.to_pydict()

    for i in range(len(trace_dict["metric"])):
        metric = trace_dict["metric"][i]
        value_final = trace_dict["value_final"][i]

        if metric == "average(price)":
            assert value_final == expected_avg, (
                f"Run {run_number} - Expected average(price) = {expected_avg}, got {value_final}"
            )
        elif metric == "sum(quantity)":
            assert value_final == expected_sum, (
                f"Run {run_number} - Expected sum(quantity) = {expected_sum}, got {value_final}"
            )


def _assert_trace_values_consistent(trace: pa.Table, run_number: int, epsilon: float = 1e-9) -> None:
    """Assert that value_analysis == value_db == value_final for all rows."""
    trace_dict = trace.to_pydict()

    for i in range(len(trace_dict["metric"])):
        metric = trace_dict["metric"][i]
        value_analysis = trace_dict["value_analysis"][i]
        value_db = trace_dict["value_db"][i]
        value_final = trace_dict["value_final"][i]

        # Assert that none of the values are None
        assert value_analysis is not None, f"Run {run_number} - {metric}: value_analysis is None"
        assert value_db is not None, f"Run {run_number} - {metric}: value_db is None"
        assert value_final is not None, f"Run {run_number} - {metric}: value_final is None"

        # Assert value_analysis == value_db
        assert abs(value_analysis - value_db) < epsilon, (
            f"Run {run_number} - {metric}: value_analysis ({value_analysis}) != value_db ({value_db})"
        )

        # Assert value_db == value_final
        assert abs(value_db - value_final) < epsilon, (
            f"Run {run_number} - {metric}: value_db ({value_db}) != value_final ({value_final})"
        )

        # Assert value_analysis == value_final (for completeness)
        assert abs(value_analysis - value_final) < epsilon, (
            f"Run {run_number} - {metric}: value_analysis ({value_analysis}) != value_final ({value_final})"
        )


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
    """
    Test metric retrieval after multiple suite executions with different execution IDs.

    This test verifies that metric traces correctly show the values from their
    respective executions, not always the latest values.
    """

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

    # Get datasets with expected values
    datasets_with_expected = _create_test_datasets_with_expected_values()

    # Run executions and collect traces immediately
    execution_ids = []
    suites = []
    traces = []

    for i, (test_data, expected_avg, expected_sum) in enumerate(datasets_with_expected):
        run_number = i + 1
        execution_id, suite, trace = _run_single_execution_with_trace(
            test_check, db, key, test_data, run_number, expected_avg, expected_sum
        )
        execution_ids.append(execution_id)
        suites.append(suite)
        traces.append(trace)

    # Assertion 1: Verify all execution IDs are unique
    assert len(set(execution_ids)) == 3, "All execution IDs should be unique"
    print("\n✓ All execution IDs are unique")

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

    # Final verification: Show that trace collected later shows latest values
    print("\n=== Re-checking traces after all executions ===")
    print("If we collect traces now for old executions, they show latest values:")

    # Get trace for run 1 now (after all executions)
    trace1_later = suites[0].metric_trace(db)
    trace1_dict = trace1_later.to_pydict()
    for i in range(len(trace1_dict["metric"])):
        if trace1_dict["metric"][i] == "average(price)":
            print(f"Run 1 trace collected now: value_final = {trace1_dict['value_final'][i]} (should be 20.0)")

    # Get trace for run 2 now
    trace2_later = suites[1].metric_trace(db)
    trace2_dict = trace2_later.to_pydict()
    for i in range(len(trace2_dict["metric"])):
        if trace2_dict["metric"][i] == "average(price)":
            print(f"Run 2 trace collected now: value_final = {trace2_dict['value_final'][i]} (should be 25.0)")

    print("\nThis demonstrates the bug: traces must be collected immediately after execution")
    print("to show correct values. Later retrieval always shows the latest values.")
