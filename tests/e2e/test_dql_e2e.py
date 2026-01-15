"""End-to-end tests for DQL suite execution.

This module contains e2e tests that use DQL files instead of Python @check decorators,
demonstrating the DQL interpreter's capabilities in realistic scenarios.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from dqx.dql import Interpreter
from dqx.orm.repositories import InMemoryMetricDB
from tests.fixtures.data_fixtures import CommercialDataSource


def test_e2e_suite_from_dql() -> None:
    """Test e2e suite execution from DQL file.

    This test achieves feature parity with test_e2e_suite from test_api_e2e.py,
    using a DQL file instead of Python @check decorators. It demonstrates:
    - Loading and executing a complete DQL suite
    - Multiple datasets with different date ranges
    - Complex metric expressions and extensions (day_over_day, week_over_week, stddev)
    - Nested extensions with offset and n parameters (stddev of day_over_day)
    - Cross-dataset assertions
    - Manual day-over-day calculations with lag parameter
    - List arguments (duplicate_count) and string literals (count_values)
    - Tags, tolerance, and severity levels
    - Collect statements (equivalent to Python's .noop())
    - Result collection and display

    Note: custom_sql is not supported in DQL. The Python test has 16 total results
    (15 validating assertions + 1 noop). This DQL test has 15 total results
    (14 validating assertions + 1 collect statement).
    """
    db = InMemoryMetricDB()

    # Define date ranges for the two datasources
    # ds1: Full month of January 2025
    ds1_start_date = dt.date(2025, 1, 1)
    ds1_end_date = dt.date(2025, 1, 31)

    # ds2: Slightly different range - starts earlier, ends on same day
    # This allows testing scenarios where historical data availability differs
    ds2_start_date = dt.date(2024, 12, 15)  # Starts mid-December 2024
    ds2_end_date = dt.date(2025, 1, 31)

    # Create the datasources with their respective date ranges
    ds1 = CommercialDataSource(
        start_date=ds1_start_date,
        end_date=ds1_end_date,
        name="ds1",
        records_per_day=30,
        seed=1050,  # Same seed as original commerce_data_c1
        skip_dates={dt.date.fromisoformat("2025-01-13")},
    )

    ds2 = CommercialDataSource(
        start_date=ds2_start_date,
        end_date=ds2_end_date,
        name="ds2",
        records_per_day=35,
        seed=2100,  # Same seed as original commerce_data_c2
        skip_dates={dt.date.fromisoformat("2025-01-14")},
    )

    # Prepare datasources mapping
    datasources = {"ds1": ds1, "ds2": ds2}

    # Execution date
    execution_date = dt.date.fromisoformat("2025-01-15")

    # Tags for result key - use dict to match API test behavior
    tags = {"env": "prod", "partner": "gha"}

    # Create interpreter and run DQL suite
    interp = Interpreter(db=db)
    dql_file = Path(__file__).parent.parent / "dql" / "commerce_suite.dql"
    results = interp.run(dql_file, datasources, execution_date, tags=tags)

    # Validate result count matches Python test expectations
    # Python test: 16 results (15 assertions + 1 noop)
    # DQL test: 15 results (14 assertions + 1 collect, minus 1 custom_sql not supported)
    assert len(results.assertions) == 15, (
        f"Expected 15 results (14 validating assertions + 1 collect statement), got {len(results.assertions)}"
    )

    # Display summary
    print(f"\n{'=' * 80}")
    print(f"Suite: {results.suite_name}")
    print(f"Execution Date: {results.execution_date}")
    print(f"{'=' * 80}")
    print(f"Total assertions: {len(results.assertions)}")
    print(f"Passed: {len(results.passes)}")
    print(f"Failed: {len(results.failures)}")
    print(f"\nAll passed: {results.all_passed()}")

    # Print detailed assertion results
    print(f"\n{'=' * 80}")
    print("Assertion Results:")
    print(f"{'=' * 80}")

    for assertion in results.assertions:
        status_symbol = "✓" if assertion.passed else "✗"
        print(f"\n{status_symbol} [{assertion.severity}] {assertion.check_name}: {assertion.assertion_name}")
        print(f"  Condition: {assertion.condition}")

        if assertion.metric_value is not None:
            print(f"  Metric Value: {assertion.metric_value:.6f}")
        if assertion.threshold is not None:
            print(f"  Threshold: {assertion.threshold:.6f}")

        if not assertion.passed and assertion.reason:
            print(f"  Reason: {assertion.reason}")

    # Print summary by check
    print(f"\n{'=' * 80}")
    print("Summary by Check:")
    print(f"{'=' * 80}")

    checks = {}
    for assertion in results.assertions:
        if assertion.check_name not in checks:
            checks[assertion.check_name] = {"passed": 0, "failed": 0}
        if assertion.passed:
            checks[assertion.check_name]["passed"] += 1
        else:
            checks[assertion.check_name]["failed"] += 1

    for check_name, counts in checks.items():
        total = counts["passed"] + counts["failed"]
        status = "✓" if counts["failed"] == 0 else "✗"
        print(f"{status} {check_name}: {counts['passed']}/{total} passed")

    print(f"\n{'=' * 80}\n")
