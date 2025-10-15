#!/usr/bin/env python3
"""Demo of the new is_between assertion functionality."""

import datetime

import sympy as sp

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.orm.repositories import InMemoryMetricDB


@check(name="Temperature Range Check")
def temperature_checks(mp: MetricProvider, ctx: Context) -> None:
    """Check that temperature values fall within expected ranges."""

    # Simulate temperature sensor reading
    temp = sp.Symbol("temperature")

    # Check if temperature is in normal operating range (20-25°C)
    ctx.assert_that(temp).where(name="Temperature in normal range").is_between(20.0, 25.0)

    # Check if temperature is exactly at a specific value (with tolerance)
    ctx.assert_that(temp).where(name="Temperature at optimal 22.5°C").is_between(22.5, 22.5)


@check(name="Data Quality Metrics")
def data_quality_checks(mp: MetricProvider, ctx: Context) -> None:
    """Check that data quality metrics are within acceptable bounds."""

    # Check null count is between 0 and 100
    ctx.assert_that(mp.null_count("user_id")).where(name="Null count within tolerance", severity="P1").is_between(
        0.0, 100.0
    )

    # Check row count is between expected bounds
    ctx.assert_that(mp.num_rows()).where(name="Row count in expected range", severity="P0").is_between(1000.0, 10000.0)

    # Check average price is between reasonable bounds
    ctx.assert_that(mp.average("price")).where(name="Average price in valid range", severity="P2").is_between(
        10.0, 100.0
    )


def main() -> None:
    """Run the demo."""
    print("=== DQX is_between Assertion Demo ===\n")

    # Set up the verification suite
    db = InMemoryMetricDB()
    suite = VerificationSuite(checks=[temperature_checks, data_quality_checks], db=db, name="is_between Demo Suite")

    # Build the dependency graph
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"env": "demo"})
    suite.build_graph(suite._context, key)

    # Display the graph structure
    print("Verification Suite Structure:")
    suite.graph.print_tree()

    print("\n✓ Successfully created assertions using is_between!")
    print("\nThe is_between assertion allows you to:")
    print("- Check if values fall within a specified range (inclusive)")
    print("- Validate exact values using equal lower and upper bounds")
    print("- Apply floating-point tolerance to both bounds")
    print("- Get clear error messages for invalid ranges")


if __name__ == "__main__":
    main()
