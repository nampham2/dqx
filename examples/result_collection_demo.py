"""Demo of collecting and using assertion results with v3 simplifications."""

import datetime

import duckdb
import pyarrow as pa
import sympy as sp
from returns.result import Failure

from dqx.api import Context, MetricProvider, VerificationSuiteBuilder, check
from dqx.common import ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB

# Create sample e-commerce data
orders_data = pa.table(
    {
        "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "amount": [50.0, 150.0, 75.0, 200.0, 30.0, 120.0, 90.0, 180.0, 60.0, 250.0],
        "status": [
            "completed",
            "completed",
            "pending",
            "completed",
            "cancelled",
            "completed",
            "completed",
            "pending",
            "completed",
            "completed",
        ],
        "returns": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros for divide by zero demo
        "temperature": [-10.0, -5.0, -15.0, -8.0, -20.0, -12.0, -18.0, -6.0, -25.0, -30.0],  # Negative for sqrt demo
    }
)


def print_results_table(results: list) -> None:
    """Print results in a nice tabular format."""
    # Calculate column widths
    check_width = max(len(r.check) for r in results) + 2
    assertion_width = max(len(r.assertion) for r in results) + 2
    expression_width = 40  # For expression column

    # Print header
    print("\n" + "=" * 120)
    print(
        f"{'Check':<{check_width}} | {'Assertion':<{assertion_width}} | {'Sev':<5} | {'Status':<8} | {'Expression':<{expression_width}} | {'Value/Error'}"
    )
    print("=" * 120)

    # Print rows
    for result in results:
        status_icon = "✅" if result.status == "OK" else "❌"
        status_text = f"{status_icon} {result.status}"

        # Truncate expression if too long
        expr = result.expression if result.expression else "N/A"
        if len(expr) > expression_width - 2:
            expr = expr[: expression_width - 5] + "..."

        if result.status == "OK":
            value_text = f"{result.metric.unwrap():.2f}"
        else:
            failures = result.metric.failure()
            if failures:
                # Show the first error message
                value_text = failures[0].error_message
                # For long messages, show key part
                if "infinity" in value_text.lower():
                    value_text = "Result is infinity"
                elif "complex" in value_text.lower():
                    value_text = "Result is complex number"
                elif "timeout" in value_text.lower():
                    value_text = "Database connection timeout"
                elif "permission denied" in value_text.lower():
                    value_text = "Permission denied"
                elif len(value_text) > 40:
                    value_text = value_text[:37] + "..."
            else:
                value_text = "Unknown error"

        print(
            f"{result.check:<{check_width}} | {result.assertion:<{assertion_width}} | {result.severity:<5} | {status_text:<8} | {expr:<{expression_width}} | {value_text}"
        )

    print("=" * 120)


# Define data quality checks including scenarios that trigger metric calculation failures
@check(name="Basic Validations", datasets=["orders"])
def basic_validations(mp: MetricProvider, ctx: Context) -> None:
    """Basic metric validations that should succeed."""
    # These will evaluate successfully
    ctx.assert_that(mp.num_rows()).where(name="Has data", severity="P0").is_gt(0)
    ctx.assert_that(mp.average("amount")).where(name="Average order value", severity="P1").is_positive()


@check(name="Calculation Failures", datasets=["orders"])
def calculation_failures(mp: MetricProvider, ctx: Context) -> None:
    """Calculations that will fail during evaluation."""
    # Create a complex expression that evaluates to NaN
    # We'll use variance of a constant column (returns) which is 0
    variance_returns = mp.variance("returns")  # This will be 0
    normalized_variance = variance_returns / variance_returns  # 0/0 = NaN

    ctx.assert_that(normalized_variance).where(name="Normalized variance (NaN)", severity="P0").is_positive()

    # Another approach: use sympy to create invalid operations
    zero_sum = mp.sum("returns")  # This is 0
    reciprocal = sp.Integer(1) / zero_sum  # 1/0 = infinity
    ctx.assert_that(reciprocal).where(name="Reciprocal of zero sum", severity="P1").is_lt(100)


@check(name="Division Operations", datasets=["orders"])
def division_operations(mp: MetricProvider, ctx: Context) -> None:
    """Division operations that might fail."""
    # Calculate return rate: sum(returns) / num_rows()
    # This will fail because sum(returns) = 0, so 0/10 = 0, then checking if 0 > 0.5 fails
    total_returns = mp.sum("returns")
    total_orders = mp.num_rows()
    return_rate = total_returns / total_orders

    ctx.assert_that(return_rate).where(name="Return rate calculation", severity="P1").is_positive()

    # This creates infinity: amount / returns where returns = 0
    amount_per_return = mp.average("amount") / mp.sum("returns")  # This will be infinity!
    ctx.assert_that(amount_per_return).where(name="Amount per return (infinity)", severity="P0").is_lt(1000)


@check(name="Complex Number Operations", datasets=["orders"])
def complex_operations(mp: MetricProvider, ctx: Context) -> None:
    """Operations that produce complex numbers."""
    # Square root of negative temperature average (will fail with complex number)
    avg_temp = mp.average("temperature")  # This is negative
    sqrt_temp = sp.sqrt(avg_temp)
    ctx.assert_that(sqrt_temp).where(name="Square root of temperature", severity="P0").is_positive()

    # Log of negative value
    log_temp = sp.log(mp.minimum("temperature"))
    ctx.assert_that(log_temp).where(name="Log of minimum temperature", severity="P1").is_positive()


@check(name="Database Failures", datasets=["orders"])
def database_failures(mp: MetricProvider, ctx: Context) -> None:
    """Simulate database connection failures."""
    # We'll mock these to fail after the suite is built
    ctx.assert_that(mp.sum("amount")).where(name="Revenue total (DB timeout)", severity="P0").is_positive()
    ctx.assert_that(mp.approx_cardinality("customer_id")).where(
        name="Customer count (permission denied)", severity="P1"
    ).is_positive()


def main() -> None:
    """Run the demo."""
    # Setup
    db = InMemoryMetricDB()
    suite = (
        VerificationSuiteBuilder("E-commerce Data Quality with Failures", db)
        .add_check(basic_validations)
        .add_check(calculation_failures)
        .add_check(division_operations)
        .add_check(complex_operations)
        .add_check(database_failures)
        .build()
    )

    # Mock some metrics to fail with database errors
    # We need to do this after building the suite but before running
    provider = suite.provider

    # Find the symbols for database failure simulation
    for symbol in provider.symbols():
        metric = provider.get_symbol(symbol)
        # Mock database timeout for revenue total
        if metric.metric_spec.name == "sum(amount)" and "Database Failures" in str(metric):
            provider._symbol_index[symbol].fn = lambda k: Failure("Database connection timeout after 30s")
        # Mock permission denied for customer count
        elif metric.metric_spec.name == "approx_cardinality(customer_id)" and "Database Failures" in str(metric):
            provider._symbol_index[symbol].fn = lambda k: Failure(
                "Permission denied: user lacks SELECT privilege on table 'orders'"
            )

    # Create data sources
    orders_datasource = ArrowDataSource(orders_data)

    # Run validation
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={"environment": "production", "version": "1.0"})

    print("Running data quality checks with intentional failures...")
    suite.run({"orders": orders_datasource}, key)

    # Check evaluation status
    print(f"Suite evaluated: {suite.is_evaluated}")

    # Collect results - no key needed anymore!
    results = suite.collect_results()

    print(f"\nCollected {len(results)} assertion results")

    # Display results in tabular format
    print_results_table(results)

    # Summary statistics
    print("\nSummary by Status:")
    success_count = sum(1 for r in results if r.status == "OK")
    failure_count = sum(1 for r in results if r.status == "FAILURE")
    print(f"✅ OK:      {success_count}")
    print(f"❌ FAILURE: {failure_count}")
    print(f"📊 Total:   {len(results)}")

    # Summary by severity
    print("\nSummary by Severity:")
    for severity in ["P0", "P1", "P2", "P3"]:
        sev_results = [r for r in results if r.severity == severity]
        if sev_results:
            passed = sum(1 for r in sev_results if r.status == "OK")
            failed = sum(1 for r in sev_results if r.status == "FAILURE")
            print(f"  {severity}: {passed} passed, {failed} failed")

    print("\nFailed Assertions (detailed):")
    failed_results = [r for r in results if r.status == "FAILURE"]
    if failed_results:
        for r in failed_results:
            print(f"\n  ❌ {r.check} / {r.assertion} [{r.severity}]")
            print(f"     Expression: {r.expression}")
            failures = r.metric.failure()
            for failure in failures:
                print(f"     Error: {failure.error_message}")
                if failure.symbols:
                    print(f"     Expression: {failure.expression}")
                    for symbol in failure.symbols:
                        print(f"       - {symbol.name}: {symbol.value}")
    else:
        print("  No failures!")

    # Demonstrate that suite cannot be run again
    print("\nAttempting to run suite again...")
    try:
        suite.run({"orders": orders_datasource}, key)
    except Exception as e:
        print(f"Expected error: {e}")

    # Example: Create DuckDB relation from results
    print("\nCreating DuckDB table from results...")
    conn = duckdb.connect()

    # Convert results to a list of dictionaries
    results_data = [
        {
            "date": r.yyyy_mm_dd,
            "suite": r.suite,
            "check": r.check,
            "assertion": r.assertion,
            "severity": r.severity,
            "status": r.status,
            "value": r.metric.unwrap() if r.status == "OK" else None,
        }
        for r in results
    ]

    # Create DuckDB table from the data
    conn.execute(
        """
        CREATE TABLE results_table AS
        SELECT * FROM (
            VALUES
            """
        + ",".join(
            [
                f"(DATE '{r['date']}', '{r['suite']}', '{r['check']}', '{r['assertion']}', "
                + f"'{r['severity']}', '{r['status']}', {r['value'] if r['value'] is not None else 'NULL'})"
                for r in results_data
            ]
        )
        + """) AS t(date, suite, "check", assertion, severity, status, value)
    """
    )

    print("\nAssertion counts by check:")
    summary = conn.execute("""
        SELECT "check", COUNT(*) as assertion_count
        FROM results_table
        GROUP BY "check"
        ORDER BY "check"
    """).fetchall()

    for check_name, count in summary:
        print(f"  {check_name}: {count} assertions")


if __name__ == "__main__":
    main()
