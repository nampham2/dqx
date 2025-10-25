#!/usr/bin/env python3
"""
Demonstration of VerificationSuite caching and is_critical features.

This example shows:
1. How collect_results() and collect_symbols() cache their results
2. How is_critical() detects P0 failures
3. Performance benefits of caching
"""

import time
from datetime import date

import pyarrow as pa

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def main() -> None:
    """Run the caching and critical level demonstration."""
    print("=== DQX Suite Caching and Critical Level Demo ===\n")

    # Create database
    db = InMemoryMetricDB()

    # Demo 1: Caching behavior
    print("1. Demonstrating Result Caching")
    print("-" * 40)
    demo_caching(db)

    print("\n2. Demonstrating is_critical() Method")
    print("-" * 40)
    demo_critical_level(db)

    print("\n3. Combined Example: Critical Alert System")
    print("-" * 40)
    demo_alert_system(db)


def demo_caching(db: InMemoryMetricDB) -> None:
    """Demonstrate caching behavior of collect_results and collect_symbols."""

    @check(name="Performance Check")
    def performance_check(mp: MetricProvider, ctx: Context) -> None:
        avg_response = mp.average("response_time_ms")
        total_requests = mp.sum("request_count")

        ctx.assert_that(avg_response).where(name="Average response time < 100ms").is_lt(100)

        ctx.assert_that(total_requests).where(name="Total requests > 1000").is_gt(1000)

    # Create suite and data
    suite = VerificationSuite([performance_check], db, "Performance Suite")
    data = pa.table(
        {
            "response_time_ms": [45, 67, 89, 123, 78, 92, 55, 88, 91, 77],
            "request_count": [150, 200, 175, 225, 180, 195, 160, 210, 190, 205],
        }
    )
    ds = DuckRelationDataSource.from_arrow(data)
    key = ResultKey(date.today(), {"env": "prod", "region": "us-east-1"})

    # Run the suite
    suite.run({"metrics": ds}, key)

    # First call - computes results
    start = time.time()
    results1 = suite.collect_results()
    time1 = time.time() - start

    # Second call - returns cached results
    start = time.time()
    results2 = suite.collect_results()
    time2 = time.time() - start

    # Verify caching
    print(f"First collect_results() call: {time1 * 1000:.3f}ms")
    print(f"Second collect_results() call: {time2 * 1000:.3f}ms")
    if time2 > 0:
        print(f"Speed improvement: {time1 / time2:.1f}x faster")
    else:
        print("Speed improvement: Second call was instantaneous (cached)")
    print(f"Same object reference: {results1 is results2}")

    # Same for symbols
    start = time.time()
    symbols1 = suite.provider.collect_symbols(key)
    time1 = time.time() - start

    start = time.time()
    symbols2 = suite.provider.collect_symbols(key)
    time2 = time.time() - start

    print(f"\nFirst collect_symbols() call: {time1 * 1000:.3f}ms")
    print(f"Second collect_symbols() call: {time2 * 1000:.3f}ms")
    if time2 > 0:
        print(f"Speed improvement: {time1 / time2:.1f}x faster")
    else:
        print("Speed improvement: Second call was instantaneous (cached)")
    print(f"Same object reference: {symbols1 is symbols2}")


def demo_critical_level(db: InMemoryMetricDB) -> None:
    """Demonstrate is_critical() method with different severity levels."""

    @check(name="Data Quality Check")
    def quality_check(mp: MetricProvider, ctx: Context) -> None:
        # Calculate completeness as (total_rows - null_count) / total_rows
        total_rows = mp.num_rows()
        null_count = mp.null_count("user_id")
        completeness = (total_rows - null_count) / total_rows

        duplicate_rate = mp.duplicate_count(["user_id"]) / mp.num_rows()
        freshness = mp.maximum("days_since_update")

        # P0 - Critical assertion
        ctx.assert_that(completeness).where(name="User ID completeness must be 100%", severity="P0").is_eq(1.0)

        # P1 - Important but not critical
        ctx.assert_that(duplicate_rate).where(name="Duplicate rate should be < 5%", severity="P1").is_lt(0.05)

        # P2 - Nice to have
        ctx.assert_that(freshness).where(name="Data should be updated within 7 days", severity="P2").is_leq(7)

    # Test Case 1: All assertions pass
    print("\nTest Case 1: All assertions pass")
    suite1 = VerificationSuite([quality_check], db, "Quality Suite")
    data1 = pa.table({"user_id": ["u1", "u2", "u3", "u4", "u5"], "days_since_update": [1, 2, 1, 3, 2]})
    ds1 = DuckRelationDataSource.from_arrow(data1)
    key1 = ResultKey(date.today(), {"dataset": "clean"})

    suite1.run({"users": ds1}, key1)
    print(f"Is critical: {suite1.is_critical()}")

    # Test Case 2: P0 assertion fails
    print("\nTest Case 2: P0 assertion fails (missing user_id)")
    suite2 = VerificationSuite([quality_check], db, "Quality Suite")
    data2 = pa.table(
        {
            "user_id": ["u1", "u2", None, "u4", "u5"],  # Missing value!
            "days_since_update": [1, 2, 1, 3, 2],
        }
    )
    ds2 = DuckRelationDataSource.from_arrow(data2)
    key2 = ResultKey(date.today(), {"dataset": "corrupted"})

    suite2.run({"users": ds2}, key2)
    print(f"Is critical: {suite2.is_critical()}")

    # Show which assertions failed
    for result in suite2.collect_results():
        if result.status == "FAILURE":
            print(f"  - {result.assertion} ({result.severity}): FAILED")


def demo_alert_system(db: InMemoryMetricDB) -> None:
    """Demonstrate a realistic alert system using is_critical()."""

    @check(name="Revenue Pipeline Monitor")
    def revenue_monitor(mp: MetricProvider, ctx: Context) -> None:
        # Critical metrics
        transaction_count = mp.num_rows()
        revenue_total = mp.sum("amount")
        fraud_rate = mp.sum("is_fraud") / mp.num_rows()

        # P0: Must have transactions
        ctx.assert_that(transaction_count).where(name="Transaction volume check", severity="P0").is_gt(0)

        # P0: Revenue must be positive
        ctx.assert_that(revenue_total).where(name="Revenue sanity check", severity="P0").is_positive()

        # P0: Fraud rate must be below 10%
        ctx.assert_that(fraud_rate).where(name="Fraud rate threshold", severity="P0").is_lt(0.10)

        # P1: Performance metrics
        avg_processing_time = mp.average("processing_time_ms")
        ctx.assert_that(avg_processing_time).where(name="Processing time SLA", severity="P1").is_lt(500)

    # Simulate different scenarios
    scenarios = [
        {
            "name": "Normal Operations",
            "data": pa.table(
                {
                    "amount": [100, 200, 150, 300, 250],
                    "is_fraud": [0, 0, 0, 0, 1],  # 20% fraud rate - will fail!
                    "processing_time_ms": [100, 200, 150, 180, 220],
                }
            ),
            "tags": {"hour": "2024-01-15T14:00", "system": "payments"},
        },
        {
            "name": "System Outage",
            "data": pa.table(
                {
                    "amount": [],  # No data!
                    "is_fraud": [],
                    "processing_time_ms": [],
                }
            ),
            "tags": {"hour": "2024-01-15T15:00", "system": "payments"},
        },
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Tags: {scenario['tags']}")

        suite = VerificationSuite([revenue_monitor], db, "Revenue Monitor")
        ds = DuckRelationDataSource.from_arrow(scenario["data"])
        key = ResultKey(date.today(), scenario["tags"])

        suite.run({"transactions": ds}, key)

        # Check if critical and trigger alerts
        if suite.is_critical():
            print("🚨 CRITICAL ALERT! P0 failures detected:")

            # Use cached results for analysis
            failures = [r for r in suite.collect_results() if r.severity == "P0" and r.status == "FAILURE"]

            for failure in failures:
                print(f"   ❌ {failure.assertion}")

            print("\n   Alert sent to: on-call@company.com")
            print(f"   Incident created: INC-{hash(str(failures)) % 10000:04d}")
        else:
            print("✅ All critical checks passed")

        # Show summary using cached results
        results = suite.collect_results()
        total = len(results)
        passed = sum(1 for r in results if r.status == "OK")
        print(f"Summary: {passed}/{total} assertions passed")


if __name__ == "__main__":
    main()
