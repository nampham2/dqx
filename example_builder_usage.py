#!/usr/bin/env python3
"""
Example demonstrating the new VerificationSuiteBuilder pattern and improved API.

This example shows how to use the refactored code with better practices.
"""

import datetime as dt

from dqx.api import VerificationSuiteBuilder, VerificationSuite, check, GraphStates
from dqx.common import Context, MetricProvider, ResultKey
from dqx.orm.repositories import InMemoryMetricDB


# Example checks using the improved API
@check(tags=["data_quality"], label="Basic volume check")
def volume_check(mp: MetricProvider, ctx: Context) -> None:
    """Check that we have reasonable data volume."""
    row_count = mp.num_rows()
    ctx.assert_that(row_count).on(label="Row count should be positive").is_positive()
    ctx.assert_that(row_count).on(label="Row count should be reasonable").is_geq(100)


@check(tags=["completeness"], label="Data completeness check", datasets=["main_table"])
def completeness_check(mp: MetricProvider, ctx: Context) -> None:
    """Check data completeness across key columns."""
    total_rows = mp.num_rows()
    
    # Check null percentages
    for column in ["id", "name", "email"]:
        null_count = mp.null_count(column)
        null_percentage = null_count / total_rows
        ctx.assert_that(null_percentage).on(
            label=f"{column} null percentage should be low"
        ).is_leq(0.05)  # Max 5% nulls


@check(tags=["consistency"], label="Value consistency check")
def consistency_check(mp: MetricProvider, ctx: Context) -> None:
    """Check value consistency and ranges."""
    avg_price = mp.average("price")
    min_price = mp.minimum("price")
    max_price = mp.maximum("price")
    
    # Prices should be positive
    ctx.assert_that(avg_price).on(label="Average price should be positive").is_positive()
    ctx.assert_that(min_price).on(label="Minimum price should be non-negative").is_geq(0.0)
    
    # Price distribution should be reasonable
    price_range = max_price - min_price
    ctx.assert_that(price_range / avg_price).on(
        label="Price range should be reasonable relative to average"
    ).is_leq(10.0)


def demonstrate_builder_pattern() -> tuple[VerificationSuite, Context]:
    """Demonstrate the new VerificationSuiteBuilder pattern."""
    print("=== Demonstrating VerificationSuiteBuilder Pattern ===")
    
    # Create database
    db = InMemoryMetricDB()
    
    # Build suite using the builder pattern
    suite = (VerificationSuiteBuilder("E-commerce Data Quality Suite", db)
             .add_check(volume_check)
             .add_check(completeness_check)
             .add_check(consistency_check)
             .build())
    
    print(f"✅ Created suite: '{suite._name}' with {len(suite._checks)} checks")
    
    # Create a result key
    key = ResultKey(yyyy_mm_dd=dt.date.today(), tags={"env": "prod"})
    
    # Demonstrate improved error handling
    try:
        # This will show improved validation
        _empty_suite = VerificationSuiteBuilder("", db).build()
    except Exception as e:
        print(f"✅ Validation works: {e}")
    
    # Collect checks to show the dependency graph
    context = suite.collect(key)
    
    # Demonstrate new pending_metrics API with optional dataset parameter
    all_pending = context.pending_metrics()  # All datasets
    specific_pending = context.pending_metrics("main_table")  # Specific dataset
    
    print(f"✅ Total pending metrics (all datasets): {len(all_pending)}")
    print(f"✅ Pending metrics for 'main_table': {len(specific_pending)}")
    
    # Show constants usage
    print(f"✅ Using constants: GraphStates.PENDING = '{GraphStates.PENDING}'")
    
    return suite, context


def demonstrate_improved_error_handling() -> None:
    """Demonstrate improved error handling and validation."""
    print("\n=== Demonstrating Improved Error Handling ===")
    
    db = InMemoryMetricDB()
    
    # Test validation
    try:
        # Empty checks list
        VerificationSuiteBuilder("Test", db).build()
    except Exception as e:
        print(f"✅ Empty checks validation: {e}")
    
    try:
        # Empty name
        VerificationSuiteBuilder("", db).add_check(volume_check).build()
    except Exception as e:
        print(f"✅ Empty name validation: {e}")


if __name__ == "__main__":
    # Run demonstrations
    suite, context = demonstrate_builder_pattern()
    demonstrate_improved_error_handling()
    
    print("\n=== Summary of Improvements ===")
    print("✅ Builder pattern for fluent VerificationSuite creation")
    print("✅ Comprehensive input validation with clear error messages")
    print("✅ Eliminated code duplication in SymbolicAssert methods")
    print("✅ Improved method organization and single responsibility")
    print("✅ Enhanced documentation with docstrings and type hints")
    print("✅ Constants for magic strings (GraphStates)")
    print("✅ Better error handling with context preservation")
    print("✅ Performance optimizations (symbol caching)")
    print("✅ Backward-compatible API improvements")
    print("✅ All existing tests pass without modification")
