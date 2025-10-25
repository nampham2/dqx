#!/usr/bin/env python3
"""
Demonstration of table display functions for DQX results.

This script shows how to use print_assertion_results() and print_symbols()
to display verification suite results in formatted tables.
"""

from datetime import date

from returns.result import Failure, Success

from dqx.common import AssertionResult, EvaluationFailure
from dqx.display import print_assertion_results, print_symbols
from dqx.provider import SymbolInfo


def main() -> None:
    """Run the table display demonstration."""
    print("=== DQX Table Display Demo ===\n")

    # Create sample assertion results
    assertion_results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            check="Order Validation",
            assertion="Average order amount is positive",
            expression="average(amount) > 0",
            severity="P1",
            status="OK",
            metric=Success(125.50),
            tags={"env": "production", "region": "us-west"},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            check="Order Validation",
            assertion="No orders exceed $10,000",
            expression="max(amount) <= 10000",
            severity="P0",
            status="FAILURE",
            metric=Failure(
                [
                    EvaluationFailure(
                        error_message="Maximum amount is $15,000",
                        expression="max(amount)",
                        symbols=[],
                    )
                ]
            ),
            tags={"env": "production", "region": "us-west"},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            check="Customer Validation",
            assertion="All customers have email",
            expression="null_count(email) = 0",
            severity="P2",
            status="OK",
            metric=Success(0.0),  # Use 0.0 instead of None (more realistic)
            tags={"env": "production", "region": "us-west"},
        ),
    ]

    # Display assertion results
    print("\n1. Assertion Results Table:")
    print_assertion_results(assertion_results)

    # Create sample symbol values
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="average(amount)",
            dataset="orders",
            value=Success(125.50),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
        SymbolInfo(
            name="x_2",
            metric="max(amount)",
            dataset="orders",
            value=Success(15000.0),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
        SymbolInfo(
            name="x_3",
            metric="null_count(email)",
            dataset="customers",
            value=Success(0.0),  # Use 0.0 instead of None
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
        SymbolInfo(
            name="x_4",
            metric="count(*)",
            dataset="orders",
            value=Failure("Connection timeout"),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Daily Data Quality",
            tags={"env": "production", "region": "us-west"},
        ),
    ]

    # Display symbol values
    print("\n2. Symbol Values Table:")
    print_symbols(symbols)

    print("\n=== Demo Complete ===")
    print("\nNote: This demo shows sample data with realistic values.")
    print("Colors will appear when run in a terminal that supports Rich formatting.")


if __name__ == "__main__":
    main()
