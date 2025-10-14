"""Tests for table display functionality."""

from datetime import date

import pytest
from returns.result import Failure, Success

from dqx.common import AssertionResult, EvaluationFailure, SymbolInfo
from dqx.display import print_assertion_results, print_symbols


def test_print_assertion_results_basic() -> None:
    """Test print_assertion_results runs without error."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            check="Price Check",
            assertion="Average price is positive",
            severity="P1",
            status="OK",
            metric=Success(42.5),
            expression="average(price) > 0",
            tags={"env": "prod", "region": "us"},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            check="Data Check",
            assertion="No null values",
            severity="P0",
            status="FAILURE",
            metric=Failure(
                [
                    EvaluationFailure(
                        error_message="Found 10 null values",
                        expression="null_count(id)",
                        symbols=[],
                    )
                ]
            ),
            expression="null_count(id) = 0",
            tags={},
        ),
    ]

    # Should not raise any exceptions
    print_assertion_results(results)


def test_print_symbols_basic() -> None:
    """Test print_symbols runs without error."""
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="average(price)",
            dataset="orders",
            value=Success(42.5),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={"env": "prod"},
        ),
        SymbolInfo(
            name="x_2",
            metric="count(*)",
            dataset="orders",
            value=Success(1000.0),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={"env": "prod"},
        ),
        SymbolInfo(
            name="x_3",
            metric="max(amount)",
            dataset=None,
            value=Failure("No data found"),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={},
        ),
    ]

    # Should not raise any exceptions
    print_symbols(symbols)


def test_print_assertion_results_empty_list() -> None:
    """Test print_assertion_results with empty list."""
    print_assertion_results([])


def test_print_symbols_empty_list() -> None:
    """Test print_symbols with empty list."""
    print_symbols([])


def test_print_assertion_results_edge_cases() -> None:
    """Test print_assertion_results with edge case values."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="Test assertion",
            severity="P1",
            status="OK",
            metric=Success(0.0),  # Zero value
            expression=None,  # None expression
            tags={},  # Empty tags
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="Complex validation",
            severity="P0",
            status="FAILURE",
            metric=Failure(
                [
                    EvaluationFailure(
                        error_message="First error",
                        expression="expr1",
                        symbols=[],
                    ),
                    EvaluationFailure(
                        error_message="Second error",
                        expression="expr2",
                        symbols=[],
                    ),
                ]
            ),
            expression="complex expression",
            tags={"test": "true"},
        ),
    ]

    # Should not raise any exceptions
    print_assertion_results(results)


def test_print_assertion_results_all_severity_levels() -> None:
    """Test print_assertion_results with all severity levels."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="P0 assertion",
            severity="P0",
            status="OK",
            metric=Success(1.0),
            expression="test",
            tags={},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="P1 assertion",
            severity="P1",
            status="OK",
            metric=Success(1.0),
            expression="test",
            tags={},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="P2 assertion",
            severity="P2",
            status="OK",
            metric=Success(1.0),
            expression="test",
            tags={},
        ),
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="P3 assertion",
            severity="P3",
            status="OK",
            metric=Success(1.0),
            expression="test",
            tags={},
        ),
    ]

    # Should not raise any exceptions
    print_assertion_results(results)


def test_print_symbols_with_none_dataset() -> None:
    """Test print_symbols handles None dataset correctly."""
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="count(*)",
            dataset=None,  # None dataset
            value=Success(100.0),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={},
        )
    ]

    # Should not raise any exceptions
    print_symbols(symbols)


def test_print_symbols_success_and_failure_values() -> None:
    """Test print_symbols handles both success and failure values."""
    symbols = [
        SymbolInfo(
            name="x_success",
            metric="average(price)",
            dataset="orders",
            value=Success(42.5),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={},
        ),
        SymbolInfo(
            name="x_failure",
            metric="max(amount)",
            dataset="orders",
            value=Failure("Connection timeout"),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={},
        ),
    ]

    # Should not raise any exceptions
    print_symbols(symbols)


def test_print_assertion_results_complex_tags() -> None:
    """Test print_assertion_results handles complex tag structures."""
    results = [
        AssertionResult(
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test",
            check="Check",
            assertion="Complex tags assertion",
            severity="P1",
            status="OK",
            metric=Success(1.0),
            expression="test",
            tags={"env": "production", "region": "us-west", "version": "1.2.3"},
        )
    ]

    # Should not raise any exceptions
    print_assertion_results(results)


def test_print_symbols_complex_tags() -> None:
    """Test print_symbols handles complex tag structures."""
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="average(price)",
            dataset="orders",
            value=Success(42.5),
            yyyy_mm_dd=date(2024, 1, 15),
            suite="Test Suite",
            tags={"env": "production", "region": "us-west", "team": "data"},
        )
    ]

    # Should not raise any exceptions
    print_symbols(symbols)


@pytest.mark.demo
def test_table_display_demo_integration() -> None:
    """Integration test that mimics the demo script."""
    # Create sample assertion results (similar to demo)
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
    ]

    # Create sample symbol values (similar to demo)
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
    ]

    # Test both functions work together without errors
    print_assertion_results(assertion_results)
    print_symbols(symbols)
