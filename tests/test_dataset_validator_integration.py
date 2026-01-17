"""Integration tests for DatasetValidator within VerificationSuite."""

import pytest

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import DQXError
from dqx.orm.repositories import InMemoryMetricDB


def test_verification_suite_detects_dataset_mismatch() -> None:
    """Test that VerificationSuite validation catches dataset mismatches."""
    db = InMemoryMetricDB()

    @check(name="Price Check", datasets=["production", "staging"])
    def price_check(mp: MetricProvider, ctx: Context) -> None:
        # Symbol with mismatched dataset
        avg_price = mp.average("price", dataset="testing")
        ctx.assert_that(avg_price).where(name="Price is positive").is_positive()

    # Creating the suite should work
    suite = VerificationSuite([price_check], db, "Test Suite")

    # Validation during build_graph should fail
    with pytest.raises(DQXError) as exc_info:
        suite.build_graph(suite._context)

    error_str = str(exc_info.value)
    assert "dataset_mismatch" in error_str
    assert "testing" in error_str
    assert "production" in error_str
    assert "staging" in error_str


def test_verification_suite_allows_valid_datasets() -> None:
    """Test that VerificationSuite validation allows matching datasets."""
    db = InMemoryMetricDB()

    @check(name="Price Check", datasets=["production", "staging"])
    def price_check(mp: MetricProvider, ctx: Context) -> None:
        # Symbol with matching dataset
        avg_price = mp.average("price", dataset="production")
        ctx.assert_that(avg_price).where(name="Price is positive").is_positive()

    suite = VerificationSuite([price_check], db, "Test Suite")

    # Validation should pass - no exception raised
    suite.build_graph(suite._context)


def test_verification_suite_dataset_ambiguity_error() -> None:
    """Test that ambiguous dataset configuration is caught during validation."""
    db = InMemoryMetricDB()

    @check(name="Price Check", datasets=["production", "staging"])
    def price_check(mp: MetricProvider, ctx: Context) -> None:
        # Symbol with NO dataset but check has multiple - ambiguous!
        avg_price = mp.average("price")  # No dataset specified
        ctx.assert_that(avg_price).where(name="Price is positive").is_positive()

    suite = VerificationSuite([price_check], db, "Test Suite")

    # Validation should report ambiguity error
    with pytest.raises(DQXError) as exc_info:
        suite.build_graph(suite._context)

    error_str = str(exc_info.value)
    assert "no dataset specified" in error_str
    assert "multiple datasets" in error_str


def test_verification_suite_allows_no_dataset_with_single_check_dataset() -> None:
    """Test that no dataset is allowed when check has single dataset."""
    db = InMemoryMetricDB()

    @check(name="Price Check", datasets=["production"])
    def price_check(mp: MetricProvider, ctx: Context) -> None:
        # Symbol with no dataset - OK because check has single dataset
        avg_price = mp.average("price")
        ctx.assert_that(avg_price).where(name="Price is positive").is_positive()

    suite = VerificationSuite([price_check], db, "Test Suite")

    # Validation should pass - imputation will handle it
    suite.build_graph(suite._context)


def test_verification_suite_no_validation_without_datasets() -> None:
    """Test that validation skips when check has no datasets."""
    db = InMemoryMetricDB()

    @check(name="Price Check")  # No datasets specified
    def price_check(mp: MetricProvider, ctx: Context) -> None:
        # Symbol with dataset
        avg_price = mp.average("price", dataset="testing")
        ctx.assert_that(avg_price).where(name="Price is positive").is_positive()

    suite = VerificationSuite([price_check], db, "Test Suite")

    # Validation should pass - no dataset validation when check has no datasets
    suite.build_graph(suite._context)


def test_verification_suite_build_graph_validates_datasets() -> None:
    """Test that build_graph method also validates datasets."""
    db = InMemoryMetricDB()

    @check(name="Price Check", datasets=["production"])
    def price_check(mp: MetricProvider, ctx: Context) -> None:
        # Mismatched dataset
        avg_price = mp.average("price", dataset="testing")
        ctx.assert_that(avg_price).where(name="Price is positive").is_positive()

    suite = VerificationSuite([price_check], db, "Test Suite")

    # Build graph should fail validation
    with pytest.raises(DQXError) as exc_info:
        suite.build_graph(suite._context)

    assert "testing" in str(exc_info.value)


def test_multiple_checks_with_dataset_issues() -> None:
    """Test validation with multiple checks having different dataset issues."""
    db = InMemoryMetricDB()

    @check(name="Valid Check", datasets=["production"])
    def valid_check(mp: MetricProvider, ctx: Context) -> None:
        # Matching dataset
        avg_price = mp.average("price", dataset="production")
        ctx.assert_that(avg_price).where(name="Price OK").is_positive()

    @check(name="Invalid Check", datasets=["staging"])
    def invalid_check(mp: MetricProvider, ctx: Context) -> None:
        # Mismatched dataset
        avg_cost = mp.average("cost", dataset="testing")
        ctx.assert_that(avg_cost).where(name="Cost OK").is_positive()

    suite = VerificationSuite([valid_check, invalid_check], db, "Test Suite")

    # Validation should report errors due to second check
    with pytest.raises(DQXError) as exc_info:
        suite.build_graph(suite._context)

    error_str = str(exc_info.value)
    assert "Invalid Check" in error_str
    assert "testing" in error_str
