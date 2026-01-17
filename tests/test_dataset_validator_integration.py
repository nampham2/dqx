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
        # Mismatched dataset
        avg_price = mp.average("price", dataset="testing")
        ctx.assert_that(avg_price).where(name="Price is positive").is_positive()

    # Build graph should fail validation during suite creation
    with pytest.raises(DQXError) as exc_info:
        VerificationSuite([price_check], db, "Test Suite")

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

    # Validation should report errors during suite creation due to second check
    with pytest.raises(DQXError) as exc_info:
        VerificationSuite([valid_check, invalid_check], db, "Test Suite")

    error_str = str(exc_info.value)
    assert "Invalid Check" in error_str
    assert "testing" in error_str
