"""Tests for VerificationSuite critical level detection."""

from datetime import date

import pyarrow as pa
import pytest

from dqx.api import Context, VerificationSuite, check
from dqx.common import DQXError, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


class TestSuiteCritical:
    """Test suite for is_critical functionality."""

    def test_is_critical_with_p0_failure(self) -> None:
        """Suite with P0 failures should be critical."""
        db = InMemoryMetricDB()

        @check(name="Critical Check")
        def critical_check(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            # This will fail - expecting 1000 but sum is 100
            ctx.assert_that(total).config(name="Critical metric", severity="P0").is_eq(1000)

        suite = VerificationSuite([critical_check], db, "Test Suite")
        data = pa.table({"value": [25, 25, 25, 25]})
        ds = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(date.today(), {"env": "test"})

        suite.run([ds], key)

        # Should be critical due to P0 failure
        assert suite.is_critical() is True

    def test_is_critical_with_no_p0_failures(self) -> None:
        """Suite without P0 failures should not be critical."""
        db = InMemoryMetricDB()

        @check(name="Non-Critical Checks")
        def non_critical_checks(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            # P1 assertion that passes
            ctx.assert_that(total).config(name="P1 check", severity="P1").is_eq(100)
            # P2 assertion that fails
            ctx.assert_that(total).config(name="P2 check", severity="P2").is_eq(200)

        suite = VerificationSuite([non_critical_checks], db, "Test Suite")
        data = pa.table({"value": [25, 25, 25, 25]})
        ds = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(date.today(), {"env": "test"})

        suite.run([ds], key)

        # Should not be critical (no P0 failures)
        assert suite.is_critical() is False

    def test_is_critical_with_mixed_severities(self) -> None:
        """Suite with mixed severities including P0 failure should be critical."""
        db = InMemoryMetricDB()

        @check(name="Mixed Severity Check")
        def mixed_check(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            # P0 that fails
            ctx.assert_that(total).config(name="Critical check", severity="P0").is_eq(1000)
            # P1 that passes
            ctx.assert_that(total).config(name="Important check", severity="P1").is_positive()
            # P2 that fails
            ctx.assert_that(total).config(name="Minor check", severity="P2").is_negative()

        suite = VerificationSuite([mixed_check], db, "Test Suite")
        data = pa.table({"value": [25, 25, 25, 25]})
        ds = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(date.today(), {"env": "test"})

        suite.run([ds], key)

        # Should be critical due to P0 failure
        assert suite.is_critical() is True

    def test_is_critical_before_run_raises_error(self) -> None:
        """Calling is_critical before run should raise DQXError."""
        db = InMemoryMetricDB()

        @check(name="Never Run")
        def never_run(mp: MetricProvider, ctx: Context) -> None:
            pass

        suite = VerificationSuite([never_run], db, "Test Suite")

        with pytest.raises(DQXError, match="not been executed"):
            suite.is_critical()

    def test_is_critical_with_no_assertions(self) -> None:
        """Suite with no assertions should not be critical."""
        db = InMemoryMetricDB()

        @check(name="Empty Check")
        def empty_check(mp: MetricProvider, ctx: Context) -> None:
            # No assertions
            pass

        suite = VerificationSuite([empty_check], db, "Test Suite")
        data = pa.table({"value": [1, 2, 3]})
        ds = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(date.today(), {"env": "test"})

        suite.run([ds], key)

        # Should not be critical (no assertions at all)
        assert suite.is_critical() is False

    def test_is_critical_uses_cached_results(self) -> None:
        """is_critical should use cached collect_results for efficiency."""
        db = InMemoryMetricDB()

        @check(name="P0 Failure")
        def p0_failure(mp: MetricProvider, ctx: Context) -> None:
            total = mp.sum("value")
            ctx.assert_that(total).config(name="Will fail", severity="P0").is_eq(999)

        suite = VerificationSuite([p0_failure], db, "Test Suite")
        data = pa.table({"value": [1, 2, 3]})
        ds = DuckRelationDataSource.from_arrow(data, "data")
        key = ResultKey(date.today(), {"env": "test"})

        suite.run([ds], key)

        # First call to collect_results
        results1 = suite.collect_results()

        # Call is_critical
        is_critical = suite.is_critical()
        assert is_critical is True

        # Get results again - should be cached
        results2 = suite.collect_results()
        assert results1 is results2  # Same object reference
