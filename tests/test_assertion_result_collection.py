"""Tests for assertion result collection functionality."""

import datetime

import pyarrow as pa
import pytest
from returns.result import Failure, Success

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import AssertionResult, DQXError, EvaluationFailure, ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


class TestAssertionResultCollection:
    """Test suite for collect_results functionality."""

    def test_collect_results_before_run_raises_error(self) -> None:
        """Should raise DQXError if collect_results called before run()."""

        @check(name="dummy check")
        def dummy_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="rows > 0").is_gt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([dummy_check], db, "Test Suite")

        # Should not be evaluated initially
        assert suite.is_evaluated is False

        # Should raise error when trying to collect results
        with pytest.raises(DQXError) as exc_info:
            suite.collect_results()

        assert "Cannot collect results before suite execution" in str(exc_info.value)

    def test_suite_cannot_run_twice(self) -> None:
        """Should raise DQXError if suite.run() called twice."""

        @check(name="dummy check")
        def dummy_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="rows > 0").is_gt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([dummy_check], db, "Test Suite")

        # Create test data
        data = pa.table({"x": [1, 2, 3]})
        datasource = ArrowDataSource(data)
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # First run should succeed
        suite.run({"data": datasource}, key)
        assert suite.is_evaluated is True

        # Second run should raise error
        with pytest.raises(DQXError) as exc_info:
            suite.run({"data": datasource}, key)

        assert "Verification suite has already been executed" in str(exc_info.value)

    def test_collect_results_no_key_needed(self) -> None:
        """Should not require key parameter since it's stored from run()."""

        @check(name="simple check")
        def simple_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row count").is_eq(3)

        db = InMemoryMetricDB()
        suite = VerificationSuite([simple_check], db, "Test Suite")

        # Run with specific key
        data = pa.table({"x": [1, 2, 3]})
        key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 15), tags={"env": "test", "version": "1.0"})
        suite.run({"data": ArrowDataSource(data)}, key)

        # Collect results without passing key
        results = suite.collect_results()  # No key parameter!

        # Verify key was stored and used
        assert len(results) == 1
        assert results[0].yyyy_mm_dd == datetime.date(2024, 1, 15)
        assert results[0].tags == {"env": "test", "version": "1.0"}

    def test_collect_results_after_successful_run(self) -> None:
        """Should return results for all assertions after successful run."""
        # Create test data
        data = pa.table(
            {"id": [1, 2, 3, 4, 5], "value": [10.0, 20.0, 30.0, 40.0, 50.0], "category": ["A", "B", "A", "B", "A"]}
        )
        datasource = ArrowDataSource(data)

        # Define checks with multiple assertions
        @check(name="data validation")
        def validate_data(mp: MetricProvider, ctx: Context) -> None:
            # This should pass
            ctx.assert_that(mp.num_rows()).where(name="Has 5 rows", severity="P0").is_eq(5)

            # This should also pass
            ctx.assert_that(mp.average("value")).where(name="Average value is 30", severity="P1").is_eq(30.0)

            # This should fail
            ctx.assert_that(mp.minimum("value")).where(name="Minimum value check", severity="P2").is_gt(
                20.0
            )  # Will fail since min is 10

        # Build and run suite
        db = InMemoryMetricDB()
        suite = VerificationSuite([validate_data], db, "Test Suite")

        key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 15), tags={"env": "test", "version": "1.0"})

        # Run the suite
        suite.run({"test_data": datasource}, key)

        # Should be evaluated after run
        assert suite.is_evaluated is True

        # Collect results (no key needed!)
        results = suite.collect_results()

        # Verify we got all 3 assertions
        assert len(results) == 3

        # Check first assertion (rows count - should pass)
        r1 = results[0]
        assert r1.yyyy_mm_dd == datetime.date(2024, 1, 15)
        assert r1.suite == "Test Suite"
        assert r1.check == "data validation"
        assert r1.assertion == "Has 5 rows"
        assert r1.severity == "P0"
        assert r1.status == "OK"
        assert isinstance(r1.metric, Success)
        assert r1.metric.unwrap() == 5.0
        assert r1.expression is not None  # Expression should be the symbolic expression like "x_1"
        assert r1.tags == {"env": "test", "version": "1.0"}

        # Check second assertion (average - should pass)
        r2 = results[1]
        assert r2.assertion == "Average value is 30"
        assert r2.severity == "P1"
        assert r2.status == "OK"
        assert r2.metric.unwrap() == 30.0

        # Check third assertion (minimum - should fail)
        r3 = results[2]
        assert r3.assertion == "Minimum value check"
        assert r3.severity == "P2"

        # With the new validation logic:
        # - The metric computation succeeds (we get 10.0)
        # - But the validation fails (10.0 is not > 20.0)
        assert r3.status == "FAILURE"  # Validation failed
        assert isinstance(r3.metric, Success)  # But metric computation succeeded
        assert r3.metric.unwrap() == 10.0  # The minimum value

        # The expression should show the full validation
        assert r3.expression is not None
        assert ">" in r3.expression
        assert "20" in r3.expression

    def test_is_evaluated_only_set_on_success(self) -> None:
        """Should NOT set is_evaluated if run() fails."""

        @check(name="failing check")
        def failing_check(mp: MetricProvider, ctx: Context) -> None:
            # This will cause an error during execution
            raise RuntimeError("Simulated failure")

        db = InMemoryMetricDB()
        suite = VerificationSuite([failing_check], db, "Test Suite")

        assert suite.is_evaluated is False

        # Run should fail
        data = pa.table({"x": [1]})
        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # This should raise an exception
        with pytest.raises(RuntimeError):
            suite.run({"data": ArrowDataSource(data)}, key)

        # Flag should NOT be set on failure
        assert suite.is_evaluated is False

        # collect_results should still raise error
        with pytest.raises(DQXError) as exc_info:
            suite.collect_results()
        assert "Cannot collect results" in str(exc_info.value)

    def test_collect_results_preserves_result_object(self) -> None:
        """Should preserve the full Result object for advanced usage."""
        data = pa.table({"x": [1, 2, 3]})
        datasource = ArrowDataSource(data)

        @check(name="result preservation")
        def check_results(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("x")).where(name="avg check").is_eq(2.0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([check_results], db, "Suite")

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.run({"data": datasource}, key)

        results = suite.collect_results()

        # The metric field should be the actual Result object
        assert len(results) == 1
        result_obj = results[0].metric

        # Should be able to use Result methods
        assert isinstance(result_obj, Success)
        assert result_obj.unwrap() == 2.0

        # Should be able to map/bind on the Result
        doubled = result_obj.map(lambda x: x * 2)
        assert doubled.unwrap() == 4.0

    def test_empty_suite_returns_empty_results(self) -> None:
        """Should return empty list if no assertions in suite."""

        # Suite with a check that has no assertions
        @check(name="empty check")
        def empty_check(mp: MetricProvider, ctx: Context) -> None:
            pass  # No assertions

        db = InMemoryMetricDB()
        suite = VerificationSuite([empty_check], db, "Empty Suite")
        data = pa.table({"x": [1]})

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

        # Empty checks will raise a warning but not fail
        # The suite will not run analyzer since there are no metrics
        try:
            suite.run({"data": ArrowDataSource(data)}, key)
        except DQXError as e:
            # Expected - no metrics to analyze
            assert "No metrics provided for analysis" in str(e)
            # Suite should still be marked as evaluated after error
            assert suite.is_evaluated is False
            return

        # If it somehow succeeded (shouldn't happen), check results
        results = suite.collect_results()
        assert results == []

    def test_failure_details_accessible_from_result(self) -> None:
        """Should be able to access failure details directly from Result object."""
        data = pa.table({"x": [5]})
        datasource = ArrowDataSource(data)

        @check(name="failure test")
        def failure_check(mp: MetricProvider, ctx: Context) -> None:
            # Create an assertion that will fail - we have 1 row, not more than 10
            ctx.assert_that(mp.num_rows()).where(name="impossible check").is_gt(10)

        db = InMemoryMetricDB()
        suite = VerificationSuite([failure_check], db, "Test Suite")

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        suite.run({"data": datasource}, key)

        results = suite.collect_results()
        assert len(results) == 1

        result = results[0]

        # With the new validation logic:
        # - The metric computation succeeds (we get 1 row)
        # - But the validation fails (1 is not > 10)
        assert result.status == "FAILURE"  # Validation failed
        assert isinstance(result.metric, Success)  # But metric computation succeeded
        assert result.metric.unwrap() == 1.0  # We have 1 row

        # The expression should show the full validation
        assert result.expression is not None
        assert ">" in result.expression
        assert "10" in result.expression


def test_assertion_result_dataclass() -> None:
    """Test AssertionResult dataclass behavior."""
    # Test creation with all fields
    result = AssertionResult(
        yyyy_mm_dd=datetime.date(2024, 1, 1),
        suite="Test Suite",
        check="Test Check",
        assertion="Test Assertion",
        severity="P1",
        status="OK",
        metric=Success(42.0),
        expression="x > 0",
        tags={"env": "prod"},
    )

    # Should be able to access all fields
    assert result.yyyy_mm_dd == datetime.date(2024, 1, 1)
    assert result.suite == "Test Suite"
    assert result.check == "Test Check"
    assert result.assertion == "Test Assertion"
    assert result.severity == "P1"
    assert result.status == "OK"
    assert result.metric.unwrap() == 42.0
    assert result.expression == "x > 0"
    assert result.tags == {"env": "prod"}

    # Test with failure
    failure_result = AssertionResult(
        yyyy_mm_dd=datetime.date(2024, 1, 1),
        suite="Test Suite",
        check="Test Check",
        assertion="Failed Assertion",
        severity="P0",
        status="FAILURE",
        metric=Failure(
            [EvaluationFailure(error_message="Value 5 is not greater than 10", expression="x > 10", symbols=[])]
        ),
        expression="x > 10",
        tags={},
    )

    assert failure_result.status == "FAILURE"
    assert isinstance(failure_result.metric, Failure)

    # Verify we can extract failures from Result
    failures = failure_result.metric.failure()
    assert len(failures) == 1
    assert failures[0].error_message == "Value 5 is not greater than 10"
