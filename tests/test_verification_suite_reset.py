"""Tests for VerificationSuite.reset() functionality."""

from __future__ import annotations

import datetime as dt

import pytest

from dqx.api import VerificationSuite, check
from dqx.common import Context, DQXError, ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.tunables import TunableFloat, TunableInt, TunablePercent
from tests.fixtures.data_fixtures import CommercialDataSource


class TestResetBasicFunctionality:
    """Tests for basic reset() behavior."""

    def test_reset_clears_state(self) -> None:
        """Verify reset() clears all execution state."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        # Create test data
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run suite
        suite.run([ds], key)

        # Verify state is set
        assert suite.is_evaluated is True
        assert suite._key is not None
        # Note: _cached_results may be populated by plugins during run()
        assert suite._analysis_reports is not None
        assert suite._metrics_stats is not None

        # Call reset()
        suite.reset()

        # Verify state is cleared
        assert suite.is_evaluated is False
        assert suite._key is None
        assert suite._cached_results is None
        assert suite._analysis_reports is None
        assert suite._metrics_stats is None
        assert suite._plugin_manager is None

    def test_reset_allows_multiple_runs(self) -> None:
        """Verify suite can be run multiple times after reset()."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        # Create test data
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run 1
        suite.run([ds], key)
        results1 = suite.collect_results()
        assert len(results1) == 1
        assert results1[0].status == "PASSED"

        # Reset and run 2
        suite.reset()
        suite.run([ds], key)
        results2 = suite.collect_results()
        assert len(results2) == 1
        assert results2[0].status == "PASSED"

        # Reset and run 3
        suite.reset()
        suite.run([ds], key)
        results3 = suite.collect_results()
        assert len(results3) == 1
        assert results3[0].status == "PASSED"

    def test_reset_before_run_is_safe(self) -> None:
        """Verify reset() is idempotent and safe to call before run()."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        # Create test data
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Reset before any run (should be safe)
        suite.reset()
        assert suite.is_evaluated is False

        # Run should succeed
        suite.run([ds], key)
        results = suite.collect_results()
        assert len(results) == 1
        assert results[0].status == "PASSED"

    def test_reset_multiple_times_without_running(self) -> None:
        """Verify reset() can be called multiple times without running."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        # Multiple resets without running
        suite.reset()
        suite.reset()
        suite.reset()

        # Should still be able to run
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})
        suite.run([ds], key)
        results = suite.collect_results()
        assert len(results) == 1


class TestResetWithTunables:
    """Tests for reset() with tunable parameter adjustments (main use case)."""

    def test_reset_with_tunable_threshold_adjustment(self) -> None:
        """
        Verify AI agent can tune threshold via reset() workflow.

        This is the primary use case: an agent runs the suite, observes failures,
        adjusts tunable parameters, resets, and runs again.
        """
        # Create a tunable threshold
        null_threshold = TunablePercent("NULL_THRESHOLD", value=0.05, bounds=(0.0, 0.50))

        # Create a check that uses the tunable
        @check(name="Null Rate Check", datasets=["orders"])
        def null_rate_check(mp: MetricProvider, ctx: Context) -> None:
            null_rate = mp.null_count("delivered") / mp.num_rows()
            ctx.assert_that(null_rate).where(name="null_rate_assertion").is_lt(null_threshold.value)

        # Set up test data with ~26% null rate
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1050,
        )

        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Create suite with tunable
        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[null_rate_check],
            db=db,
            name="Test Suite",
            tunables=[null_threshold],
        )

        # Run with initial threshold (5%) - should FAIL (null rate ~26% > 5%)
        suite.run([ds], key)
        result1 = suite.collect_results()
        assert result1[0].status == "FAILED", "Initial check should fail with 5% threshold"
        assert suite.get_param("NULL_THRESHOLD") == 0.05

        # AI agent tunes threshold to 30% (more lenient)
        suite.set_param("NULL_THRESHOLD", 0.30, agent="rl_optimizer", reason="Tuning iteration 1")

        # Verify the tunable was updated
        assert null_threshold.value == 0.30
        assert suite.get_param("NULL_THRESHOLD") == 0.30

        # Reset and run again
        suite.reset()
        suite.run([ds], key)
        result2 = suite.collect_results()

        # Should PASS now (null rate ~26% < 30%)
        assert result2[0].status == "PASSED", "Updated check should pass with 30% threshold"
        assert suite.get_param("NULL_THRESHOLD") == 0.30

    def test_reset_preserves_tunables(self) -> None:
        """Verify reset() preserves all tunable values and modifications."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        min_rows = TunableInt("MIN_ROWS", value=1000, bounds=(100, 10000))
        tolerance = TunableFloat("TOLERANCE", value=0.5, bounds=(0.0, 1.0))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            tunables=[threshold, min_rows, tolerance],
        )

        # Modify all tunables
        suite.set_param("THRESHOLD", 0.10, agent="test")
        suite.set_param("MIN_ROWS", 5000, agent="test")
        suite.set_param("TOLERANCE", 0.8, agent="test")

        # Run suite
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})
        suite.run([ds], key)

        # Reset
        suite.reset()

        # Verify tunables are preserved
        assert suite.get_param("THRESHOLD") == 0.10
        assert suite.get_param("MIN_ROWS") == 5000
        assert suite.get_param("TOLERANCE") == 0.8

        # Verify tunable history is preserved
        history = suite.get_param_history("THRESHOLD")
        assert len(history) == 1
        assert history[0].new_value == 0.10

    def test_reset_allows_iterative_tuning(self) -> None:
        """Verify multiple tune-reset-run cycles work correctly."""
        threshold = TunablePercent("THRESHOLD", value=0.01, bounds=(0.0, 0.50))

        @check(name="Null Rate Check", datasets=["orders"])
        def null_rate_check(mp: MetricProvider, ctx: Context) -> None:
            null_rate = mp.null_count("delivered") / mp.num_rows()
            ctx.assert_that(null_rate).where(name="null_rate_assertion").is_lt(threshold.value)

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1050,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[null_rate_check],
            db=db,
            name="Test Suite",
            tunables=[threshold],
        )

        # Iteration 1: threshold=0.01 (too strict) -> FAIL
        suite.run([ds], key)
        assert suite.collect_results()[0].status == "FAILED"

        # Iteration 2: threshold=0.10 (still too strict) -> FAIL
        suite.set_param("THRESHOLD", 0.10, agent="rl", reason="iter 2")
        suite.reset()
        suite.run([ds], key)
        assert suite.collect_results()[0].status == "FAILED"

        # Iteration 3: threshold=0.20 (still too strict) -> FAIL
        suite.set_param("THRESHOLD", 0.20, agent="rl", reason="iter 3")
        suite.reset()
        suite.run([ds], key)
        assert suite.collect_results()[0].status == "FAILED"

        # Iteration 4: threshold=0.30 (just right) -> PASS
        suite.set_param("THRESHOLD", 0.30, agent="rl", reason="iter 4")
        suite.reset()
        suite.run([ds], key)
        assert suite.collect_results()[0].status == "PASSED"

        # Verify tunable history tracks all changes (3 set_param calls)
        history = suite.get_param_history("THRESHOLD")
        assert len(history) == 3
        assert [h.new_value for h in history] == [0.10, 0.20, 0.30]


class TestResetExecutionId:
    """Tests for execution_id behavior with reset()."""

    def test_reset_generates_new_execution_id(self) -> None:
        """Verify each reset generates a unique execution_id."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        # Capture initial execution_id
        initial_execution_id = suite.execution_id

        # Create test data
        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run 1
        suite.run([ds], key)
        run1_execution_id = suite.execution_id
        assert run1_execution_id == initial_execution_id

        # Reset and verify new execution_id
        suite.reset()
        run2_execution_id = suite.execution_id
        assert run2_execution_id != run1_execution_id

        # Run 2
        suite.run([ds], key)
        assert suite.execution_id == run2_execution_id

        # Reset again and verify another new execution_id
        suite.reset()
        run3_execution_id = suite.execution_id
        assert run3_execution_id != run1_execution_id
        assert run3_execution_id != run2_execution_id

    def test_reset_execution_id_distinguishes_tuning_iterations_in_db(self) -> None:
        """Verify different execution_ids allow tracking tuning iterations in DB."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.50))

        @check(name="Null Rate Check", datasets=["orders"])
        def null_rate_check(mp: MetricProvider, ctx: Context) -> None:
            null_rate = mp.null_count("delivered") / mp.num_rows()
            ctx.assert_that(null_rate).where(name="null_rate_assertion").is_lt(threshold.value)

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1050,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[null_rate_check],
            db=db,
            name="Test Suite",
            tunables=[threshold],
        )

        # Run iteration 1
        suite.run([ds], key)
        exec_id_1 = suite.execution_id
        metrics_1 = db.get_by_execution_id(exec_id_1)

        # Run iteration 2 after reset
        suite.set_param("THRESHOLD", 0.30, agent="rl")
        suite.reset()
        suite.run([ds], key)
        exec_id_2 = suite.execution_id
        metrics_2 = db.get_by_execution_id(exec_id_2)

        # Verify different execution_ids
        assert exec_id_1 != exec_id_2

        # Verify metrics are stored separately by execution_id
        assert len(metrics_1) > 0
        assert len(metrics_2) > 0
        # All metrics from iteration 1 should have exec_id_1 in metadata
        for metric in metrics_1:
            assert metric.metadata is not None
            assert metric.metadata.execution_id == exec_id_1
        # All metrics from iteration 2 should have exec_id_2 in metadata
        for metric in metrics_2:
            assert metric.metadata is not None
            assert metric.metadata.execution_id == exec_id_2


class TestResetPreservesConfiguration:
    """Tests that reset() preserves suite configuration."""

    def test_reset_preserves_checks(self) -> None:
        """Verify reset() doesn't lose check definitions."""

        @check(name="Check 1", datasets=["orders"])
        def check1(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="rows_positive").is_positive()

        @check(name="Check 2", datasets=["orders"])
        def check2(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.average("price")).where(name="avg_price_positive").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[check1, check2],
            db=db,
            name="Test Suite",
        )

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run
        suite.run([ds], key)
        results1 = suite.collect_results()
        assert len(results1) == 2

        # Reset and run again
        suite.reset()
        suite.run([ds], key)
        results2 = suite.collect_results()
        assert len(results2) == 2

        # Verify same checks are present
        assert {r.assertion for r in results1} == {r.assertion for r in results2}

    def test_reset_preserves_data_av_threshold(self) -> None:
        """Verify reset() preserves data_av_threshold configuration."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            data_av_threshold=0.85,  # Custom threshold
        )

        assert suite.data_av_threshold == 0.85

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        suite.run([ds], key)
        suite.reset()

        # Threshold should be preserved
        assert suite.data_av_threshold == 0.85

    def test_reset_preserves_suite_name(self) -> None:
        """Verify reset() preserves suite name."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite_name = "My Custom Suite Name"
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name=suite_name,
        )

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        suite.run([ds], key)
        result1 = suite.collect_results()[0]
        assert result1.suite == suite_name

        suite.reset()
        suite.run([ds], key)
        result2 = suite.collect_results()[0]
        assert result2.suite == suite_name


class TestResetPluginManager:
    """Tests for plugin manager clearing on reset()."""

    def test_reset_clears_plugin_manager(self) -> None:
        """Verify reset() clears lazy-loaded plugin_manager."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run suite
        suite.run([ds], key)

        # Access plugin_manager to trigger lazy loading
        _ = suite.plugin_manager
        assert suite._plugin_manager is not None

        # Reset should clear it
        suite.reset()
        assert suite._plugin_manager is None

        # Should be lazy-loaded again on next access
        _ = suite.plugin_manager
        assert suite._plugin_manager is not None

    def test_reset_reinitializes_plugin_manager_with_all_plugins(self) -> None:
        """Verify plugin_manager is properly initialized with plugins after reset and run."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run 1 - plugins should be registered and executed
        suite.run([ds], key)
        plugin_mgr_1 = suite.plugin_manager
        assert plugin_mgr_1 is not None
        assert suite.plugin_manager.plugin_exists("audit"), "Default audit plugin should be registered"

        # Verify audit plugin actually ran by checking that results were processed
        # The audit plugin sets _cached_results during process_all()
        results_1 = suite.collect_results()
        assert len(results_1) == 1
        assert results_1[0].status == "PASSED"

        # Store reference to first plugin manager instance
        first_plugin_manager_id = id(plugin_mgr_1)

        # Reset - plugin manager should be cleared
        suite.reset()
        assert suite._plugin_manager is None

        # Run 2 - plugins should be re-registered and executed
        suite.run([ds], key)
        plugin_mgr_2 = suite.plugin_manager
        assert plugin_mgr_2 is not None

        # Verify it's a NEW plugin manager instance (not the same object)
        assert id(plugin_mgr_2) != first_plugin_manager_id, "Should create new plugin manager after reset"

        # Verify plugins are registered in the new instance
        assert suite.plugin_manager.plugin_exists("audit"), "Audit plugin should be registered after reset"

        # Verify audit plugin ran again by checking results
        results_2 = suite.collect_results()
        assert len(results_2) == 1
        assert results_2[0].status == "PASSED"

        # Run 3 - Verify it works multiple times
        suite.reset()
        assert suite._plugin_manager is None

        suite.run([ds], key)
        plugin_mgr_3 = suite.plugin_manager
        assert plugin_mgr_3 is not None
        assert id(plugin_mgr_3) != first_plugin_manager_id
        assert id(plugin_mgr_3) != id(plugin_mgr_2)
        assert suite.plugin_manager.plugin_exists("audit"), "Audit plugin should be registered after second reset"

        results_3 = suite.collect_results()
        assert len(results_3) == 1
        assert results_3[0].status == "PASSED"


class TestResetErrorHandling:
    """Tests for reset() behavior with errors and edge cases."""

    def test_reset_after_failed_run(self) -> None:
        """Verify reset() works correctly after a failed run."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            # This will fail because price is always > 0
            ctx.assert_that(mp.average("price")).where(name="price_negative").is_negative()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run with failing assertion
        suite.run([ds], key)
        result1 = suite.collect_results()[0]
        assert result1.status == "FAILED"

        # Reset should work fine
        suite.reset()
        assert suite.is_evaluated is False

        # Should be able to run again
        suite.run([ds], key)
        result2 = suite.collect_results()[0]
        assert result2.status == "FAILED"

    def test_cannot_collect_results_after_reset(self) -> None:
        """Verify collect_results() raises error after reset()."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run and collect results (should work)
        suite.run([ds], key)
        results = suite.collect_results()
        assert len(results) == 1

        # Reset
        suite.reset()

        # Attempting to collect results should raise error
        with pytest.raises(DQXError, match="not been executed"):
            suite.collect_results()

    def test_cannot_access_graph_after_reset(self) -> None:
        """Verify graph property raises error after reset()."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="row_count").is_positive()

        db = InMemoryMetricDB()
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        ds = CommercialDataSource(
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 1, 31),
            name="orders",
            records_per_day=30,
            seed=1000,
        )
        key = ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})

        # Run
        suite.run([ds], key)
        _ = suite.graph  # Should work

        # Reset
        suite.reset()

        # Accessing graph should raise error
        with pytest.raises(DQXError, match="not been executed"):
            _ = suite.graph
