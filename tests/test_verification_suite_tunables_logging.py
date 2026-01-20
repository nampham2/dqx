"""Tests for tunable logging during VerificationSuite.run() evaluation phase."""

from __future__ import annotations

import datetime as dt

import pytest

from dqx.api import Context, VerificationSuite, check
from dqx.common import ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.tunables import TunableChoice, TunableFloat, TunableInt
from tests.fixtures.data_fixtures import CommercialDataSource


@pytest.fixture
def test_db() -> InMemoryMetricDB:
    """Provide a fresh InMemoryMetricDB for each test."""
    return InMemoryMetricDB()


@pytest.fixture
def test_datasource() -> CommercialDataSource:
    """Provide a standard CommercialDataSource for testing."""
    return CommercialDataSource(
        start_date=dt.date(2025, 1, 1),
        end_date=dt.date(2025, 1, 31),
        name="orders",
        records_per_day=30,
        seed=1000,
    )


@pytest.fixture
def test_key() -> ResultKey:
    """Provide a standard ResultKey for testing."""
    return ResultKey(yyyy_mm_dd=dt.date(2025, 1, 15), tags={})


class TestTunableLoggingFloat:
    """Tests for TunableFloat logging during evaluation."""

    def test_log_tunables_float_during_run(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that TunableFloat values are logged during evaluation."""
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.2))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            expr = x - threshold  # Use tunable in expression
            ctx.assert_that(expr).where(name="Test").is_gt(0)

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        # Capture logs during run
        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify log output contains tunable information
        assert "Evaluating with 1 tunable(s):" in captured.out
        assert "THRESHOLD (float) = 0.05 [0.0-0.2]" in captured.out

    def test_log_tunables_float_bounds_format(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that TunableFloat bounds are formatted correctly."""
        threshold = TunableFloat("MAX_ERROR_RATE", value=0.001, bounds=(0.0, 1.0))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x * threshold).where(name="Test").is_lt(100)

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify bounds format
        assert "MAX_ERROR_RATE (float) = 0.001 [0.0-1.0]" in captured.out


class TestTunableLoggingInt:
    """Tests for TunableInt logging during evaluation."""

    def test_log_tunables_int_during_run(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that TunableInt values are logged during evaluation."""
        min_rows = TunableInt("MIN_ROWS", value=1000, bounds=(100, 10000))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            expr = x - min_rows
            ctx.assert_that(expr).where(name="Test").is_gt(0)

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify log output
        assert "Evaluating with 1 tunable(s):" in captured.out
        assert "MIN_ROWS (int) = 1000 [100-10000]" in captured.out


class TestTunableLoggingChoice:
    """Tests for TunableChoice logging during evaluation."""

    def test_log_tunables_choice_during_run(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that TunableChoice values are logged during evaluation."""
        method = TunableChoice("METHOD", value="mean", choices=("mean", "median", "max"))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            # Store method in context to ensure it's tracked
            # Use in assertion via closure
            ctx.assert_that(mp.num_rows()).where(name="Test").is_gt(0)
            # Add tunable reference so it's collected
            _ = method.value  # Reference tunable to ensure collection

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        # Manually add tunable to suite for this test
        suite._tunables["METHOD"] = method

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify log output with quoted value (choices may be truncated by Rich)
        assert "Evaluating with 1 tunable(s):" in captured.out
        assert 'METHOD (choice) = "mean"' in captured.out


class TestTunableLoggingMultiple:
    """Tests for logging multiple tunables."""

    def test_log_multiple_tunables_sorted_order(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that multiple tunables are logged in alphabetical order."""
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.2))
        min_rows = TunableInt("MIN_ROWS", value=1000, bounds=(100, 10000))
        max_error = TunableFloat("MAX_ERROR", value=0.01, bounds=(0.0, 0.1))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="Test1").is_gt(0)
            ctx.assert_that(x - min_rows).where(name="Test2").is_gt(0)
            ctx.assert_that(x * max_error).where(name="Test3").is_lt(100)

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify count
        assert "Evaluating with 3 tunable(s):" in captured.out

        # Verify all tunables logged
        assert "MAX_ERROR (float) = 0.01 [0.0-0.1]" in captured.out
        assert "MIN_ROWS (int) = 1000 [100-10000]" in captured.out
        assert "THRESHOLD (float) = 0.05 [0.0-0.2]" in captured.out

        # Verify alphabetical order (MAX_ERROR, MIN_ROWS, THRESHOLD)
        # Search for specific tunable log entries (not just names, to avoid finding them in "Discovered" log)
        max_error_pos = captured.out.find("- MAX_ERROR (float)")
        min_rows_pos = captured.out.find("- MIN_ROWS (int)")
        threshold_pos = captured.out.find("- THRESHOLD (float)")
        assert max_error_pos < min_rows_pos < threshold_pos

    def test_log_mixed_tunable_types(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test logging with all three tunable types."""
        threshold = TunableFloat("Z_THRESHOLD", value=0.05, bounds=(0.0, 0.2))
        min_rows = TunableInt("A_MIN_ROWS", value=1000, bounds=(100, 10000))
        method = TunableChoice("M_METHOD", value="mean", choices=("mean", "median", "max"))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="Test1").is_gt(0)
            ctx.assert_that(x - min_rows).where(name="Test2").is_gt(0)
            _ = method.value  # Reference to ensure collection

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        # Manually add choice tunable
        suite._tunables["M_METHOD"] = method

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify all types logged correctly (choices may be truncated by Rich)
        assert "Evaluating with 3 tunable(s):" in captured.out
        assert "A_MIN_ROWS (int) = 1000 [100-10000]" in captured.out
        assert 'M_METHOD (choice) = "mean"' in captured.out
        assert "Z_THRESHOLD (float) = 0.05 [0.0-0.2]" in captured.out


class TestTunableLoggingEdgeCases:
    """Tests for edge cases in tunable logging."""

    def test_no_log_when_no_tunables(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that no tunable logging occurs when suite has no tunables."""

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="Test").is_positive()

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify NO tunable-related log appears
        assert "tunable" not in captured.out.lower()
        assert "Evaluating with" not in captured.out

    def test_log_appears_after_active_profiles(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that tunable log appears after 'Active profiles' log."""
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.2))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).where(name="Test").is_gt(0)

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Find positions in log text
        profiles_pos = captured.out.find("Active profiles:")
        tunables_pos = captured.out.find("Evaluating with 1 tunable(s):")

        # Verify ordering
        assert profiles_pos >= 0, "Active profiles log not found"
        assert tunables_pos >= 0, "Tunable log not found"
        assert profiles_pos < tunables_pos, "Tunable log should appear after Active profiles"


class TestTunableLoggingClosureBased:
    """Tests for closure-based tunable logging."""

    def test_tunable_logging_with_closure_based_tunables(
        self,
        test_db: InMemoryMetricDB,
        test_datasource: CommercialDataSource,
        test_key: ResultKey,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test tunables used in closure-based assertions are logged correctly."""
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.2))

        @check(name="Test Check", datasets=["orders"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            # Closure-based: tunable used in comparison method
            ctx.assert_that(mp.num_rows()).where(name="Test").is_gt(threshold)

        suite = VerificationSuite(
            checks=[test_check],
            db=test_db,
            name="Test Suite",
        )

        suite.run([test_datasource], test_key)
        captured = capsys.readouterr()

        # Verify tunable is logged even when used in closure
        assert "Evaluating with 1 tunable(s):" in captured.out
        assert "THRESHOLD (float) = 0.05 [0.0-0.2]" in captured.out
