"""Test for timer fallback in VerificationSuite._process_plugins."""

import datetime
from unittest.mock import patch

import pyarrow as pa

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey, SqlDataSource
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_process_plugins_timer_fallback_with_execution_start() -> None:
    """Test _process_plugins falls back to _execution_start when timer fails."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create test data
    data = pa.table({"x": [1, 2, 3]})
    datasources: dict[str, SqlDataSource] = {"test": ArrowDataSource(data)}
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite to ensure it's evaluated and _execution_start is set
    suite.run(datasources, key, enable_plugins=False)

    # Now simulate timer failure when processing plugins
    with patch.object(suite._analyze_ms, "elapsed_ms", side_effect=RuntimeError("Timer not stopped")):
        # Mock time.time in the api module specifically to control the fallback calculation
        with patch("dqx.api.time.time", return_value=1001.0) as mock_time:
            # Process plugins - should use fallback
            suite._process_plugins(datasources)

            # Verify that time.time was called at least once for the fallback calculation
            assert mock_time.call_count >= 1


def test_process_plugins_timer_fallback_without_execution_start() -> None:
    """Test _process_plugins uses 0.0 when timer fails and no _execution_start."""
    db = InMemoryMetricDB()

    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create test data
    data = pa.table({"x": [1, 2, 3]})
    datasources: dict[str, SqlDataSource] = {"test": ArrowDataSource(data)}
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Run the suite
    suite.run(datasources, key, enable_plugins=False)

    # Delete _execution_start to simulate it not being set
    if hasattr(suite, "_execution_start"):
        delattr(suite, "_execution_start")

    # Now simulate timer failure when processing plugins
    with patch.object(suite._analyze_ms, "elapsed_ms", side_effect=AttributeError("No tock")):
        # Process plugins - should use 0.0 as fallback
        suite._process_plugins(datasources)

        # The function should complete without error
        # Duration will be 0.0 since no _execution_start exists
