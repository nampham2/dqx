"""Tests for execution ID functionality in VerificationSuite."""

from __future__ import annotations

import uuid
from datetime import date

import pyarrow as pa

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_verification_suite_execution_id() -> None:
    """Test that VerificationSuite generates unique execution ID."""

    # Create a dummy check to satisfy the requirement
    @check(name="Dummy Check")
    def dummy_check(mp: MetricProvider, ctx: Context) -> None:
        pass

    db = InMemoryMetricDB()
    suite1 = VerificationSuite([dummy_check], db, "Test Suite 1")
    suite2 = VerificationSuite([dummy_check], db, "Test Suite 2")

    # Should have execution_id property
    assert hasattr(suite1, "execution_id")
    assert hasattr(suite2, "execution_id")

    # Should be valid UUIDs
    uuid.UUID(suite1.execution_id)  # Raises if invalid
    uuid.UUID(suite2.execution_id)  # Raises if invalid

    # Should be unique
    assert suite1.execution_id != suite2.execution_id


def test_execution_id_in_metadata() -> None:
    """Test that execution_id is stored in metric metadata."""

    # Create a check that uses a metric
    @check(name="Test Check")
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        avg_price = mp.average("price")
        ctx.assert_that(avg_price).config(name="Price average check").is_positive()

    # Create test data
    table = pa.table({"price": [10.0, 20.0, 30.0]})
    datasource = DuckRelationDataSource.from_arrow(table, "test_data")

    # Run the suite
    db = InMemoryMetricDB()
    suite = VerificationSuite([test_check], db, "Test Suite")
    key = ResultKey(date.today(), {"env": "test"})
    suite.run([datasource], key)

    # Check that all metrics in the DB have execution_id in metadata
    execution_id = suite.execution_id

    # Get all metrics with this execution_id
    all_metrics = db.get_by_execution_id(execution_id)

    # Verify all metrics have the execution_id in metadata
    assert len(all_metrics) > 0, "Should have at least one metric persisted"
    for metric in all_metrics:
        assert metric.metadata is not None
        assert metric.metadata.execution_id == execution_id
        # Also verify the metric is from the correct date and dataset
        assert metric.key.yyyy_mm_dd == date.today()
        assert metric.dataset == "test_data"
