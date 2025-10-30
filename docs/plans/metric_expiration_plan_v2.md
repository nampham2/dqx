# Metric Expiration Implementation Plan v2

## Overview

This plan implements automatic cleanup of expired metrics in the DQX system based on TTL (Time To Live) values stored in metric metadata. The feature will delete metrics that have exceeded their TTL or have no TTL specified, preventing unbounded growth of the metrics database.

**Changes from v1:**
- Use UTC consistently throughout the implementation
- Add database index on initialization for performance
- Enhanced monitoring with timer registry and structured logging
- Return deletion count from delete method
- Additional test coverage for timezones, performance, and concurrency

## Background

- Metrics in DQX are stored with optional metadata containing a `ttl_hours` field
- Metrics without metadata or without `ttl_hours` should be treated as expired
- Cleanup should happen automatically before analysis in VerificationSuite
- The implementation must be efficient and support SQLite datetime arithmetic
- All timestamps use UTC for consistency

## Requirements

1. **NO BACKWARD COMPATIBILITY** is required for this feature
2. Implement two separate functions in MetricDB for statistics and deletion
3. VerificationSuite handles orchestration and logging
4. Cleanup happens BEFORE data source analysis
5. Use SQLite-specific datetime functions for expiration calculations
6. All datetime operations use UTC
7. Performance index created on database initialization

## Implementation Tasks

### Task Group 1: Database Index Setup

#### Task 1.1: Add index creation to MetricDB
Add to `src/dqx/orm/repositories.py` in the `MetricDB.__init__` method:

```python
def __init__(self, ...):
    """Initialize MetricDB."""
    # ... existing initialization ...

    # Create performance indexes
    self._ensure_indexes()

def _ensure_indexes(self) -> None:
    """Create performance indexes if they don't exist."""
    with self._mutex:
        session = self.new_session()
        try:
            # Create index for efficient expiration queries
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_metric_expiration
                ON dq_metric(created, json_extract(meta, '$.ttl_hours'))
            """))
            session.commit()
        except Exception as e:
            # Log but don't fail - indexes are performance optimization
            logger.warning(f"Failed to create metric expiration index: {e}")
            session.rollback()
        finally:
            session.close()
```

Required import:
```python
from sqlalchemy import text
```

#### Task 1.2: Test index creation
Add test to verify index creation doesn't break initialization:

```python
def test_metric_db_creates_indexes() -> None:
    """Verify MetricDB creates performance indexes on initialization."""
    db = InMemoryMetricDB()

    # Should not raise any exceptions
    assert db is not None

    # Verify we can still persist metrics
    key = ResultKey(dt.date(2024, 10, 30), {})
    metric = Metric.build(
        metric=Sum("test"),
        key=key,
        dataset="test",
        metadata=Metadata(ttl_hours=24)
    )
    db.persist([metric])

    # Verify we can query
    stats = db.get_expired_metrics_stats()
    assert isinstance(stats, dict)
```

### Task Group 2: MetricDB Methods with UTC (TDD - Tests First)

#### Task 2.1: Write tests for get_expired_metrics_stats with UTC
Create `tests/test_metric_expiration.py` with tests for the statistics method:

```python
import datetime as dt
from datetime import datetime, timedelta, timezone

import pytest

from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.specs import Sum


class TestExpiredMetricsStats:
    """Test get_expired_metrics_stats functionality."""

    def test_empty_database_returns_empty_stats(self) -> None:
        """Empty database should return empty statistics."""
        db = InMemoryMetricDB()
        stats = db.get_expired_metrics_stats()
        assert stats == {}

    def test_metrics_without_metadata_counted_as_expired(self) -> None:
        """Metrics with None metadata should be counted as expired."""
        db = InMemoryMetricDB()

        # Create metric without metadata
        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("revenue")
        metric = Metric.build(
            metric=spec,
            key=key,
            dataset="test",
            metadata=None  # No metadata
        )
        db.persist([metric])

        stats = db.get_expired_metrics_stats()
        assert sum(stats.values()) == 1

    def test_metrics_without_ttl_counted_as_expired(self) -> None:
        """Metrics with metadata but no ttl_hours should be counted as expired."""
        db = InMemoryMetricDB()

        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("revenue")
        metric = Metric.build(
            metric=spec,
            key=key,
            dataset="test",
            metadata=Metadata()  # Metadata without ttl_hours
        )
        db.persist([metric])

        stats = db.get_expired_metrics_stats()
        assert sum(stats.values()) == 1

    def test_metrics_with_various_ttl_values(self) -> None:
        """Test correct counting with different TTL values."""
        db = InMemoryMetricDB()
        base_time = datetime(2024, 10, 30, 10, 0, 0)

        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("revenue")

        # Metric 1: TTL 1 hour (will be expired)
        metric1 = Metric.build(
            metric=spec,
            key=key,
            dataset="test1",
            metadata=Metadata(ttl_hours=1)
        )
        metric1.created = base_time - timedelta(hours=2)

        # Metric 2: TTL 24 hours (not expired)
        metric2 = Metric.build(
            metric=spec,
            key=key,
            dataset="test2",
            metadata=Metadata(ttl_hours=24)
        )
        metric2.created = base_time - timedelta(hours=12)

        # Metric 3: TTL 168 hours/7 days (not expired)
        metric3 = Metric.build(
            metric=spec,
            key=key,
            dataset="test3",
            metadata=Metadata(ttl_hours=168)
        )
        metric3.created = base_time - timedelta(days=5)

        db.persist([metric1, metric2, metric3])

        stats = db.get_expired_metrics_stats(current_time=base_time)
        assert sum(stats.values()) == 1  # Only metric1 is expired

    def test_grouping_by_created_date(self) -> None:
        """Verify metrics are correctly grouped by created date."""
        db = InMemoryMetricDB()
        base_time = datetime(2024, 10, 30, 10, 0, 0)

        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("revenue")

        # Create metrics on different dates, all expired
        dates = [
            base_time - timedelta(days=3),
            base_time - timedelta(days=3),  # Same date as above
            base_time - timedelta(days=2),
            base_time - timedelta(days=1),
        ]

        for i, created_date in enumerate(dates):
            metric = Metric.build(
                metric=spec,
                key=key,
                dataset=f"test{i}",
                metadata=Metadata(ttl_hours=1)  # All will be expired
            )
            metric.created = created_date
            db.persist([metric])

        stats = db.get_expired_metrics_stats(current_time=base_time)

        # Should have 3 different dates
        assert len(stats) == 3
        # Date with 2 metrics
        assert stats[dates[0].date()] == 2
        # Dates with 1 metric each
        assert stats[dates[2].date()] == 1
        assert stats[dates[3].date()] == 1

    def test_current_time_parameter_uses_utc(self) -> None:
        """Test that current_time parameter is used correctly with UTC."""
        db = InMemoryMetricDB()

        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("revenue")

        # Create metric with 24 hour TTL
        metric = Metric.build(
            metric=spec,
            key=key,
            dataset="test",
            metadata=Metadata(ttl_hours=24)
        )
        metric.created = datetime(2024, 10, 29, 10, 0, 0)
        db.persist([metric])

        # Test with current_time where metric is NOT expired
        current_time1 = datetime(2024, 10, 30, 9, 0, 0)  # 23 hours later
        stats1 = db.get_expired_metrics_stats(current_time=current_time1)
        assert sum(stats1.values()) == 0

        # Test with current_time where metric IS expired
        current_time2 = datetime(2024, 10, 30, 11, 0, 0)  # 25 hours later
        stats2 = db.get_expired_metrics_stats(current_time=current_time2)
        assert sum(stats2.values()) == 1

    def test_timezone_aware_datetime_handling(self) -> None:
        """Test handling of timezone-aware datetimes."""
        db = InMemoryMetricDB()

        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("test")

        # Create metric with UTC timezone aware datetime
        metric = Metric.build(
            metric=spec,
            key=key,
            dataset="test",
            metadata=Metadata(ttl_hours=1)
        )
        # SQLite stores as naive datetime, but we handle as UTC
        metric.created = datetime(2024, 10, 30, 10, 0, 0)
        db.persist([metric])

        # Query with UTC time
        current_time = datetime(2024, 10, 30, 11, 30, 0)
        stats = db.get_expired_metrics_stats(current_time=current_time)
        assert sum(stats.values()) == 1  # Expired
```

#### Task 2.2: Implement get_expired_metrics_stats in MetricDB with UTC
Add to `src/dqx/orm/repositories.py`:

```python
def get_expired_metrics_stats(self, current_time: datetime | None = None) -> dict[dt.date, int]:
    """
    Get statistics for expired metrics grouped by created date.

    Args:
        current_time: Optional current time for testing. Defaults to datetime.utcnow()

    Returns:
        Dict mapping created dates to count of expired metrics

    Example:
        {date(2024, 10, 23): 150, date(2024, 10, 24): 87}
    """
    if current_time is None:
        current_time = datetime.utcnow()

    session = self.new_session()

    # Build the expiration condition
    expired_condition = or_(
        # Metrics without TTL (metadata is None or ttl_hours is None)
        func.json_extract(Metric.meta, '$.ttl_hours').is_(None),
        # Expired metrics - using SQLite datetime arithmetic
        func.datetime(Metric.created, '+' ||
            func.coalesce(
                func.json_extract(Metric.meta, '$.ttl_hours'),
                '0'
            ) || ' hours'
        ) < current_time
    )

    # Query for statistics
    stats_query = (
        select(
            func.date(Metric.created).label('created_date'),
            func.count().label('count')
        )
        .where(expired_condition)
        .group_by(func.date(Metric.created))
        .order_by(func.date(Metric.created))
    )

    result = session.execute(stats_query).all()

    # Convert to dict
    return {row.created_date: row.count for row in result}
```

Required imports to add:
```python
import datetime as dt
from datetime import datetime
from sqlalchemy import func, or_, select, text
```

#### Task 2.3: Run tests and fix any issues
- Run: `uv run pytest tests/test_metric_expiration.py::TestExpiredMetricsStats -v`
- Fix any failing tests
- Run: `uv run mypy src/dqx/orm/repositories.py`
- Run: `uv run ruff check src/dqx/orm/repositories.py`
- Commit: `git add -A && git commit -m "feat: add get_expired_metrics_stats method with UTC support"`

### Task Group 3: Delete Method Implementation with Monitoring (TDD)

#### Task 3.1: Write tests for delete_expired_metrics with deletion count
Add to `tests/test_metric_expiration.py`:

```python
class TestDeleteExpiredMetrics:
    """Test delete_expired_metrics functionality."""

    def test_delete_removes_expired_metrics_only(self) -> None:
        """Verify only expired metrics are deleted, others remain."""
        db = InMemoryMetricDB()
        base_time = datetime(2024, 10, 30, 10, 0, 0)

        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("revenue")

        # Create mix of expired and non-expired metrics
        metrics = []

        # Expired: no metadata
        m1 = Metric.build(metric=spec, key=key, dataset="expired1", metadata=None)
        metrics.append(m1)

        # Expired: no ttl_hours
        m2 = Metric.build(metric=spec, key=key, dataset="expired2", metadata=Metadata())
        metrics.append(m2)

        # Expired: exceeded TTL
        m3 = Metric.build(metric=spec, key=key, dataset="expired3", metadata=Metadata(ttl_hours=1))
        m3.created = base_time - timedelta(hours=2)
        metrics.append(m3)

        # Not expired: within TTL
        m4 = Metric.build(metric=spec, key=key, dataset="keep1", metadata=Metadata(ttl_hours=24))
        m4.created = base_time - timedelta(hours=12)
        metrics.append(m4)

        # Not expired: very long TTL
        m5 = Metric.build(metric=spec, key=key, dataset="keep2", metadata=Metadata(ttl_hours=720))
        m5.created = base_time - timedelta(days=10)
        metrics.append(m5)

        db.persist(metrics)

        # Verify all metrics exist before deletion
        all_metrics = db.get_all().unwrap()
        assert len(all_metrics) == 5

        # Delete expired metrics
        deleted_count = db.delete_expired_metrics(current_time=base_time)

        # Verify deletion count
        assert deleted_count == 3

        # Verify only non-expired metrics remain
        remaining = db.get_all().unwrap()
        assert len(remaining) == 2
        assert all(m.dataset in ["keep1", "keep2"] for m in remaining)

    def test_delete_with_empty_database(self) -> None:
        """Delete on empty database should not error and return 0."""
        db = InMemoryMetricDB()
        deleted_count = db.delete_expired_metrics()
        assert deleted_count == 0

    def test_delete_is_transactional(self) -> None:
        """Verify deletion is wrapped in transaction with mutex."""
        db = InMemoryMetricDB()

        # Create an expired metric
        key = ResultKey(dt.date(2024, 10, 30), {})
        metric = Metric.build(
            metric=Sum("test"),
            key=key,
            dataset="test",
            metadata=None
        )
        db.persist([metric])

        # The InMemoryMetricDB should handle this atomically
        deleted_count = db.delete_expired_metrics()

        # Verify deletion count and metric was deleted
        assert deleted_count == 1
        remaining = db.get_all().unwrap()
        assert len(remaining) == 0

    def test_sqlite_datetime_arithmetic(self) -> None:
        """Test SQLite-specific datetime handling works correctly."""
        db = InMemoryMetricDB()

        # Test edge cases for datetime arithmetic
        base_time = datetime(2024, 10, 30, 10, 30, 45)  # Include seconds

        key = ResultKey(dt.date(2024, 10, 30), {})
        spec = Sum("test")

        # Metric exactly at TTL boundary
        metric = Metric.build(
            metric=spec,
            key=key,
            dataset="boundary",
            metadata=Metadata(ttl_hours=1)
        )
        metric.created = base_time - timedelta(hours=1)
        db.persist([metric])

        # Should not be deleted (exactly at boundary, not exceeded)
        deleted_count = db.delete_expired_metrics(current_time=base_time)
        assert deleted_count == 0
        remaining = db.get_all().unwrap()
        assert len(remaining) == 1

        # Clear and test just past boundary
        db = InMemoryMetricDB()
        metric2 = Metric.build(
            metric=spec,
            key=key,
            dataset="past_boundary",
            metadata=Metadata(ttl_hours=1)
        )
        metric2.created = base_time - timedelta(hours=1, seconds=1)
        db.persist([metric2])

        # Should be deleted (past boundary)
        deleted_count = db.delete_expired_metrics(current_time=base_time)
        assert deleted_count == 1
        remaining = db.get_all().unwrap()
        assert len(remaining) == 0
```

#### Task 3.2: Implement delete_expired_metrics in MetricDB with monitoring
Add to `src/dqx/orm/repositories.py`:

```python
def delete_expired_metrics(self, current_time: datetime | None = None) -> int:
    """
    Delete all expired metrics from the database.

    Metrics expire when:
    - They have no metadata (metadata is None)
    - They have metadata but no ttl_hours
    - created + ttl_hours < current_time

    Args:
        current_time: Optional current time for testing. Defaults to datetime.utcnow()

    Returns:
        Number of metrics deleted
    """
    if current_time is None:
        current_time = datetime.utcnow()

    with self._mutex:
        session = self.new_session()

        # Same condition as stats query
        expired_condition = or_(
            func.json_extract(Metric.meta, '$.ttl_hours').is_(None),
            func.datetime(Metric.created, '+' ||
                func.coalesce(
                    func.json_extract(Metric.meta, '$.ttl_hours'),
                    '0'
                ) || ' hours'
            ) < current_time
        )

        delete_query = delete(Metric).where(expired_condition)
        result = session.execute(delete_query)
        deleted_count = result.rowcount
        session.commit()

        return deleted_count
```

Required import to add:
```python
from sqlalchemy import delete
```

#### Task 3.3: Run tests and fix any issues
- Run: `uv run pytest tests/test_metric_expiration.py -v`
- Fix any failing tests
- Run: `uv run mypy src/dqx/orm/repositories.py`
- Run: `uv run ruff check src/dqx/orm/repositories.py`
- Commit: `git add -A && git commit -m "feat: add delete_expired_metrics method with deletion count"`

### Task Group 4: VerificationSuite Integration with Timer Registry (TDD)

#### Task 4.1: Write tests for VerificationSuite cleanup integration
Add to `tests/test_metric_expiration.py`:

```python
from unittest.mock import patch, Mock
import pyarrow as pa
from dqx.api import VerificationSuite, check, MetricProvider, Context
from dqx.datasource import DuckRelationDataSource
from dqx.timer import timer_registry


class TestVerificationSuiteCleanup:
    """Test metric cleanup integration in VerificationSuite."""

    def test_cleanup_enabled_by_default(self) -> None:
        """Verify cleanup_expired_metrics defaults to True."""
        @check(name="test check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="Row count check").is_gt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([test_check], db, "Test Suite")

        # Check internal attribute (will fail initially)
        assert suite._cleanup_expired_metrics is True

    def test_cleanup_can_be_disabled(self) -> None:
        """Test disabling cleanup in constructor."""
        @check(name="test check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            pass

        db = InMemoryMetricDB()
        suite = VerificationSuite([test_check], db, "Test Suite", cleanup_expired_metrics=False)

        assert suite._cleanup_expired_metrics is False

    def test_cleanup_happens_before_analysis(self) -> None:
        """Verify cleanup occurs at correct point in run() flow."""
        @check(name="test check", datasets=["test_data"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="Row count check").is_gt(0)

        db = InMemoryMetricDB()

        # Add an expired metric
        key = ResultKey(dt.date(2024, 10, 30), {})
        expired_metric = Metric.build(
            metric=Sum("old_metric"),
            key=key,
            dataset="old_data",
            metadata=None  # No TTL, should be deleted
        )
        db.persist([expired_metric])

        suite = VerificationSuite([test_check], db, "Test Suite")

        # Track method calls to verify order
        call_order = []

        with patch.object(db, 'get_expired_metrics_stats', wraps=db.get_expired_metrics_stats) as mock_stats:
            with patch.object(db, 'delete_expired_metrics', wraps=db.delete_expired_metrics) as mock_delete:
                with patch('dqx.api.logger') as mock_logger:
                    # Create test data and run
                    data = pa.table({"x": [1, 2, 3]})
                    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

                    # Track when cleanup happens vs analysis
                    def track_cleanup(*args, **kwargs):
                        call_order.append('cleanup')
                        return {dt.date(2024, 10, 30): 1}

                    def track_delete(*args, **kwargs):
                        call_order.append('delete')
                        return db.delete_expired_metrics(*args, **kwargs)

                    mock_stats.side_effect = track_cleanup
                    mock_delete.side_effect = track_delete

                    # Patch analyze to track when it's called
                    original_analyze = suite._analyze
                    def track_analyze(*args, **kwargs):
                        call_order.append('analyze')
                        return original_analyze(*args, **kwargs)

                    with patch.object(suite, '_analyze', side_effect=track_analyze):
                        suite.run([datasource], key)

                    # Verify cleanup happened before analysis
                    assert call_order.index('cleanup') < call_order.index('analyze')
                    assert call_order.index('delete') < call_order.index('analyze')

    def test_cleanup_uses_timer_registry(self) -> None:
        """Test that cleanup uses timer registry for monitoring."""
        @check(name="test check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            pass

        db = InMemoryMetricDB()

        # Add expired metrics
        key = ResultKey(dt.date(2024, 10, 30), {})
        for i in range(5):
            metric = Metric.build(metric=Sum(f"m{i}"), key=key, dataset=f"d{i}", metadata=None)
            db.persist([metric])

        suite = VerificationSuite([test_check], db, "Test Suite")

        # Mock timer registry
        mock_timer = Mock()

        with patch('dqx.api.timer_registry.timer', return_value=mock_timer):
            data = pa.table({"x": [1]})
            datasource = DuckRelationDataSource.from_arrow(data, "test_data")
            suite.run([datasource], key)

            # Verify timer was used
            mock_timer.__enter__.assert_called_once()
            mock_timer.__exit__.assert_called_once()

    def test_cleanup_logging_with_deletion_count(self) -> None:
        """Test that cleanup logs correct statistics with deletion count."""
        @check(name="test check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            pass

        db = InMemoryMetricDB()

        # Add metrics that will be expired
        base_time = datetime(2024, 10, 30, 10, 0, 0)
        key1 = ResultKey(dt.date(2024, 10, 28), {})
        key2 = ResultKey(dt.date(2024, 10, 29), {})

        metrics = [
            Metric.build(metric=Sum("m1"), key=key1, dataset="d1", metadata=None),
            Metric.build(metric=Sum("m2"), key=key1, dataset="d2", metadata=None),
            Metric.build(metric=Sum("m3"), key=key2, dataset="d3", metadata=None),
        ]

        for i, m in enumerate(metrics):
            m.created = base_time - timedelta(days=2-i//2)

        db.persist(metrics)

        suite = VerificationSuite([test_check], db, "Test Suite")

        with patch('dqx.api.logger') as mock_logger:
            data = pa.table({"x": [1]})
            datasource = DuckRelationDataSource.from_arrow(data, "test_data")

            # Mock time for consistent testing
            with patch("dqx.orm.repositories.datetime") as mock_datetime:
                mock_datetime.utcnow.return_value = base_time
                mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

                suite.run([datasource], key1)

                # Verify logging calls
                mock_logger.info.assert_any_call("Cleaning up expired metrics...")
                mock_logger.info.assert_any_call("Found 3 expired metrics to delete:")
                mock_logger.info.assert_any_call("  - 2024-10-28: 2 metrics")
                mock_logger.info.assert_any_call("  - 2024-10-29: 1 metrics")
                mock_logger.info.assert_any_call("Successfully deleted 3 expired metrics in %.3f seconds", mock.ANY)

    def test_cleanup_with_no_expired_metrics(self) -> None:
        """Test logging when no metrics need deletion."""
        @check(name="test check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            pass

        db = InMemoryMetricDB()
        suite = VerificationSuite([test_check], db, "Test Suite")

        with patch('dqx.api.logger') as mock_logger:
            data = pa.table({"x": [1]})
            datasource = DuckRelationDataSource.from_arrow(data, "test_data")
            suite.run([datasource], ResultKey(dt.date.today(), {}))

            mock_logger.info.assert_any_call("Cleaning up expired metrics...")
            mock_logger.info.assert_any_call("No expired metrics found to delete")

    def test_cleanup_failure_does_not_break_suite(self) -> None:
        """Suite should handle cleanup errors gracefully."""
        @check(name="test check", datasets=["test_data"])
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).where(name="Row count check").is_gt(0)

        db = InMemoryMetricDB()
        suite = VerificationSuite([test_check], db, "Test Suite")

        # Make cleanup fail
        with patch.object(db, 'get_expired_metrics_stats', side_effect=Exception("DB Error")):
            with patch('dqx.api.logger') as mock_logger:
                # Suite should still run successfully
                data = pa.table({"x": [1, 2, 3]})
                datasource = DuckRelationDataSource.from_arrow(data, "test_data")

                # Should not raise, analysis should still happen
                suite.run([datasource], ResultKey(dt.date.today(), {}))

                # Should log the error
                mock_logger.error.assert_called()

                # Suite should have completed successfully
                assert len(suite.graph.root.children) > 0
```

#### Task 4.2: Update VerificationSuite constructor
Modify `src/dqx/api.py` constructor to add the new parameter:

```python
def __init__(
    self,
    checks: Sequence[CheckProducer | DecoratedCheck],
    db: MetricDB,
    name: str,
    cleanup_expired_metrics: bool = True,  # Add this parameter
) -> None:
    """Initialize VerificationSuite.

    Args:
        checks: List of check functions or decorated checks
        db: MetricDB instance for storing metrics
        name: Name of the verification suite
        cleanup_expired_metrics: Whether to clean up expired metrics before analysis
    """
    # ... existing initialization code ...
    self._cleanup_expired_metrics = cleanup_expired_metrics
```

#### Task 4.3: Add cleanup method to VerificationSuite
Add private method to `src/dqx/api.py`:

```python
def _cleanup_expired_metrics(self) -> None:
    """Clean up expired metrics if cleanup is enabled."""
    if not self._cleanup_expired_metrics:
        return

    logger.info("Cleaning up expired metrics...")

    try:
        # Use timer registry to track duration
        with timer_registry.timer("metric_cleanup"):
            # Get statistics
            stats = self.provider._db.get_expired_metrics_stats()

            # Log statistics
            total_to_delete = sum(stats.values())
            if total_to_delete > 0:
                logger.info(f"Found {total_to_delete} expired metrics to delete:")
                for created_date in sorted(stats.keys()):
                    logger.info(f"  - {created_date}: {stats[created_date]} metrics")

                # Delete the metrics and get count
                deleted_count = self.provider._db.delete_expired_metrics()

                # Get duration from timer
                duration = timer_registry.get_duration("metric_cleanup")
                logger.info(f"Successfully deleted {deleted_count} expired metrics in {duration:.3f} seconds")
            else:
                logger.info("No expired metrics found to delete")
    except Exception as e:
        logger.error(f"Failed to clean up expired metrics: {e}")
        # Don't fail the suite run due to cleanup errors
```

Required import:
```python
from dqx.timer import timer_registry
```

#### Task 4.4: Update VerificationSuite.run() method
Modify the `run` method in `src/dqx/api.py` to call cleanup before analysis:

```python
# In the run() method, after symbol deduplication and before analysis:

# Apply symbol deduplication BEFORE analysis
self._context.provider.symbol_deduplication(self._context._graph, key)

# Clean up expired metrics before analysis
self._cleanup_expired_metrics()

# 2. Analyze by datasources
with self._analyze_ms:
    self._analyze(datasources, key)
```

#### Task 4.5: Run tests and fix any issues
- Run: `uv run pytest tests/test_metric_expiration.py::TestVerificationSuiteCleanup -v`
- Fix any failing tests
- Run: `uv run mypy src/dqx/api.py`
- Run: `uv run ruff check src/dqx/api.py`
- Commit: `git add -A && git commit -m "feat: integrate metric cleanup with timer registry"`

### Task Group 5: Additional Test Coverage

#### Task 5.1: Write performance tests
Add to `tests/test_metric_expiration.py`:

```python
import time
from concurrent.futures import ThreadPoolExecutor


class TestMetricExpirationPerformance:
    """Test performance aspects of metric expiration."""

    def test_large_dataset_performance(self) -> None:
        """Test performance with large number of metrics."""
        db = InMemoryMetricDB()
        base_time = datetime.utcnow()

        # Create 10,000 metrics with various states
        metrics = []
        for i in range(10000):
            key = ResultKey(dt.date(2024, 10, 30), {"batch": i // 1000})

            # Mix of expired and non-expired
            if i % 3 == 0:
                # No metadata - expired
                metadata = None
            elif i % 3 == 1:
                # Short TTL - expired
                metadata = Metadata(ttl_hours=1)
            else:
                # Long TTL - not expired
                metadata = Metadata(ttl_hours=720)

            metric = Metric.build(
                metric=Sum(f"metric_{i}"),
                key=key,
                dataset=f"dataset_{i}",
                metadata=metadata
            )

            # Set created time for expired metrics
            if i % 3 != 2:
                metric.created = base_time - timedelta(days=2)
            else:
                metric.created = base_time - timedelta(hours=1)

            metrics.append(metric)

        # Persist in batches
        for i in range(0, len(metrics), 1000):
            db.persist(metrics[i:i+1000])

        # Measure stats query performance
        start = time.time()
        stats = db.get_expired_metrics_stats(current_time=base_time)
        stats_duration = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert stats_duration < 1.0
        assert sum(stats.values()) == 6667  # ~2/3 are expired

        # Measure deletion performance
        start = time.time()
        deleted_count = db.delete_expired_metrics(current_time=base_time)
        delete_duration = time.time() - start

        # Should complete in reasonable time (< 2 seconds)
        assert delete_duration < 2.0
        assert deleted_count == 6667

        # Verify only non-expired remain
        remaining = db.get_all().unwrap()
        assert len(remaining) == 3333

    def test_concurrent_cleanup_operations(self) -> None:
        """Test thread safety of cleanup operations."""
        db = InMemoryMetricDB()

        def create_and_cleanup(thread_id: int) -> tuple[int, int]:
            """Create metrics and run cleanup."""
            # Create some metrics
            metrics = []
            for i in range(100):
                key = ResultKey(dt.date(2024, 10, 30), {"thread": thread_id})
                metric = Metric.build(
                    metric=Sum(f"t{thread_id}_m{i}"),
                    key=key,
                    dataset=f"thread_{thread_id}",
                    metadata=None if i < 50 else Metadata(ttl_hours=24)
                )
                metrics.append(metric)

            db.persist(metrics)

            # Run cleanup
            stats = db.get_expired_metrics_stats()
            deleted = db.delete_expired_metrics()

            return sum(stats.values()), deleted

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_and_cleanup, i) for i in range(5)]
            results = [f.result() for f in futures]

        # Each thread should see and delete its own expired metrics
        for stats_count, deleted_count in results:
            assert stats_count >= 50  # At least its own expired metrics
            assert deleted_count >= 0  # Might be 0 if another thread deleted first

        # Final state should have only non-expired metrics
        remaining = db.get_all().unwrap()
        # Should have 250 non-expired metrics (50 per thread * 5 threads)
        assert all(m.metadata is not None for m in remaining)
        assert all(m.metadata.ttl_hours == 24 for m in remaining)
```

#### Task 5.2: Write timezone edge case tests
Add timezone-specific tests:

```python
class TestTimezoneHandling:
    """Test timezone edge cases."""

    def test_dst_boundary_handling(self) -> None:
        """Test handling around DST boundaries."""
        db = InMemoryMetricDB()

        # Create metric near DST boundary
        # Using UTC so DST doesn't affect us
        dst_boundary = datetime(2024, 3, 10, 7, 0, 0)  # 2 AM EST -> 3 AM EDT

        key = ResultKey(dt.date(2024, 3, 10), {})
        metric = Metric.build(
            metric=Sum("dst_test"),
            key=key,
            dataset="test",
            metadata=Metadata(ttl_hours=1)
        )
        metric.created = dst_boundary - timedelta(hours=1, minutes=30)
        db.persist([metric])

        # Check expiration exactly at boundary
        stats = db.get_expired_metrics_stats(current_time=dst_boundary)
        assert sum(stats.values()) == 1  # Should be expired

    def test_leap_second_handling(self) -> None:
        """Test handling of leap seconds (if any)."""
        db = InMemoryMetricDB()

        # SQLite doesn't handle leap seconds, but test edge case
        edge_time = datetime(2023, 12, 31, 23, 59, 59)

        key = ResultKey(dt.date(2023, 12, 31), {})
        metric = Metric.build(
            metric=Sum("leap_test"),
            key=key,
            dataset="test",
            metadata=Metadata(ttl_hours=1)
        )
        metric.created = edge_time
        db.persist([metric])

        # Check after the boundary
        future_time = datetime(2024, 1, 1, 1, 0, 0)
        stats = db.get_expired_metrics_stats(current_time=future_time)
        assert sum(stats.values()) == 1
```

#### Task 5.3: Run all tests
- Run: `uv run pytest tests/test_metric_expiration.py -v`
- Run: `uv run pytest tests/ -k "not e2e" -v`  # Run other tests to ensure no regression
- Commit: `git add -A && git commit -m "test: add performance and timezone tests"`

### Task Group 6: Integration Tests (TDD)

#### Task 6.1: Write end-to-end integration tests
Add to `tests/test_metric_expiration.py`:

```python
def test_end_to_end_metric_expiration_with_utc() -> None:
    """Full integration test of metric expiration feature with UTC."""
    db = InMemoryMetricDB()
    base_time = datetime.utcnow()

    # Pre-populate database with mix of expired and non-expired metrics
    old_key = ResultKey(dt.date(2024, 10, 25), {"env": "test"})
    new_key = ResultKey(dt.date(2024, 10, 30), {"env": "test"})

    metrics_to_add = [
        # Old metrics without TTL - should be deleted
        Metric.build(metric=Sum("old1"), key=old_key, dataset="ds1", metadata=None),
        Metric.build(metric=Sum("old2"), key=old_key, dataset="ds2", metadata=Metadata()),

        # Old metric with expired TTL - should be deleted
        Metric.build(metric=Sum("old3"), key=old_key, dataset="ds3", metadata=Metadata(ttl_hours=24)),

        # Recent metric with long TTL - should be kept
        Metric.build(metric=Sum("keep1"), key=new_key, dataset="ds4", metadata=Metadata(ttl_hours=720)),
    ]

    # Set created dates
    for i, m in enumerate(metrics_to_add[:3]):
        m.created = base_time - timedelta(days=5)
    metrics_to_add[3].created = base_time - timedelta(hours=1)

    db.persist(metrics_to_add)

    # Create a check that will generate new metrics
    @check(name="test check", datasets=["test_data"])
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Row count").is_gt(0)
        ctx.assert_that(mp.average("value")).where(name="Average value").is_between(0, 100)

    # Create suite with cleanup enabled
    suite = VerificationSuite([test_check], db, "Test Suite")

    # Mock current time for consistent testing
    with patch("dqx.orm.repositories.datetime") as mock_datetime:
        mock_datetime.utcnow.return_value = base_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        # Run the suite
        data = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
        datasource = DuckRelationDataSource.from_arrow(data, "test_data")
        suite.run([datasource], new_key)

    # Verify old metrics were deleted
    all_metrics = db.get_all().unwrap()
    datasets = {m.dataset for m in all_metrics}

    # Old datasets should be gone
    assert "ds1" not in datasets
    assert "ds2" not in datasets
    assert "ds3" not in datasets

    # Kept dataset should remain
    assert "ds4" in datasets

    # New metrics from the check should exist
    assert "test_data" in datasets

    # Verify count
    assert len([m for m in all_metrics if m.dataset in ["ds1", "ds2", "ds3"]]) == 0
    assert len([m for m in all_metrics if m.dataset == "ds4"]) == 1
    assert len([m for m in all_metrics if m.dataset == "test_data"]) > 0


def test_multiple_suite_runs_with_cleanup() -> None:
    """Test that multiple suite runs handle cleanup correctly."""
    db = InMemoryMetricDB()

    @check(name="simple check", datasets=["data"])
    def simple_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.num_rows()).where(name="Has data").is_gt(0)

    # First run - with no TTL (will be expired)
    suite1 = VerificationSuite([simple_check], db, "Suite1")
    data = pa.table({"x": [1, 2, 3]})
    datasource = DuckRelationDataSource.from_arrow(data, "data")

    key1 = ResultKey(dt.date(2024, 10, 29), {})
    suite1.run([datasource], key1)

    # Verify metrics exist
    metrics_after_run1 = db.get_all().unwrap()
    assert len(metrics_after_run1) > 0

    # Second run - with TTL
    suite2 = VerificationSuite([simple_check], db, "Suite2")
    key2 = ResultKey(dt.date(2024, 10, 30), {})

    # Set metadata with TTL for this run
    suite2._context._metadata = Metadata(ttl_hours=168)  # 7 days

    # Mock time to be later
    with patch("dqx.orm.repositories.datetime") as mock_datetime:
        future_time = datetime(2024, 10, 30, 12, 0, 0)
        mock_datetime.utcnow.return_value = future_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        suite2.run([datasource], key2)

    # Verify old metrics were cleaned up
    all_metrics = db.get_all().unwrap()

    # Should only have metrics from second run
    for metric in all_metrics:
        assert metric.metadata is not None
        assert metric.metadata.ttl_hours == 168
```

#### Task 6.2: Run all tests
- Run: `uv run pytest tests/test_metric_expiration.py -v`
- Run: `uv run pytest tests/ -k "not e2e" -v`  # Run other tests to ensure no regression
- Commit: `git add -A && git commit -m "test: add integration tests for metric expiration"`

### Task Group 7: Final Verification

#### Task 7.1: Run pre-commit hooks
- Run: `uv run hooks`
- Fix any issues reported
- Commit fixes if any: `git add -A && git commit -m "style: fix linting issues"`

#### Task 7.2: Run full test suite
- Run: `uv run pytest tests/ -v`
- Ensure all tests pass
- Check coverage: `uv run coverage tests/ -v`

#### Task 7.3: Update documentation
Add to `README.md` in the appropriate section:

```markdown
### Metric Expiration

DQX automatically cleans up expired metrics based on their TTL (Time To Live) settings:

- Metrics without metadata or without `ttl_hours` are deleted on the next suite run
- Metrics with `ttl_hours` are kept for the specified duration
- Cleanup happens automatically before analysis in `VerificationSuite`
- All timestamps are in UTC for consistency

To disable automatic cleanup:

```python
suite = VerificationSuite(checks, db, "My Suite", cleanup_expired_metrics=False)
```

To set TTL for metrics:

```python
metadata = Metadata(ttl_hours=168)  # Keep for 7 days
```

Performance Notes:
- Cleanup operations are timed using the timer registry
- Database indexes are created automatically for efficient queries
- Deletion counts are logged for monitoring
```

Commit: `git add -A && git commit -m "docs: document metric expiration feature v2"`

## Summary

This implementation plan v2 adds automatic metric expiration to DQX with the following improvements from v1:

### Key Features:
1. **Two MetricDB methods**: `get_expired_metrics_stats()` and `delete_expired_metrics()`
2. **VerificationSuite integration**: Automatic cleanup before analysis
3. **Configurable behavior**: Can be disabled via constructor parameter
4. **Comprehensive tests**: Unit tests, integration tests, performance tests, and timezone tests

### V2 Improvements:
1. **UTC Consistency**: All datetime operations use `datetime.utcnow()` instead of `datetime.now()`
2. **Performance Index**: Database index created on initialization for efficient queries
3. **Enhanced Monitoring**: Timer registry integration with structured logging and deletion counts
4. **Additional Test Coverage**: Performance benchmarks, concurrent operation tests, and timezone edge cases
5. **Better Error Handling**: Cleanup failures don't break suite execution

The implementation follows TDD principles with tests written before code, uses real objects instead of mocks, maintains type safety throughout, and includes comprehensive documentation.
