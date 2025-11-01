"""Tests for MetricCache functionality."""

from datetime import date, timedelta

import pytest
from returns.maybe import Maybe, Nothing, Some

from dqx.cache import MetricCache
from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.specs import MetricSpec, Sum
from dqx.states import SimpleAdditiveState


@pytest.fixture
def db() -> InMemoryMetricDB:
    """Create in-memory database for testing."""
    return InMemoryMetricDB()


@pytest.fixture
def cache(db: InMemoryMetricDB) -> MetricCache:
    """Create cache with DB."""
    return MetricCache(db)


@pytest.fixture
def sample_metric() -> Metric:
    """Create a sample metric."""
    return Metric.build(
        metric=Sum("revenue"),
        key=ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={"env": "test"}),
        dataset="sales",
        state=SimpleAdditiveState(value=100.0),
        metadata=Metadata(execution_id="exec-123"),
    )


class TestMetricCache:
    """Tests for MetricCache class."""

    def test_get_miss_returns_nothing(self, cache: MetricCache) -> None:
        """Test cache miss returns Nothing."""
        key = (Sum("revenue"), ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}), "sales", "exec-123")
        result = cache.get(key)
        assert result == Nothing

    def test_put_and_get_single_metric(self, cache: MetricCache, sample_metric: Metric) -> None:
        """Test putting and getting single metric."""
        # Put metric
        cache.put(sample_metric)

        # Get metric
        assert sample_metric.metadata is not None
        assert sample_metric.metadata.execution_id is not None
        key = (
            sample_metric.spec,
            sample_metric.key,
            sample_metric.dataset,
            sample_metric.metadata.execution_id,
        )
        result = cache.get(key)

        assert isinstance(result, Some)
        assert result.unwrap() == sample_metric

    def test_put_sequence_of_metrics(self, cache: MetricCache) -> None:
        """Test putting multiple metrics at once."""
        metrics = []
        for i in range(3):
            metric = Metric.build(
                metric=Sum("revenue"),
                key=ResultKey(yyyy_mm_dd=date(2024, 1, 10 + i), tags={}),
                dataset="sales",
                state=SimpleAdditiveState(value=100.0 + i),
                metadata=Metadata(execution_id="exec-123"),
            )
            metrics.append(metric)

        # Put all metrics
        cache.put(metrics)

        # Verify all are cached
        for metric in metrics:
            key = (metric.spec, metric.key, metric.dataset, "exec-123")
            result = cache.get(key)
            assert isinstance(result, Some)

    def test_get_from_db_on_miss(self, db: InMemoryMetricDB, cache: MetricCache, sample_metric: Metric) -> None:
        """Test cache fetches from DB on miss."""
        # Add to DB but not cache
        db.persist([sample_metric])

        # Get from cache (should fetch from DB)
        assert sample_metric.metadata is not None
        assert sample_metric.metadata.execution_id is not None
        key = (
            sample_metric.spec,
            sample_metric.key,
            sample_metric.dataset,
            sample_metric.metadata.execution_id,
        )
        match cache.get(key):
            case Some(cached_metric):
                assert cached_metric.value == sample_metric.value
            case _:
                pytest.fail("Expected to find metric in DB")

        # Second get should use cache
        result2 = cache.get(key)
        assert isinstance(result2, Some)

    def test_execution_id_mismatch(self, db: InMemoryMetricDB, cache: MetricCache, sample_metric: Metric) -> None:
        """Test cache doesn't return metric with wrong execution_id."""
        # Add to DB
        db.persist([sample_metric])

        # Try to get with different execution_id
        key = (
            sample_metric.spec,
            sample_metric.key,
            sample_metric.dataset,
            "different-exec-456",  # Different execution ID
        )
        result = cache.get(key)

        assert result == Nothing

    def test_get_window(self, db: InMemoryMetricDB, cache: MetricCache) -> None:
        """Test getting a window of metrics."""
        # Create metrics for multiple days
        metrics = []
        base_date = date(2024, 1, 10)
        for i in range(5):
            metric = Metric.build(
                metric=Sum("revenue"),
                key=ResultKey(yyyy_mm_dd=base_date - timedelta(days=i), tags={}),
                dataset="sales",
                state=SimpleAdditiveState(value=100.0 + i),
                metadata=Metadata(execution_id="exec-123"),
            )
            metrics.append(metric)

        # Add to cache
        cache.put(metrics)

        # Get window
        time_series = cache.get_window(Sum("revenue"), ResultKey(yyyy_mm_dd=base_date, tags={}), "sales", "exec-123", 5)

        assert len(time_series) == 5
        assert time_series[base_date] == 100.0
        assert time_series[base_date - timedelta(days=4)] == 104.0

    def test_clear(self, cache: MetricCache, sample_metric: Metric) -> None:
        """Test clearing cache."""
        # Add metric
        cache.put(sample_metric)

        # Clear cache
        cache.clear()

        # Verify cleared
        assert sample_metric.metadata is not None
        assert sample_metric.metadata.execution_id is not None
        key = (
            sample_metric.spec,
            sample_metric.key,
            sample_metric.dataset,
            sample_metric.metadata.execution_id,
        )
        result = cache.get(key)
        assert result == Nothing

    def test_thread_safety(self, cache: MetricCache) -> None:
        """Test cache is thread-safe."""
        import threading

        results = []

        def put_and_get() -> None:
            metric = Metric.build(
                metric=Sum("revenue"),
                key=ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}),
                dataset="sales",
                state=SimpleAdditiveState(value=100.0),
                metadata=Metadata(execution_id="exec-123"),
            )
            cache.put(metric)

            key = (metric.spec, metric.key, metric.dataset, "exec-123")
            result = cache.get(key)
            results.append(isinstance(result, Some))

        # Run in multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=put_and_get)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should succeed
        assert all(results)

    def test_dirty_tracking(self, cache: MetricCache, sample_metric: Metric) -> None:
        """Test dirty metric tracking."""
        # Put as dirty
        cache.put(sample_metric, mark_dirty=True)

        key = cache._build_key(sample_metric)
        assert cache.is_dirty(key)
        assert cache.get_dirty_count() == 1

        # Flush to DB
        flushed = cache.flush_dirty()
        assert flushed == 1
        assert cache.get_dirty_count() == 0
        assert not cache.is_dirty(key)

    def test_put_clean_removes_dirty_flag(self, cache: MetricCache, sample_metric: Metric) -> None:
        """Test putting clean metric removes dirty flag."""
        # Put as dirty
        cache.put(sample_metric, mark_dirty=True)
        assert cache.get_dirty_count() == 1

        # Put same metric as clean
        cache.put(sample_metric, mark_dirty=False)
        assert cache.get_dirty_count() == 0

    def test_flush_empty_dirty_set(self, cache: MetricCache) -> None:
        """Test flushing with no dirty metrics."""
        assert cache.flush_dirty() == 0

    def test_get_window_with_missing_dates(self, cache: MetricCache) -> None:
        """Test get_window handles missing dates gracefully."""
        # Only add metrics for some days
        base_date = date(2024, 1, 10)
        metric1 = Metric.build(
            metric=Sum("revenue"),
            key=ResultKey(yyyy_mm_dd=base_date, tags={}),
            dataset="sales",
            state=SimpleAdditiveState(value=100.0),
            metadata=Metadata(execution_id="exec-123"),
        )
        metric2 = Metric.build(
            metric=Sum("revenue"),
            key=ResultKey(yyyy_mm_dd=base_date - timedelta(days=2), tags={}),
            dataset="sales",
            state=SimpleAdditiveState(value=102.0),
            metadata=Metadata(execution_id="exec-123"),
        )
        cache.put([metric1, metric2])

        # Get window including missing dates
        time_series = cache.get_window(Sum("revenue"), ResultKey(yyyy_mm_dd=base_date, tags={}), "sales", "exec-123", 5)

        # Should only have 2 values
        assert len(time_series) == 2
        assert time_series[base_date] == 100.0
        assert time_series[base_date - timedelta(days=2)] == 102.0

    def test_build_key_with_missing_metadata(self, cache: MetricCache) -> None:
        """Test _build_key raises ValueError for metric without metadata."""
        metric = Metric.build(
            metric=Sum("revenue"),
            key=ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}),
            dataset="sales",
            state=SimpleAdditiveState(value=100.0),
            metadata=None,
        )

        with pytest.raises(ValueError, match="Metric missing metadata"):
            cache._build_key(metric)

    def test_build_key_with_missing_execution_id(self, cache: MetricCache) -> None:
        """Test _build_key raises ValueError for metric without execution_id."""
        metric = Metric.build(
            metric=Sum("revenue"),
            key=ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}),
            dataset="sales",
            state=SimpleAdditiveState(value=100.0),
            metadata=Metadata(execution_id=None),
        )

        with pytest.raises(ValueError, match="Metric missing metadata or execution_id"):
            cache._build_key(metric)

    def test_db_value_without_full_metric(self) -> None:
        """Test warning when DB has value but not full metric."""

        # Create a custom DB that returns a value but no metrics
        class InconsistentDB(InMemoryMetricDB):
            def get_metric_value(
                self, metric_spec: MetricSpec, result_key: ResultKey, dataset: str, execution_id: str
            ) -> Maybe[float]:
                # Return a value
                return Some(100.0)

            def get_by_execution_id(self, execution_id: str) -> list[Metric]:
                # Return empty list - no metrics found
                return []

        # Create cache with the inconsistent DB
        inconsistent_db = InconsistentDB()
        cache = MetricCache(inconsistent_db)

        # Try to get a metric - should trigger the warning path
        key = (Sum("revenue"), ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}), "sales", "exec-123")
        result = cache.get(key)

        # Should return Nothing since full metric not found
        assert result == Nothing
