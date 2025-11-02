"""Tests for MetricCache functionality."""

from datetime import date, timedelta

import pytest
from returns.maybe import Nothing, Some

from dqx.cache import CacheStats, MetricCache
from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.specs import Sum
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
                # Handle both Metric objects and raw float values
                if isinstance(cached_metric, Metric):
                    assert cached_metric.value == sample_metric.value
                else:
                    assert cached_metric == sample_metric.value
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
        time_series = cache.get_timeseries(
            Sum("revenue"), ResultKey(yyyy_mm_dd=base_date, tags={}), "sales", "exec-123", 5
        )

        assert len(time_series) == 5
        assert time_series[base_date].value == 100.0
        assert time_series[base_date - timedelta(days=4)].value == 104.0

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
        flushed = cache.write_back()
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
        assert cache.write_back() == 0

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
        time_series = cache.get_timeseries(
            Sum("revenue"), ResultKey(yyyy_mm_dd=base_date, tags={}), "sales", "exec-123", 5
        )

        # Should only have 2 values
        assert len(time_series) == 2
        assert time_series[base_date].value == 100.0
        assert time_series[base_date - timedelta(days=2)].value == 102.0

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

        with pytest.raises(ValueError, match="Metric missing execution_id"):
            cache._build_key(metric)

    def test_db_value_without_full_metric(self, db: InMemoryMetricDB, cache: MetricCache) -> None:
        """Test when DB doesn't have the requested metric."""
        # Try to get a metric that doesn't exist
        key = (Sum("revenue"), ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}), "sales", "exec-123")
        result = cache.get(key)

        # Should return Nothing since metric doesn't exist
        assert result == Nothing

    def test_get_sparse_timeseries(self, cache: MetricCache) -> None:
        """Test getting sparse time series with specific lags."""
        # Create metrics for specific lags
        base_date = date(2024, 1, 10)
        lags = [0, 2, 5, 7]  # Sparse lags

        metrics = []
        for lag in lags:
            metric = Metric.build(
                metric=Sum("revenue"),
                key=ResultKey(yyyy_mm_dd=base_date - timedelta(days=lag), tags={}),
                dataset="sales",
                state=SimpleAdditiveState(value=100.0 + lag),
                metadata=Metadata(execution_id="exec-123"),
            )
            metrics.append(metric)

        # Add to cache
        cache.put(metrics)

        # Get sparse time series
        time_series = cache.get_sparse_timeseries(
            Sum("revenue"),
            ResultKey(yyyy_mm_dd=base_date, tags={}),
            "sales",
            "exec-123",
            lags=lags,
        )

        # Should have exactly the requested lags
        assert len(time_series) == len(lags)
        assert time_series[base_date].value == 100.0
        assert time_series[base_date - timedelta(days=2)].value == 102.0
        assert time_series[base_date - timedelta(days=5)].value == 105.0
        assert time_series[base_date - timedelta(days=7)].value == 107.0

    def test_get_sparse_timeseries_with_missing_values(self, cache: MetricCache) -> None:
        """Test sparse time series handles missing values gracefully."""
        base_date = date(2024, 1, 10)

        # Only add metrics for some of the requested lags
        metric1 = Metric.build(
            metric=Sum("revenue"),
            key=ResultKey(yyyy_mm_dd=base_date, tags={}),
            dataset="sales",
            state=SimpleAdditiveState(value=100.0),
            metadata=Metadata(execution_id="exec-123"),
        )
        metric2 = Metric.build(
            metric=Sum("revenue"),
            key=ResultKey(yyyy_mm_dd=base_date - timedelta(days=5), tags={}),
            dataset="sales",
            state=SimpleAdditiveState(value=105.0),
            metadata=Metadata(execution_id="exec-123"),
        )
        cache.put([metric1, metric2])

        # Request more lags than available
        requested_lags = [0, 2, 5, 7, 10]
        time_series = cache.get_sparse_timeseries(
            Sum("revenue"),
            ResultKey(yyyy_mm_dd=base_date, tags={}),
            "sales",
            "exec-123",
            lags=requested_lags,
        )

        # Should only have 2 values
        assert len(time_series) == 2
        assert time_series[base_date].value == 100.0
        assert time_series[base_date - timedelta(days=5)].value == 105.0
        # Missing lags should not be in the result
        assert (base_date - timedelta(days=2)) not in time_series
        assert (base_date - timedelta(days=7)) not in time_series
        assert (base_date - timedelta(days=10)) not in time_series

    def test_get_sparse_timeseries_empty_lags(self, cache: MetricCache) -> None:
        """Test sparse time series with empty lags list."""
        time_series = cache.get_sparse_timeseries(
            Sum("revenue"),
            ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}),
            "sales",
            "exec-123",
            lags=[],
        )

        # Should return empty time series
        assert len(time_series) == 0


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_cache_stats_hit_ratio(self) -> None:
        """Test CacheStats hit ratio calculation."""
        # Test all edge cases in one test
        assert CacheStats().hit_ratio() == 0.0
        assert CacheStats(hit=10, missed=0).hit_ratio() == 1.0
        assert CacheStats(hit=3, missed=7).hit_ratio() == 0.3
        assert CacheStats(hit=0, missed=10).hit_ratio() == 0.0

    def test_get_stats_tracking(self, cache: MetricCache, db: InMemoryMetricDB, sample_metric: Metric) -> None:
        """Test stats tracking for hits, misses, and DB fetches."""
        # Initial state
        stats = cache.get_stats()
        assert stats == CacheStats()

        # Cache miss
        key = (Sum("revenue"), ResultKey(yyyy_mm_dd=date(2024, 1, 10), tags={}), "sales", "exec-123")
        result = cache.get(key)
        assert result == Nothing

        stats = cache.get_stats()
        assert stats == CacheStats(hit=0, missed=1)

        # Add to cache and hit
        cache.put(sample_metric)
        key2 = cache._build_key(sample_metric)
        result = cache.get(key2)
        assert isinstance(result, Some)

        stats = cache.get_stats()
        assert stats == CacheStats(hit=1, missed=1)
        assert stats.hit_ratio() == 0.5

        # Test DB fetch counts as miss
        db.persist([sample_metric])
        # Different key to trigger DB lookup
        different_key = (Sum("costs"), ResultKey(yyyy_mm_dd=date(2024, 1, 11), tags={}), "sales", "exec-123")
        result = cache.get(different_key)
        assert result == Nothing

        stats = cache.get_stats()
        assert stats == CacheStats(hit=1, missed=2)

    def test_stats_reset_on_clear(self, cache: MetricCache, sample_metric: Metric) -> None:
        """Test stats are reset when cache is cleared."""
        # Generate some stats
        cache.put(sample_metric)
        cache_key = cache._build_key(sample_metric)
        cache.get(cache_key)  # Hit

        # Miss
        miss_key = (Sum("costs"), ResultKey(yyyy_mm_dd=date(2024, 1, 11), tags={}), "sales", "exec-123")
        cache.get(miss_key)

        # Verify stats before clear
        stats = cache.get_stats()
        assert stats.hit == 1
        assert stats.missed == 1

        # Clear and verify reset
        cache.clear()
        assert cache.get_stats() == CacheStats()
