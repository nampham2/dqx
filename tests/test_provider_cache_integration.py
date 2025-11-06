"""Tests for MetricProvider cache integration."""

from datetime import date, timedelta

import pytest
from returns.maybe import Some
from returns.result import Success

from dqx.cache import MetricCache
from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
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
def provider(db: InMemoryMetricDB) -> MetricProvider:
    """Create provider with its own internal cache."""
    return MetricProvider(db=db, execution_id="test-exec", data_av_threshold=0.8)


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


class TestProviderCacheIntegration:
    """Tests for MetricProvider cache integration."""

    def test_provider_uses_cache_on_get(self, provider: MetricProvider, sample_metric: Metric) -> None:
        """Test provider uses cache when getting metrics."""
        # Put metric in provider's cache
        provider.cache.put(sample_metric)

        # Get through provider - should hit cache
        result = provider.get_metric(
            metric_spec=sample_metric.spec,
            result_key=sample_metric.key,
            dataset=sample_metric.dataset,
            execution_id="exec-123",
        )

        assert isinstance(result, Success)
        assert result.unwrap() == sample_metric

        # Verify it came from cache by checking cache stats
        stats = provider.cache.get_stats()
        assert stats.hit > 0

    def test_provider_populates_cache_on_miss(
        self, provider: MetricProvider, db: InMemoryMetricDB, sample_metric: Metric
    ) -> None:
        """Test provider populates cache when fetching from DB."""
        # Add to DB only
        db.persist([sample_metric])

        # Get through provider - should miss cache, hit DB, then populate cache
        result = provider.get_metric(
            metric_spec=sample_metric.spec,
            result_key=sample_metric.key,
            dataset=sample_metric.dataset,
            execution_id="exec-123",
        )

        assert isinstance(result, Success)

        # Now it should be in provider's cache
        cache_result = provider.cache.get((sample_metric.spec, sample_metric.key, sample_metric.dataset, "exec-123"))
        assert isinstance(cache_result, Some)

    def test_provider_persist_updates_cache(self, provider: MetricProvider, sample_metric: Metric) -> None:
        """Test provider updates cache when persisting metrics."""
        # Persist through provider
        provider.persist([sample_metric])

        # Should be in provider's cache
        cache_result = provider.cache.get((sample_metric.spec, sample_metric.key, sample_metric.dataset, "exec-123"))
        assert isinstance(cache_result, Some)
        assert cache_result.unwrap() == sample_metric

    def test_provider_batch_operations_use_cache(self, provider: MetricProvider) -> None:
        """Test provider batch operations leverage cache."""
        # Create multiple metrics
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

        # Persist batch through provider
        provider.persist(metrics)

        # Get batch through provider
        results = provider.get_metrics_by_execution_id("exec-123")

        assert len(results) == 5

        # All should have been served from cache
        stats = provider.cache.get_stats()
        assert stats.hit >= 5

    def test_provider_cache_invalidation_on_update(self, provider: MetricProvider, sample_metric: Metric) -> None:
        """Test cache is properly updated when metric is updated."""
        # Initial persist
        provider.persist([sample_metric])

        # Update the metric
        updated_metric = Metric.build(
            metric=sample_metric.spec,
            key=sample_metric.key,
            dataset=sample_metric.dataset,
            state=SimpleAdditiveState(value=200.0),  # Different value
            metadata=Metadata(execution_id="exec-123"),
        )

        # Persist update
        provider.persist([updated_metric])

        # Get should return updated value
        result = provider.get_metric(
            metric_spec=sample_metric.spec,
            result_key=sample_metric.key,
            dataset=sample_metric.dataset,
            execution_id="exec-123",
        )

        from returns.result import Success

        assert isinstance(result, Success)
        assert result.unwrap().value == 200.0

    def test_provider_cache_respects_execution_id(self, provider: MetricProvider, sample_metric: Metric) -> None:
        """Test cache properly isolates metrics by execution_id."""
        # Persist with one execution_id
        provider.persist([sample_metric])

        # Try to get with different execution_id
        result = provider.get_metric(
            metric_spec=sample_metric.spec,
            result_key=sample_metric.key,
            dataset=sample_metric.dataset,
            execution_id="different-exec-456",
        )

        # Should not find it
        from returns.result import Failure

        assert isinstance(result, Failure)

        # Original should still be there
        result = provider.get_metric(
            metric_spec=sample_metric.spec,
            result_key=sample_metric.key,
            dataset=sample_metric.dataset,
            execution_id="exec-123",
        )
        assert isinstance(result, Success)

    def test_provider_clear_cache(self, provider: MetricProvider, sample_metric: Metric) -> None:
        """Test provider can clear cache."""
        # Add metric
        provider.persist([sample_metric])

        # Verify it's in provider's cache
        cache_result = provider.cache.get((sample_metric.spec, sample_metric.key, sample_metric.dataset, "exec-123"))
        assert isinstance(cache_result, Some)

        # Clear cache through provider
        provider.clear_cache()

        # Check internal cache state (not through get which would reload from DB)
        assert len(provider.cache._cache) == 0
        assert provider.cache.get_dirty_count() == 0

    def test_provider_get_window_uses_cache(self, provider: MetricProvider) -> None:
        """Test provider's cache can serve time windows."""
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

        # Persist through provider
        provider.persist(metrics)

        # Get window through provider's cache
        window = provider.cache.get_timeseries(
            Sum("revenue"), ResultKey(yyyy_mm_dd=base_date, tags={}), "sales", "exec-123", 5
        )

        assert len(window) == 5
        assert window[base_date].value == 100.0
        assert window[base_date - timedelta(days=4)].value == 104.0

        # Should have used cache
        stats = provider.cache.get_stats()
        assert stats.hit > 0
