"""Metric cache implementation for performance optimization.

This module provides a cache for metrics to reduce redundant database queries
when extended metrics repeatedly request the same base metrics during evaluation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from threading import RLock
from typing import TYPE_CHECKING, TypeAlias, overload

from returns.maybe import Maybe, Nothing, Some

from dqx.common import DatasetName, ExecutionId, ResultKey, TimeSeries
from dqx.models import Metric
from dqx.specs import MetricSpec

if TYPE_CHECKING:
    from dqx.orm.repositories import MetricDB

logger = logging.getLogger(__name__)

# Type alias for cache key
CacheKey: TypeAlias = tuple[MetricSpec, ResultKey, DatasetName, ExecutionId]


@dataclass
class CacheStats:
    """Statistics for cache performance tracking."""

    hit: int = 0
    missed: int = 0

    def hit_ratio(self) -> float:
        """Calculate cache hit ratio.

        Returns:
            Hit ratio as hit / (hit + missed), or 0.0 if no requests
        """
        total = self.hit + self.missed
        return self.hit / total if total > 0 else 0.0

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hit += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.missed += 1

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.hit = 0
        self.missed = 0


class MetricCache:
    """Thread-safe cache for metrics with dirty tracking.

    This cache provides:
    - Unlimited size (no LRU eviction)
    - Thread-safe operations
    - Automatic DB fallback on cache miss
    - Dirty tracking for batch persistence
    - Manual cache clearing

    The cache key is a tuple of (MetricSpec, ResultKey, dataset, execution_id)
    to ensure metrics are properly isolated by execution context.
    """

    def __init__(self, db: MetricDB) -> None:
        """Initialize cache with database reference.

        Args:
            db: MetricDB instance for fallback queries
        """
        self._db = db
        self._cache: dict[CacheKey, Metric] = {}
        self._dirty: set[CacheKey] = set()
        self._lock = RLock()
        self._stats = CacheStats()

    def get(self, key: CacheKey) -> Maybe[Metric]:
        """Get metric from cache or DB.

        First checks cache, then falls back to DB if not found.
        If found in DB, updates cache for future lookups.

        Args:
            key: Cache key tuple (metric_spec, result_key, dataset, execution_id)

        Returns:
            Maybe[Metric]: Some(metric) if found, Nothing otherwise
        """
        # First check: acquire lock just for cache lookup
        with self._lock:
            # Check cache first
            if key in self._cache:
                self._stats.record_hit()
                return Some(self._cache[key])

        # Cache miss - perform DB I/O without holding lock
        metric_spec, result_key, dataset, execution_id = key

        # Query DB with execution_id filter (no lock held)
        db_result: Maybe[Metric] = self._db.get_metric(metric_spec, result_key, dataset, execution_id)

        match db_result:
            case Some(value):
                with self._lock:
                    self._cache[key] = value
                    self._stats.record_miss()
                    return Some(value)
            case _:
                with self._lock:
                    self._stats.record_miss()
                return Nothing

    @overload
    def put(self, metrics: Metric, mark_dirty: bool = False) -> None:
        """Put a single metric into cache.

        Args:
            metrics: Single metric to cache
            mark_dirty: If True, marks metric as dirty (not persisted)
        """
        ...

    @overload
    def put(self, metrics: Sequence[Metric], mark_dirty: bool = False) -> None:
        """Put a sequence of metrics into cache.

        Args:
            metrics: Sequence of metrics to cache
            mark_dirty: If True, marks metrics as dirty (not persisted)
        """
        ...

    def put(self, metrics: Metric | Sequence[Metric], mark_dirty: bool = False) -> None:
        """Put metric(s) into cache.

        Args:
            metrics: Single metric or sequence of metrics to cache
            mark_dirty: If True, marks metrics as dirty (not persisted)
        """
        with self._lock:
            # Handle single metric
            if isinstance(metrics, Metric):
                metrics = [metrics]

            for metric in metrics:
                key = self._build_key(metric)
                self._cache[key] = metric

                if mark_dirty:
                    self._dirty.add(key)
                else:
                    # Clean put removes dirty flag
                    self._dirty.discard(key)

                # logger.debug("Cached metric: %s (dirty=%s)", key, mark_dirty)

    def get_timeseries(
        self,
        metric_spec: MetricSpec,
        nominal_key: ResultKey,
        dataset: str,
        execution_id: ExecutionId,
        window: int,
    ) -> TimeSeries:
        """Get a window of metrics as a time series.

        Fetches metrics for consecutive days ending at the nominal date.

        Args:
            metric_spec: The metric specification
            nominal_key: The end date for the window
            dataset: Dataset name
            execution_id: Execution ID for filtering
            window: Number of days to include

        Returns:
            TimeSeries mapping dates to values
        """
        time_series: TimeSeries = {}

        for lag in range(window):
            date_key = nominal_key.yyyy_mm_dd - timedelta(days=lag)
            key = (
                metric_spec,
                ResultKey(yyyy_mm_dd=date_key, tags=nominal_key.tags),
                dataset,
                execution_id,
            )

            result: Maybe[Metric] = self.get(key)
            match result:
                case Some(metric):
                    time_series[metric.key.yyyy_mm_dd] = metric
                case _:
                    pass  # Skip missing dates
        return time_series

    def get_sparse_timeseries(
        self,
        metric_spec: MetricSpec,
        nominal_key: ResultKey,
        dataset: str,
        execution_id: ExecutionId,
        lags: Sequence[int],
    ) -> TimeSeries:
        time_series: TimeSeries = {}

        for lag in lags:
            key = (metric_spec, nominal_key.lag(lag), dataset, execution_id)
            result: Maybe[Metric] = self.get(key)

            match result:
                case Some(metric):
                    time_series[metric.key.yyyy_mm_dd] = metric
                case _:
                    pass  # Skip missing dates

        return time_series

    def clear(self) -> None:
        """Clear all cached metrics."""
        with self._lock:
            self._cache.clear()
            self._dirty.clear()
            self._stats.reset()
            logger.info("Cache cleared")

    def is_dirty(self, key: CacheKey) -> bool:
        """Check if a metric is marked as dirty.

        Args:
            key: Cache key to check

        Returns:
            True if metric is dirty (not persisted)
        """
        with self._lock:
            return key in self._dirty

    def get_dirty_count(self) -> int:
        """Get count of dirty metrics.

        Returns:
            Number of metrics marked as dirty
        """
        with self._lock:
            return len(self._dirty)

    def write_back(self) -> int:
        """Write back all cached dirty metrics to the database.

        Returns:
            Number of metrics written to database
        """
        with self._lock:
            if not self._dirty:
                return 0

            # Collect dirty metrics
            dirty_metrics = []
            for key in self._dirty:
                if key in self._cache:
                    dirty_metrics.append(self._cache[key])

            # Persist to DB
            if dirty_metrics:
                self._db.persist(dirty_metrics)
                logger.info("Flushed %d dirty metrics to DB", len(dirty_metrics))

            # Clear dirty set
            count = len(self._dirty)
            self._dirty.clear()

            return count

    def _build_key(self, metric: Metric) -> CacheKey:
        """Build cache key from metric.

        Args:
            metric: Metric to build key for

        Returns:
            Cache key tuple
        """
        # All metrics should have metadata
        if not metric.metadata:
            raise ValueError(f"Metric missing metadata: {metric}")

        # execution_id can be empty string but not None
        if metric.metadata.execution_id is None:
            raise ValueError(f"Metric missing execution_id: {metric}")

        return (
            metric.spec,
            metric.key,
            metric.dataset,
            metric.metadata.execution_id,
        )

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics.

        Returns:
            CacheStats with current hit/miss counts and ratio
        """
        with self._lock:
            return self._stats
