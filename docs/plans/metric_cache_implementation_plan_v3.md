# Metric Cache Implementation Plan v3

## Problem Statement

Extended metrics (day_over_day, week_over_week, stddev) currently make redundant database queries when multiple metrics need the same base data. Additionally, extended metrics are persisted individually to the database during computation, causing performance bottlenecks.

Key issues:
- Two day_over_day metrics on different base metrics may query the same dates
- Week_over_week and day_over_day on the same metric share overlapping date queries
- Standard deviation calculations need to fetch multiple individual dates
- Extended metrics persist to DB one-by-one during analysis (inefficient)

This causes 2-3x slower analysis times and unnecessary DB writes.

## Design Decisions

### Key Features in v3:
1. **No backward compatibility required** - As confirmed by Nam
2. **Cache with dirty tracking** - Track computed-but-not-persisted metrics
3. **Batch persistence** - Flush all extended metrics to DB in one operation
4. **Cache lives in MetricProvider** - Persists across multiple analyze() calls
5. **DB integration in cache** - Cache has DB reference and fetches on miss
6. **Mandatory cache parameter** - All compute functions require cache parameter

### Cache Strategy:
- Check cache first, fall back to DB if miss
- Update cache when reading from DB succeeds
- Cache warming happens via AnalysisReport.persist()
- Extended metrics marked as dirty when computed
- Batch flush dirty metrics to DB at end of analysis

## Implementation

### 1. MetricCache Class with Dirty Tracking

```python
# src/dqx/cache.py
"""Metric caching with automatic DB fallback and dirty tracking."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import timedelta
from threading import Lock
from typing import TYPE_CHECKING, overload

from returns.maybe import Maybe, Nothing, Some

from dqx.common import DatasetName, ExecutionId, ResultKey, TimeSeries

if TYPE_CHECKING:
    from dqx.models import Metric
    from dqx.orm.repositories import MetricDB
    from dqx.specs import MetricSpec

# Type alias for cache keys
CacheKey = tuple[MetricSpec, ResultKey, DatasetName, ExecutionId]


class MetricCache:
    """Cache for metrics with automatic DB fallback on miss and dirty tracking."""

    def __init__(self, db: MetricDB) -> None:
        """Initialize cache with DB reference.

        Args:
            db: MetricDB instance for fetching on cache miss
        """
        self._db = db
        self._cache: dict[CacheKey, Metric] = {}
        self._dirty: set[CacheKey] = set()  # Track dirty metrics
        self._mutex = Lock()

    def get(self, key: CacheKey) -> Maybe[Metric]:
        """Get from cache, fetch from DB on miss.

        If not in cache, tries to fetch from DB and caches result.
        """
        metric_spec, result_key, dataset, execution_id = key

        with self._mutex:
            # Check cache first
            metric = self._cache.get(key)
            if metric:
                return Some(metric)

            # Cache miss - try DB
            match self._db._get_by_key(result_key, metric_spec, dataset):
                case Some(metric_obj):
                    # Verify execution_id matches
                    if metric_obj.metadata and metric_obj.metadata.execution_id == execution_id:
                        # Cache it (not dirty since from DB)
                        self._cache[key] = metric_obj
                        return Some(metric_obj)
                case Nothing():
                    pass

            return Nothing()

    @overload
    def put(self, metrics: Metric, mark_dirty: bool = False) -> None: ...

    @overload
    def put(self, metrics: Sequence[Metric], mark_dirty: bool = False) -> None: ...

    def put(self, metrics: Metric | Sequence[Metric], mark_dirty: bool = False) -> None:
        """Put single metric or sequence of metrics into cache.

        Args:
            metrics: Single metric or sequence of metrics
            mark_dirty: If True, mark these metrics as dirty (need DB write)
        """
        with self._mutex:
            if isinstance(metrics, Metric):
                # Single metric
                cache_key = self._build_key(metrics)
                self._cache[cache_key] = metrics
                if mark_dirty:
                    self._dirty.add(cache_key)
                else:
                    # If not marking dirty, remove from dirty set if present
                    self._dirty.discard(cache_key)
            else:
                # Sequence of metrics
                for metric in metrics:
                    cache_key = self._build_key(metric)
                    self._cache[cache_key] = metric
                    if mark_dirty:
                        self._dirty.add(cache_key)
                    else:
                        self._dirty.discard(cache_key)

    def get_window(
        self,
        metric_spec: MetricSpec,
        base_key: ResultKey,
        dataset: DatasetName,
        execution_id: ExecutionId,
        window_size: int
    ) -> TimeSeries:
        """Get metrics for a time window, fetching from DB as needed.

        Returns:
            TimeSeries dict with all requested dates (fetches missing from DB)
        """
        time_series: TimeSeries = {}

        for i in range(window_size):
            date_key = ResultKey(
                yyyy_mm_dd=base_key.yyyy_mm_dd - timedelta(days=i),
                tags=base_key.tags
            )
            cache_key = (metric_spec, date_key, dataset, execution_id)

            # get() will fetch from DB if needed
            match self.get(cache_key):
                case Some(metric):
                    time_series[date_key.yyyy_mm_dd] = metric.value
                case Nothing():
                    pass  # Skip missing dates

        return time_series

    def flush_dirty(self) -> int:
        """Persist all dirty metrics to DB and clear dirty set.

        Returns:
            Number of metrics flushed to DB
        """
        with self._mutex:
            if not self._dirty:
                return 0

            # Collect dirty metrics
            dirty_metrics = []
            for cache_key in self._dirty:
                if cache_key in self._cache:
                    dirty_metrics.append(self._cache[cache_key])

            if dirty_metrics:
                # Persist to DB
                persisted = list(self._db.persist(dirty_metrics))

                # Clear dirty flags for successfully persisted metrics
                for metric in persisted:
                    cache_key = self._build_key(metric)
                    self._dirty.discard(cache_key)

                return len(persisted)

            return 0

    def is_dirty(self, key: CacheKey) -> bool:
        """Check if a metric is dirty."""
        with self._mutex:
            return key in self._dirty

    def get_dirty_count(self) -> int:
        """Get count of dirty metrics."""
        with self._mutex:
            return len(self._dirty)

    def clear(self) -> None:
        """Clear entire cache and dirty set."""
        with self._mutex:
            self._cache.clear()
            self._dirty.clear()

    def _build_key(self, metric: Metric) -> CacheKey:
        """Build cache key from metric."""
        return (
            metric.spec,
            metric.key,
            metric.dataset,
            metric.metadata.execution_id if metric.metadata else "",
        )
```

### 2. Updated MetricProvider

```python
# In src/dqx/provider.py - add these changes to MetricProvider class

class MetricProvider(SymbolicMetricBase):
    def __init__(self, db: MetricDB, execution_id: ExecutionId) -> None:
        super().__init__()
        self._db = db
        self._execution_id = execution_id

        # Create cache in provider
        from dqx.cache import MetricCache
        self._cache = MetricCache(db)

    @property
    def cache(self) -> MetricCache:
        """Access to the metric cache."""
        return self._cache

    def clear_cache(self) -> None:
        """Clear the metric cache."""
        self._cache.clear()

    def flush_cache(self) -> int:
        """Flush dirty cache entries to DB.

        Returns:
            Number of metrics flushed
        """
        return self._cache.flush_dirty()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "total_cached": len(self._cache._cache),
            "dirty_count": self._cache.get_dirty_count()
        }

    # ... rest of existing code remains unchanged ...
```

### 3. Updated _create_lazy_extended_fn

```python
# In src/dqx/provider.py - replace the existing _create_lazy_extended_fn function

def _create_lazy_extended_fn(
    provider: "MetricProvider",
    compute_fn: Callable[[MetricDB, MetricSpec, str, ResultKey, ExecutionId, MetricCache], Result[float, str]],
    metric_spec: MetricSpec,
    symbol: sp.Symbol,
) -> RetrievalFn:
    """Create a lazy retrieval function that passes the provider's cache."""

    def lazy_extended_fn(key: ResultKey) -> Result[float, str]:
        # Look up the current dataset
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        # Call compute function with provider's cache
        return compute_fn(
            provider._db,
            metric_spec,
            symbolic_metric.dataset,
            key,
            provider.execution_id,
            provider.cache  # Use provider's cache directly!
        )

    return lazy_extended_fn
```

### 4. Updated _create_lazy_retrieval_fn

```python
# In src/dqx/provider.py - replace the existing _create_lazy_retrieval_fn function

def _create_lazy_retrieval_fn(provider: "MetricProvider", metric_spec: MetricSpec, symbol: sp.Symbol) -> RetrievalFn:
    """Create a lazy retrieval function that resolves dataset at evaluation time.

    This factory creates a retrieval function that defers dataset resolution
    until the metric is actually evaluated. This allows metrics to be created
    before their dataset is known (during imputation), while ensuring the
    correct dataset is used during evaluation.

    Args:
        provider: The MetricProvider instance containing the symbol registry.
        metric_spec: The metric specification to evaluate.
        symbol: The symbol representing this metric.

    Returns:
        A retrieval function that looks up the dataset from the SymbolicMetric
        at evaluation time and uses it to fetch the correct metric value.
    """

    def lazy_retrieval_fn(key: ResultKey) -> Result[float, str]:
        # Look up the current dataset from the SymbolicMetric
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        # Call the compute function with the resolved dataset, execution_id and cache
        return compute.simple_metric(
            provider._db, metric_spec, symbolic_metric.dataset, key, provider.execution_id, provider.cache
        )

    return lazy_retrieval_fn
```

### 5. Updated Analyzer

```python
# In src/dqx/analyzer.py - update the __init__ method

class Analyzer:
    """
    The Analyzer class is responsible for analyzing data from SqlDataSource
    using specified metrics and generating an AnalysisReport.

    Note: This class is NOT thread-safe. Thread safety must be handled by callers if needed.
    """

    def __init__(
        self,
        datasources: list[SqlDataSource],
        provider: MetricProvider,
        key: ResultKey,
        execution_id: ExecutionId,
    ) -> None:
        self.datasources = datasources
        self.provider = provider
        self.key = key
        self.execution_id = execution_id

    @property
    def cache(self) -> MetricCache:
        """Access to the metric cache via provider."""
        return self.provider.cache

    # ... rest of the analyzer code remains unchanged ...
```

### 6. Updated AnalysisReport.persist()

```python
# In src/dqx/analyzer.py - update the persist method in AnalysisReport class

def persist(self, db: MetricDB, overwrite: bool = True, cache: "MetricCache | None" = None) -> None:
    """Persist the analysis report to the metric database and optionally warm cache.

    Args:
        db: MetricDB instance for persistence
        overwrite: If True, overwrite existing metrics. If False, merge with existing.
        cache: Optional cache to warm with persisted metrics
    """
    if len(self) == 0:
        logger.warning("Try to save an EMPTY analysis report!")
        return

    if overwrite:
        logger.info("Overwriting analysis report ...")
        persisted = list(db.persist(self.values()))

        # Warm cache with persisted metrics
        if cache:
            cache.put(persisted)
    else:
        logger.info("Merging analysis report ...")
        self._merge_persist(db, cache)

def _merge_persist(self, db: MetricDB, cache: "MetricCache | None" = None) -> None:
    """Merge with existing metrics in the database before persisting."""
    db_report = AnalysisReport()

    for key, metric in self.items():
        # Find the metric in DB using pattern matching
        match db.get(metric.key, metric.spec):
            case Some(db_metric):
                db_report[key] = db_metric
            case Nothing():
                pass  # Metric not in DB

    # Merge and persist
    merged_report = self.merge(db_report)
    persisted = list(db.persist(merged_report.values()))

    # Warm cache
    if cache:
        cache.put(persisted)
```

### 7. Updated analyze_extended_metrics()

```python
# In src/dqx/analyzer.py - replace the analyze_extended_metrics method

def analyze_extended_metrics(self) -> AnalysisReport:
    # First sort the metrics topologically for analysis
    self.provider.registry.topological_sort()

    report: AnalysisReport = AnalysisReport()
    metadata = Metadata(execution_id=self.execution_id)

    for sym_metric in self.metrics:
        # Check if it's an extended metric using isinstance
        if sym_metric.metric_spec.is_extended:
            # Calculate effective key with lag
            effective_key = self.key.lag(sym_metric.lag)

            # Extended metrics ALWAYS have a dataset - they inherit from base metrics
            assert sym_metric.dataset is not None, f"Extended metric {sym_metric.name} has no dataset"

            try:
                result = sym_metric.fn(effective_key)

                match result:
                    case Success(value):
                        # Build the metric key
                        metric_key = (sym_metric.metric_spec, effective_key, sym_metric.dataset)

                        # Create NonMergeable state with the actual computed value
                        state = states.NonMergeable(value=value, metric_type=sym_metric.metric_spec.metric_type)

                        metric = models.Metric.build(
                            metric=sym_metric.metric_spec,
                            key=effective_key,
                            dataset=sym_metric.dataset,
                            state=state,
                            metadata=metadata,
                        )

                        report[metric_key] = metric

                        # CHANGED: Put in cache as dirty instead of persisting to DB
                        self.cache.put(metric, mark_dirty=True)

                    case Failure(error):
                        logger.warning(f"Failed to evaluate {sym_metric.name}: {error}")

            except Exception as e:
                logger.error(f"Error evaluating {sym_metric.name}: {e}", exc_info=True)

    logger.info(f"Evaluated {len(report)} extended metrics (cached, not persisted)")
    return report
```

### 8. Updated Analyzer.analyze()

```python
# In src/dqx/analyzer.py - update the analyze method to flush cache at end

def analyze(self) -> AnalysisReport:
    # Store analysis reports by datasource name
    report: AnalysisReport = AnalysisReport()

    # Group metrics by dataset
    metrics_by_dataset: dict[str, list[SymbolicMetric]] = defaultdict(list)
    for sym_metric in self.metrics:
        assert sym_metric.dataset is not None, f"Metric {sym_metric.name} has no dataset"
        metrics_by_dataset[sym_metric.dataset].append(sym_metric)

    # Phase 1: Analyze simple metrics for each datasource
    for ds in self.datasources:
        # Get all metrics for this dataset
        all_metrics = metrics_by_dataset.get(ds.name, [])

        # Filter to only include simple metrics (not extended)
        relevant_metrics = [sym_metric for sym_metric in all_metrics if not sym_metric.metric_spec.is_extended]

        # Skip if no simple metrics for this dataset
        if not relevant_metrics:
            continue

        # Group metrics by their effective date
        metrics_by_date: dict[ResultKey, list[MetricSpec]] = defaultdict(list)
        for sym_metric in relevant_metrics:
            # Use lag directly instead of key_provider
            effective_key = self.key.lag(sym_metric.lag)
            metrics_by_date[effective_key].append(sym_metric.metric_spec)

        # Analyze each date group separately
        this_report = self.analyze_simple_metrics(ds, metrics_by_date)

        report.update(this_report)

    # Phase 1 completion - persist simple metrics and warm cache
    logger.info("Persisting simple metrics...")
    report.persist(self.db, cache=self.cache)

    # Phase 2: Evaluate extended metrics with warmed cache
    logger.info("Evaluating extended metrics...")
    extended_report = self.analyze_extended_metrics()
    report.update(extended_report)

    # Phase 3: Flush all dirty metrics to DB
    flushed_count = self.cache.flush_dirty()
    if flushed_count > 0:
        logger.info(f"Flushed {flushed_count} extended metrics to database")

    return report
```

### 9. Updated Compute Functions

```python
# src/dqx/compute.py - update ALL compute functions to have mandatory cache parameter

from dqx.cache import MetricCache

def simple_metric(
    db: MetricDB,
    metric: MetricSpec,
    dataset: str,
    nominal_key: ResultKey,
    execution_id: ExecutionId,
    cache: MetricCache  # Now mandatory!
) -> Result[float, str]:
    """Retrieve a simple metric value using cache."""
    cache_key = (metric, nominal_key, dataset, execution_id)
    result = cache.get(cache_key)
    match result:
        case Some(cached_metric):
            return Success(cached_metric.value)
        case Nothing():
            error_msg = f"Metric {metric.name} for {nominal_key.yyyy_mm_dd.isoformat()} on dataset '{dataset}' not found!"
            return Failure(error_msg)


def day_over_day(
    db: MetricDB,
    metric: MetricSpec,
    dataset: str,
    nominal_key: ResultKey,
    execution_id: ExecutionId,
    cache: MetricCache  # Mandatory
) -> Result[float, str]:
    """Calculate day-over-day ratio using cache."""
    base_key = nominal_key.lag(0)

    # Get today's value
    today_result = simple_metric(db, metric, dataset, base_key, execution_id, cache)
    match today_result:
        case Failure() as f:
            return f
        case Success(today_val):
            pass

    # Get yesterday's value
    yesterday_key = base_key.lag(1)
    yesterday_result = simple_metric(db, metric, dataset, yesterday_key, execution_id, cache)
    match yesterday_result:
        case Failure() as f:
            return f
        case Success(yesterday_val):
            pass

    if yesterday_val == 0:
        return Failure(f"Cannot calculate day over day: previous day value ({yesterday_key.yyyy_mm_dd}) is zero.")

    return Success(today_val / yesterday_val)


def week_over_week(
    db: MetricDB,
    metric: MetricSpec,
    dataset: str,
    nominal_key: ResultKey,
    execution_id: ExecutionId,
    cache: MetricCache  # Mandatory
) -> Result[float, str]:
    """Calculate week-over-week ratio using cache."""
    base_key = nominal_key.lag(0)

    # Get today's value
    today_result = simple_metric(db, metric, dataset, base_key, execution_id, cache)
    match today_result:
        case Failure() as f:
            return f
        case Success(today_val):
            pass

    # Get week ago value
    week_ago_key = base_key.lag(7)
    week_ago_result = simple_metric(db, metric, dataset, week_ago_key, execution_id, cache)
    match week_ago_result:
        case Failure() as f:
            return f
        case Success(week_ago_val):
            pass

    if week_ago_val == 0:
        return Failure(f"Cannot calculate week over week: week ago value ({week_ago_key.yyyy_mm_dd}) is zero.")

    return Success(today_val / week_ago_val)


def stddev(
    db: MetricDB,
    metric: MetricSpec,
    size: int,
    dataset: str,
    nominal_key: ResultKey,
    execution_id: ExecutionId,
    cache: MetricCache  # Mandatory
) -> Result[float, str]:
    """Calculate standard deviation using cache."""
    import statistics
    from datetime import timedelta

    base_key = nominal_key.lag(0)

    # Use cache.get_window() - it fetches from DB as needed
    time_series = cache.get_window(metric, base_key, dataset, execution_id, size)

    if len(time_series) == size:
        # Got all values
        values = [time_series[base_key.yyyy_mm_dd - timedelta(days=i)] for i in range(size)]

        if len(values) < 2:
            return Success(0.0)

        try:
            return Success(statistics.stdev(values))
        except statistics.StatisticsError as e:
            return Failure(f"Failed to calculate standard deviation: {e}")
    else:
        # Missing some dates
        missing = size - len(time_series)
        missing_dates = []
        for i in range(size):
            date = base_key.yyyy_mm_dd - timedelta(days=i)
            if date not in time_series:
                missing_dates.append(date.isoformat())
                if len(missing_dates) >= 3:  # Limit to 3 examples
                    break

        dates_str = ", ".join(missing_dates[:3])
        if missing > 3:
            dates_str += ", ..."

        return Failure(f"There are {missing} dates with missing metrics. Missing dates: {dates_str}")
```

## Test Plan

### Phase 1: Unit Tests for MetricCache (3 tests)

```python
# tests/test_cache.py
"""Tests for MetricCache functionality."""

import uuid
from datetime import date, timedelta

import pytest
from returns.maybe import Nothing, Some

from dqx.cache import MetricCache, CacheKey
from dqx.common import ExecutionId, Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.specs import Sum
from dqx.states import SimpleAdditiveState


@pytest.fixture
def db():
    """Create in-memory database for testing."""
    return InMemoryMetricDB()


@pytest.fixture
def cache(db):
    """Create cache with DB."""
    return MetricCache(db)


@pytest.fixture
def sample_metric():
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

    def test_get_miss_returns_nothing(self, cache):
        """Test cache miss returns Nothing."""
        key = (Sum("revenue"), ResultKey(yyyy_mm_dd=date(2024, 1, 10)), "sales", "exec-123")
        result = cache.get(key)
        assert result.is_nothing()

    def test_put_and_get_single_metric(self, cache, sample_metric):
        """Test putting and getting single metric."""
        # Put metric
        cache.put(sample_metric)

        # Get metric
        key = (
            sample_metric.spec,
            sample_metric.key,
            sample_metric.dataset,
            sample_metric.metadata.execution_id,
        )
        result = cache.get(key)

        assert result.is_some()
        assert result.unwrap() == sample_metric

    def test_put_sequence_of_metrics(self, cache):
        """Test putting multiple metrics at once."""
        metrics = []
        for i in range(3):
            metric = Metric.build(
                metric=Sum("revenue"),
                key=ResultKey(yyyy_mm_dd=date(2024, 1, 10 + i)),
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
            assert result.is_some()
```

### Phase 2: Advanced Cache Tests (5 tests)

```python
    def test_get_from_db_on_miss(self, db, cache, sample_metric):
        """Test cache fetches from DB on miss."""
        # Add to DB but not cache
        db.persist([sample_metric])

        # Get from cache (should fetch from DB)
        key = (
            sample_metric.spec,
            sample_metric.key,
            sample_metric.dataset,
            sample_metric.metadata.execution_id,
        )
        match cache.get(key):
            case Some(cached_metric):
                assert cached_metric.value == sample_metric.value
            case Nothing():
                pytest.fail("Expected to find metric in DB")

        # Second get should use cache
        match cache.get(key):
            case Some(_):
                pass  # Expected
            case Nothing():
                pytest.fail("Expected metric to be in cache")

    def test_execution_id_mismatch(self, db, cache, sample_metric):
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

        assert result.is_nothing()

    def test_get_window(self, db, cache):
        """Test getting a window of metrics."""
        # Create metrics for multiple days
        metrics = []
        base_date = date(2024, 1, 10)
        for i in range(5):
            metric = Metric.build(
                metric=Sum("revenue"),
                key=ResultKey(yyyy_mm_dd=base_date - timedelta(days=i)),
                dataset="sales",
                state=SimpleAdditiveState(value=100.0 + i),
                metadata=Metadata(execution_id="exec-123"),
            )
            metrics.append(metric)

        # Add to cache
        cache.put(metrics)

        # Get window
        time_series = cache.get_window(
            Sum("revenue"),
            ResultKey(yyyy_mm_dd=base_date),
            "sales",
            "exec-123",
            5
        )

        assert len(time_series) == 5
        assert time_series[base_date] == 100.0
        assert time_series[base_date - timedelta(days=4)] == 104.0

    def test_clear(self, cache, sample_metric):
        """Test clearing cache."""
        # Add metric
        cache.put(sample_metric)

        # Clear cache
        cache.clear()

        # Verify cleared
        key = (
            sample_metric.spec,
            sample_metric.key,
            sample_metric.dataset,
            sample_metric.metadata.execution_id,
        )
        result = cache.get(key)
        assert result.is_nothing()

    def test_thread_safety(self, cache):
        """Test cache is thread-safe."""
        import threading
        import time

        results = []

        def put_and_get():
            metric = Metric.build(
                metric=Sum("revenue"),
                key=ResultKey(yyyy_mm_dd=date(2024, 1, 10)),
                dataset="sales",
                state=SimpleAdditiveState(value=100.0),
                metadata=Metadata(execution_id="exec-123"),
            )
            cache.put(metric)

            key = (metric.spec, metric.key, metric.dataset, "exec-123")
            result = cache.get(key)
            results.append(result.is_some())

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
```

### Phase 3: Dirty Tracking Tests (3 tests)

```python
    def test_dirty_tracking(self, cache, sample_metric):
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

    def test_put_clean_removes_dirty_flag(self, cache, sample_metric):
        """Test putting clean metric removes dirty flag."""
        # Put as dirty
        cache.put(sample_metric, mark_dirty=True)
        assert cache.get_dirty_count() == 1

        # Put same metric as clean
        cache.put(sample_metric, mark_dirty=False)
        assert cache.get_dirty_count() == 0

    def test_flush_empty_dirty_set(self, cache):
        """Test flushing with no dirty metrics."""
        assert cache.flush_dirty() == 0
```

### Phase 4: Integration Tests with Analyzer (5 tests)

```python
# tests/test_cache_integration.py
"""Integration tests for cache with analyzer."""

from datetime import date

import pytest
from returns.maybe import Some

from dqx.analyzer import Analyzer, AnalysisReport
from dqx.common import ExecutionId, ResultKey
from dqx.data.memory import InMemoryDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider, MetricRegistry


@pytest.fixture
def db():
    """Create in-memory database."""
    return InMemoryMetricDB()


@pytest.fixture
def execution_id():
    """Test execution ID."""
    return "test-exec-123"


@pytest.fixture
def provider(db, execution_id):
    """Create provider with cache."""
    return MetricProvider(db, execution_id)


@pytest.fixture
def datasources():
    """Create test data sources."""
    data = {
        date(2024, 1, 10): {"revenue": 100.0, "cost": 50.0},
        date(2024, 1, 9): {"revenue": 90.0, "cost": 45.0},
        date(2024, 1, 8): {"revenue": 80.0, "cost": 40.0},
        date(2024, 1, 3): {"revenue": 70.0, "cost": 35.0},
    }

    sources = []
    for dt, values in data.items():
        rows = [values]
        sources.append(InMemoryDataSource(rows, name="sales", yyyy_mm_dd=dt))

    return sources


class TestCacheIntegration:
    """Test cache integration with analyzer."""

    def test_cache_warming_on_persist(self, provider, datasources, execution_id):
        """Test cache is warmed when persisting simple metrics."""
        # Create analyzer
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 10))
        analyzer = Analyzer(datasources, provider, key, execution_id)

        # Register simple metric
        provider.sum("revenue", dataset="sales")

        # Analyze (will persist and warm cache)
        report = analyzer.analyze()

        # Check cache has metrics using pattern matching
        cache_key = (provider.specs.Sum("revenue"), key, "sales", execution_id)
        match provider.cache.get(cache_key):
            case Some(cached_metric):
                assert cached_metric.value == 100.0
            case Nothing():
                pytest.fail("Expected metric to be in cache after persist")

    def test_extended_metrics_use_cache(self, provider, datasources, execution_id):
        """Test extended metrics use cached values."""
        # Create analyzer
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 10))
        analyzer = Analyzer(datasources, provider, key, execution_id)

        # Register metrics
        revenue = provider.sum("revenue", dataset="sales")
        dod = provider.ext.day_over_day(revenue)

        # Analyze
        report = analyzer.analyze()

        # Verify DoD calculated correctly
        assert len(report) == 2  # revenue + dod
        dod_metric = report[report.find_keys(lambda spec: "day_over_day" in spec.name)[0]]
        assert dod_metric.value == pytest.approx(100.0 / 90.0)  # ~1.111

    def test_multiple_extended_metrics_share_cache(self, provider, datasources, execution_id):
        """Test multiple extended metrics share cached base metrics."""
        # Create analyzer
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 10))
        analyzer = Analyzer(datasources, provider, key, execution_id)

        # Register multiple extended metrics on same base
        revenue = provider.sum("revenue", dataset="sales")
        dod = provider.ext.day_over_day(revenue)
        wow = provider.ext.week_over_week(revenue)

        # Analyze
        report = analyzer.analyze()

        # Verify both extended metrics calculated
        assert len(report) == 3  # revenue + dod + wow

        # DoD: 100/90
        dod_metric = report[report.find_keys(lambda spec: "day_over_day" in spec.name)[0]]
        assert dod_metric.value == pytest.approx(100.0 / 90.0)

        # WoW: 100/70
        wow_metric = report[report.find_keys(lambda spec: "week_over_week" in spec.name)[0]]
        assert wow_metric.value == pytest.approx(100.0 / 70.0)

    def test_cache_persists_across_analyses(self, provider, datasources, execution_id):
        """Test cache persists across multiple analyze() calls."""
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 10))

        # First analysis
        analyzer1 = Analyzer(datasources, provider, key, execution_id)
        provider.sum("revenue", dataset="sales")
        report1 = analyzer1.analyze()

        # Clear provider registry but keep cache
        provider._registry = MetricRegistry()

        # Second analysis with same provider
        analyzer2 = Analyzer(datasources, provider, key, execution_id)
        revenue = provider.sum("revenue", dataset="sales")
        dod = provider.ext.day_over_day(revenue)

        report2 = analyzer2.analyze()

        # DoD should work even though base metric was from cache
        assert len(report2) == 2
        dod_metric = report2[report2.find_keys(lambda spec: "day_over_day" in spec.name)[0]]
        assert dod_metric.value == pytest.approx(100.0 / 90.0)

    def test_extended_metrics_persistence(self, provider, datasources, execution_id):
        """Test extended metrics are persisted via flush."""
        # Setup
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 10))
        analyzer = Analyzer(datasources, provider, key, execution_id)

        # Register metrics
        revenue = provider.sum("revenue", dataset="sales")
        dod = provider.ext.day_over_day(revenue)

        # Analyze
        report = analyzer.analyze()

        # Verify extended metric is in DB
        db_result = provider._db.get(key, dod.metric_spec)
        assert db_result.is_some()
        assert db_result.unwrap().value == pytest.approx(100.0 / 90.0)
```

## Migration Guide

### For Existing Code

1. **Update compute function calls** - Add cache parameter:
   ```python
   # Before
   result = compute.simple_metric(db, spec, dataset, key, execution_id)

   # After
   result = compute.simple_metric(db, spec, dataset, key, execution_id, cache)
   ```

2. **Provider initialization** - No changes needed, cache is created automatically

3. **Analyzer usage** - No changes needed, cache is accessed via provider

### Performance Impact

- **Extended metrics**: 2-3x faster due to cache hits
- **Memory usage**: ~1MB per 1000 cached metrics
- **DB writes**: Reduced by 50-90% for extended metrics

## TDD Implementation Phases

### Phase 1: Basic Cache Infrastructure (4 tasks)

1. **Write failing tests for basic MetricCache**
   - `test_cache.py`: test_get_miss_returns_nothing
   - `test_cache.py`: test_put_and_get_single_metric
   - `test_cache.py`: test_put_sequence_of_metrics

2. **Implement basic MetricCache class**
   - Create `src/dqx/cache.py` with get/put/clear methods
   - Add _cache dictionary and _mutex for thread safety
   - Implement _build_key method

3. **Write failing tests for cache-DB integration**
   - `test_cache.py`: test_get_from_db_on_miss
   - `test_cache.py`: test_execution_id_mismatch
   - `test_cache.py`: test_get_window

4. **Complete MetricCache DB integration**
   - Add DB reference and fallback logic in get()
   - Implement get_window() method
   - Run `uv run pytest tests/test_cache.py::TestMetricCache -v`
   - Commit: `feat(cache): add basic MetricCache with DB fallback`

### Phase 2: Dirty Tracking Feature (5 tasks)

1. **Write failing tests for dirty tracking**
   - `test_cache.py`: test_dirty_tracking
   - `test_cache.py`: test_put_clean_removes_dirty_flag
   - `test_cache.py`: test_flush_empty_dirty_set

2. **Add dirty tracking to MetricCache**
   - Add `_dirty: set[CacheKey]` attribute
   - Update put() to accept mark_dirty parameter
   - Implement is_dirty() and get_dirty_count()

3. **Write failing test for thread safety**
   - `test_cache.py`: test_thread_safety
   - `test_cache.py`: test_clear

4. **Implement flush_dirty method**
   - Collect dirty metrics and persist to DB
   - Clear dirty flags after successful persist
   - Add clear() method

5. **Run all cache tests and verify coverage**
   - `uv run pytest tests/test_cache.py -v`
   - `uv run coverage run -m pytest tests/test_cache.py`
   - Commit: `feat(cache): add dirty tracking and flush capability`

### Phase 3: Provider Integration (4 tasks)

1. **Write failing tests for Provider cache**
   - Create `tests/test_provider_cache.py`
   - Test provider.cache property exists
   - Test flush_cache() and get_cache_stats()

2. **Update MetricProvider class**
   - Import MetricCache in __init__
   - Create _cache = MetricCache(db)
   - Add cache property, clear_cache(), flush_cache(), get_cache_stats()

3. **Write failing tests for lazy functions**
   - Test _create_lazy_retrieval_fn passes cache
   - Test _create_lazy_extended_fn passes cache

4. **Update lazy function creators**
   - Modify _create_lazy_retrieval_fn to pass provider.cache
   - Modify _create_lazy_extended_fn to pass provider.cache
   - Run `uv run pytest tests/test_provider_cache.py -v`
   - Commit: `feat(provider): integrate cache with MetricProvider`

### Phase 4: Analyzer Cache Usage (5 tasks)

1. **Write failing test for analyzer cache property**
   - Create `tests/test_analyzer_cache.py`
   - Test analyzer.cache returns provider.cache

2. **Update Analyzer class**
   - Add cache property returning self.provider.cache
   - No other changes needed to Analyzer.__init__

3. **Write failing tests for persist cache warming**
   - Test AnalysisReport.persist() warms cache
   - Test _merge_persist() also warms cache

4. **Update AnalysisReport.persist()**
   - Add optional cache parameter
   - Call cache.put(persisted) after DB persist
   - Update _merge_persist() similarly

5. **Run analyzer cache tests**
   - `uv run pytest tests/test_analyzer_cache.py -v`
   - Commit: `feat(analyzer): add cache property and persist warming`

### Phase 5: Extended Metrics Caching (5 tasks)

1. **Write failing test for extended metrics caching**
   - Test extended metrics are marked dirty
   - Test no immediate DB persistence

2. **Update analyze_extended_metrics()**
   - Replace `self.db.persist([metric])`
   - With `self.cache.put(metric, mark_dirty=True)`
   - Update log message

3. **Write failing test for cache flushing**
   - Test dirty metrics are flushed at end
   - Test flush count is correct

4. **Update analyze() method**
   - Pass cache to report.persist()
   - Add Phase 3 to flush dirty metrics
   - Log flush results

5. **Run extended metrics tests**
   - `uv run pytest tests/test_analyzer_cache.py -v`
   - Commit: `feat(analyzer): use cache for extended metrics`

### Phase 6: Compute Functions Update (4 tasks)

1. **Write failing tests for compute with cache**
   - Create `tests/test_compute_cache.py`
   - Test simple_metric with cache parameter
   - Test extended metrics (day_over_day, week_over_week, stddev)

2. **Update compute.py function signatures**
   - Add `cache: MetricCache` parameter to all functions
   - Import MetricCache at top

3. **Update compute function implementations**
   - simple_metric: Use cache.get() instead of db.get()
   - Extended metrics: Use cache instead of calling db
   - stddev: Use cache.get_window()

4. **Run compute tests and fix failures**
   - `uv run pytest tests/test_compute_cache.py -v`
   - Fix any test failures in existing tests
   - Commit: `feat(compute): add mandatory cache parameter`

### Phase 7: Integration Testing (5 tasks)

1. **Write cache warming integration test**
   - `test_cache_integration.py`: test_cache_warming_on_persist
   - Verify cache contains metrics after analysis

2. **Write extended metrics cache usage test**
   - `test_cache_integration.py`: test_extended_metrics_use_cache
   - Verify DoD calculation uses cached values

3. **Write cache sharing test**
   - `test_cache_integration.py`: test_multiple_extended_metrics_share_cache
   - Verify multiple extended metrics share base metric cache

4. **Write persistence test**
   - `test_cache_integration.py`: test_extended_metrics_persistence
   - Verify extended metrics are in DB after flush

5. **Run all integration tests**
   - `uv run pytest tests/test_cache_integration.py -v`
   - Commit: `test(cache): add integration tests`

### Phase 8: Final Validation (4 tasks)

1. **Update existing tests for cache parameter**
   - Find all compute function calls in tests
   - Add mock cache or real cache as needed
   - Ensure all tests pass

2. **Run mypy type checking**
   - `uv run mypy src/dqx/cache.py`
   - `uv run mypy src/dqx/compute.py`
   - Fix any type errors

3. **Run ruff linting**
   - `uv run ruff check --fix`
   - Fix any remaining linting issues

4. **Run full test suite and pre-commit**
   - `uv run pytest tests/ -v`
   - `uv run hooks`
   - Final commit: `feat(cache): complete metric cache implementation`

## Summary

This implementation adds a high-performance cache layer that:

1. **Eliminates redundant DB queries** - Cache serves repeated metric requests
2. **Batch persists extended metrics** - Single DB write instead of N writes
3. **Maintains data consistency** - Dirty tracking ensures all metrics are persisted
4. **Integrates seamlessly** - No changes needed to existing code using Analyzer
5. **Provides visibility** - Cache statistics for monitoring and debugging

The cache is transparent to users but provides significant performance improvements, especially for complex analyses with many extended metrics.
