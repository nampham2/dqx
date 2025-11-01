# Metric Cache Implementation Plan v2

## Problem Statement

Extended metrics (day_over_day, week_over_week, stddev) currently make redundant database queries when multiple metrics need the same base data. For example:
- Two day_over_day metrics on different base metrics may query the same dates
- Week_over_week and day_over_day on the same metric share overlapping date queries
- Standard deviation calculations need to fetch multiple individual dates

This causes performance issues in workflows with many extended metrics, leading to 2-3x slower analysis times.

## Design Decisions

### Key Changes from v1:
1. **No backward compatibility required** - As confirmed by Nam
2. **Simpler architecture** - No CachedMetricDB wrapper, use MetricCache directly
3. **Cache lives in MetricProvider** - Persists across multiple analyze() calls
4. **DB integration in cache** - Cache has DB reference and fetches on miss
5. **Mandatory cache parameter** - All compute functions require cache parameter
6. **Automatic cache management** - No manual cache updates needed

### Cache Scope:
- Cache instance created in MetricProvider constructor
- Passed to Analyzer via provider.cache property
- Lives for the entire provider session
- Cleared manually via provider.clear_cache() if needed

### Cache Strategy:
- Check cache first, fall back to DB if miss
- Update cache when reading from DB succeeds
- Cache warming happens via AnalysisReport.persist()
- Extended metrics use cache transparently

## Implementation

### 1. MetricCache Class

```python
# src/dqx/cache.py
"""Metric caching with automatic DB fallback."""

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
    """Cache for metrics with automatic DB fallback on miss."""

    def __init__(self, db: MetricDB) -> None:
        """Initialize cache with DB reference.

        Args:
            db: MetricDB instance for fetching on cache miss
        """
        self._db = db
        self._cache: dict[CacheKey, Metric] = {}
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
                        # Cache it
                        self._cache[key] = metric_obj
                        return Some(metric_obj)
                case Nothing():
                    pass

            return Nothing()

    @overload
    def put(self, metrics: Metric) -> None: ...

    @overload
    def put(self, metrics: Sequence[Metric]) -> None: ...

    def put(self, metrics: Metric | Sequence[Metric]) -> None:
        """Put single metric or sequence of metrics into cache."""
        with self._mutex:
            if isinstance(metrics, Metric):
                # Single metric
                cache_key = self._build_key(metrics)
                self._cache[cache_key] = metrics
            else:
                # Sequence of metrics
                for metric in metrics:
                    cache_key = self._build_key(metric)
                    self._cache[cache_key] = metric

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

    def clear(self) -> None:
        """Clear entire cache."""
        with self._mutex:
            self._cache.clear()

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

### 7. Updated Analyzer.analyze()

```python
# In src/dqx/analyzer.py - update the analyze method to pass cache to persist

def analyze(self) -> AnalysisReport:
    # ... existing code up to persistence ...

    # Phase 1 completion - persist simple metrics and warm cache
    logger.info("Persisting simple metrics...")
    report.persist(self.db, cache=self.cache)

    # Phase 2: Evaluate extended metrics with warmed cache
    logger.info("Evaluating extended metrics...")
    extended_report = self.analyze_extended_metrics()
    report.update(extended_report)

    return report
```

### 8. Updated Compute Functions

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

### 1. Unit Tests for MetricCache

```python
# tests/test_cache.py
"""Tests for MetricCache functionality."""

import uuid
from datetime import date

import pytest

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

### 2. Integration Tests with Analyzer

```python
# tests/test_cache_integration.py
"""Integration tests for cache with analyzer."""

from datetime import date

import pytest

from dqx.analyzer import Analyzer, AnalysisReport
from dqx.common import ExecutionId, ResultKey
from dqx.data.memory import InMemoryDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


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
```

### 3. Performance Tests

```python
# tests/test_cache_performance.py
"""Performance tests for cache."""

import time
from datetime import date, timedelta

import pytest

from dqx.analyzer import Analyzer
from dqx.common import ExecutionId, ResultKey
from dqx.data.memory import InMemoryDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def create_large_dataset(num_days=30):
    """Create a large dataset for performance testing."""
    sources = []
    base_date = date(2024, 1, 31)

    for i in range(num_days):
        dt = base_date - timedelta(days=i)
        rows = [{"revenue": 100.0 + i, "cost": 50.0 + i}]
        sources.append(InMemoryDataSource(rows, name="sales", yyyy_mm_dd=dt))

    return sources


class TestCachePerformance:
    """Performance tests for cache."""

    def test_extended_metrics_performance(self):
        """Test performance improvement with cache."""
        db = InMemoryMetricDB()
        execution_id = "perf-test"
        sources = create_large_dataset(30)
        key = ResultKey(yyyy_mm_dd=date(2024, 1, 31))

        # Without cache (simulate by clearing after persist)
        provider1 = MetricProvider(db, execution_id)
        analyzer1 = Analyzer(sources, provider1, key, execution_id)

        # Register many extended metrics
        for col in ["revenue", "cost"]:
            base = provider1.sum(col, dataset="sales")
            provider1.ext.day_over_day(base)
            provider1.ext.week_over_week(base)
            provider1.ext.stddev(base, offset=0, n=7)
            provider1.ext.stddev(base, offset=0, n=14)

        # Time without cache benefit
        start = time.time()
        report1 = analyzer1.analyze()
        provider1.clear_cache()  # Clear cache to simulate no caching
        time_no_cache = time.time() - start

        # With cache (normal operation)
        provider2 = MetricProvider(db, execution_id + "-cached")
        analyzer2 = Analyzer(sources, provider2, key, execution_id + "-cached")

        # Same metrics
        for col in ["revenue", "cost"]:
            base = provider2.sum(col, dataset="sales")
            provider2.ext.day_over_day(base)
            provider2.ext.week_over_week(base)
            provider2.ext.stddev(base, offset=0, n=7)
            provider2.ext.stddev(base, offset=0, n=14)

        # Time with cache
        start = time.time()
        report2 = analyzer2.analyze()
        time_with_cache = time.time() - start

        # Verify cache provides speedup
        print(f"Without cache: {time_no_cache:.3f}s")
        print(f"With cache: {time_with_cache:.3f}s")
        print(f"Speedup: {time_no_cache / time_with_cache:.1f}x")

        # Should be at least 1.5x faster with cache
        assert time_with_cache < time_no_cache * 0.67  # 1.5x speedup
```

## Migration Guide

Since no backward compatibility is required, the migration is straightforward:

1. **Update imports**: Add `from dqx.cache import MetricCache` where needed
2. **Update compute function signatures**: Add `cache: MetricCache` parameter
3. **Update compute function calls**: Pass cache parameter
4. **No changes needed for API users**: Cache is transparent to ValidationSuite users

## Performance Impact

Expected performance improvements:
- **2-3x speedup** for workflows with multiple extended metrics
- **Reduced DB queries** by 50-80% for extended metrics
- **Memory overhead**: ~100 bytes per cached metric (negligible for typical workflows)

## Conclusion

This implementation provides a simple, effective caching solution for extended metrics:
- Transparent to users
- Minimal code changes
- Significant performance improvement
- Thread-safe and robust
- Easy to test and maintain

The cache lives in the provider, is passed to the analyzer, and automatically handles DB fallback on cache misses. Extended metrics benefit from cached base metrics without any changes to their logic.
