"""Tests for compute module functions."""

import statistics
from datetime import date, timedelta

import pytest
from returns.result import Failure, Success

from dqx.cache import MetricCache
from dqx.common import ExecutionId, Metadata, ResultKey, TimeSeries
from dqx.compute import (
    _sparse_timeseries_check,
    _timeseries_check,
    day_over_day,
    simple_metric,
    stddev,
    week_over_week,
)
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB, MetricDB
from dqx.specs import MetricSpec, Sum
from dqx.states import SimpleAdditiveState


@pytest.fixture
def db() -> MetricDB:
    """Create in-memory database for testing."""
    return InMemoryMetricDB()


@pytest.fixture
def cache(db: MetricDB) -> MetricCache:
    """Create cache with database."""
    return MetricCache(db)


@pytest.fixture
def metric_spec() -> MetricSpec:
    """Sample metric specification."""
    return Sum("revenue")


@pytest.fixture
def execution_id() -> ExecutionId:
    """Test execution ID."""
    return "test-exec-123"


@pytest.fixture
def base_date() -> date:
    """Base date for testing."""
    return date(2024, 1, 10)


@pytest.fixture
def result_key(base_date: date) -> ResultKey:
    """Result key for testing."""
    return ResultKey(yyyy_mm_dd=base_date, tags={"env": "test"})


def populate_metric(
    db: MetricDB,
    metric_spec: MetricSpec,
    key: ResultKey,
    value: float,
    dataset: str = "test_dataset",
    execution_id: ExecutionId = "test-exec-123",
) -> None:
    """Helper to add a metric to the database."""
    metadata = Metadata(execution_id=execution_id)
    metric = Metric.build(
        metric=metric_spec,
        key=key,
        dataset=dataset,
        state=SimpleAdditiveState(value=value),
        metadata=metadata,
    )
    db.persist([metric])


def populate_time_series(
    db: MetricDB,
    metric_spec: MetricSpec,
    base_date: date,
    values: dict[int, float],  # {days_offset: value}
    dataset: str = "test_dataset",
    execution_id: ExecutionId = "test-exec-123",
    tags: dict[str, str] | None = None,
) -> None:
    """Helper to populate metrics for multiple days."""
    for offset, value in values.items():
        key = ResultKey(yyyy_mm_dd=base_date + timedelta(days=offset), tags=tags or {})
        populate_metric(db, metric_spec, key, value, dataset, execution_id)


class TestSimpleMetric:
    """Tests for simple_metric function."""

    def test_success(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        result_key: ResultKey,
        execution_id: ExecutionId,
    ) -> None:
        """Test successful metric retrieval."""
        # Populate database
        populate_metric(db, metric_spec, result_key, 100.0, "test_dataset", execution_id)

        # Retrieve metric
        result = simple_metric(cache, metric_spec, "test_dataset", result_key, execution_id)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == pytest.approx(100.0)

    def test_not_found(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        result_key: ResultKey,
        execution_id: ExecutionId,
    ) -> None:
        """Test metric not found."""
        # Empty database
        result = simple_metric(cache, metric_spec, "test_dataset", result_key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        expected_msg = (
            f"Metric {metric_spec.name} for {result_key.yyyy_mm_dd.isoformat()} on dataset 'test_dataset' not found!"
        )
        assert result.failure() == expected_msg

    def test_different_datasets(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        result_key: ResultKey,
        execution_id: ExecutionId,
    ) -> None:
        """Test metrics in different datasets are isolated."""
        # Populate different datasets
        populate_metric(db, metric_spec, result_key, 100.0, "dataset1", execution_id)
        populate_metric(db, metric_spec, result_key, 200.0, "dataset2", execution_id)

        # Retrieve from each dataset
        result1 = simple_metric(cache, metric_spec, "dataset1", result_key, execution_id)
        result2 = simple_metric(cache, metric_spec, "dataset2", result_key, execution_id)

        # Verify
        assert isinstance(result1, Success)
        assert result1.unwrap() == pytest.approx(100.0)
        assert isinstance(result2, Success)
        assert result2.unwrap() == pytest.approx(200.0)

    def test_different_execution_ids(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        result_key: ResultKey,
    ) -> None:
        """Test metrics with different execution IDs are isolated."""
        # Populate with different execution IDs
        populate_metric(db, metric_spec, result_key, 100.0, "test_dataset", "exec-1")
        populate_metric(db, metric_spec, result_key, 200.0, "test_dataset", "exec-2")

        # Retrieve with each execution ID
        result1 = simple_metric(cache, metric_spec, "test_dataset", result_key, "exec-1")
        result2 = simple_metric(cache, metric_spec, "test_dataset", result_key, "exec-2")

        # Verify
        assert isinstance(result1, Success)
        assert result1.unwrap() == pytest.approx(100.0)
        assert isinstance(result2, Success)
        assert result2.unwrap() == pytest.approx(200.0)


class TestDayOverDay:
    """Tests for day_over_day function."""

    def test_success(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test successful day-over-day calculation."""
        # Populate two days
        values = {0: 150.0, -1: 100.0}  # Today: 150, Yesterday: 100
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = day_over_day(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == pytest.approx(1.5)  # 150/100

    def test_negative_values(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test calculation with negative values."""
        # Populate with negative values
        values = {0: -50.0, -1: -100.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = day_over_day(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == pytest.approx(0.5)  # -50/-100

    def test_missing_today(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test missing today's data."""
        # Only populate yesterday
        values = {-1: 100.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = day_over_day(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 1 dates with missing metrics" in result.failure()

    def test_missing_yesterday(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test missing yesterday's data."""
        # Only populate today
        values = {0: 100.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = day_over_day(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 1 dates with missing metrics" in result.failure()

    def test_division_by_zero(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test division by zero when yesterday's value is zero."""
        # Yesterday is zero
        values = {0: 100.0, -1: 0.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = day_over_day(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        assert "Cannot calculate day over day: previous day value" in result.failure()
        assert "is zero" in result.failure()

    def test_no_data(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test when database returns no data."""
        # Empty database
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = day_over_day(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        # DB returns empty TimeSeries, so _timeseries_check reports missing dates
        assert "There are 2 dates with missing metrics" in result.failure()


class TestWeekOverWeek:
    """Tests for week_over_week function."""

    def test_success_sparse(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test successful week-over-week with only required dates."""
        # Populate only the required dates
        values = {0: 210.0, -7: 100.0}  # Today: 210, Week ago: 100
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = week_over_week(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == pytest.approx(2.1)  # 210/100

    def test_success_full_window(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test with full 8 days of data."""
        # Populate all 8 days
        values: dict[int, float] = {i: 100.0 + i * 10.0 for i in range(-7, 1)}  # -7 to 0
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = week_over_week(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        # Today: 100 + 0*10 = 100, Week ago: 100 + (-7)*10 = 30
        assert result.unwrap() == pytest.approx(100.0 / 30.0)

    def test_missing_current_week(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test missing current week data."""
        # Only populate week ago
        values = {-7: 100.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = week_over_week(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 1 dates with missing metrics" in result.failure()

    def test_missing_previous_week(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test missing previous week data."""
        # Only populate today
        values = {0: 100.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = week_over_week(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 1 dates with missing metrics" in result.failure()

    def test_division_by_zero(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test division by zero when week ago value is zero."""
        # Week ago is zero
        values = {0: 100.0, -7: 0.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate ratio
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = week_over_week(cache, metric_spec, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        assert "Cannot calculate week over week: week ago value" in result.failure()
        assert "is zero" in result.failure()


class TestStddev:
    """Tests for stddev function."""

    def test_success_multiple_values(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test successful standard deviation calculation."""
        # Populate 5 days with values [10, 20, 30, 40, 50]
        values = {-i: float(50 - i * 10) for i in range(5)}  # Creates [10, 20, 30, 40, 50]
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate stddev
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = stddev(cache, metric_spec, 5, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        # Standard deviation of [10, 20, 30, 40, 50]
        expected = statistics.stdev([10, 20, 30, 40, 50])
        assert result.unwrap() == pytest.approx(expected)

    def test_success_window_size_2(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test with minimum window size of 2."""
        # Populate 2 days
        values = {0: 10.0, -1: 20.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate stddev
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = stddev(cache, metric_spec, 2, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        expected = statistics.stdev([20.0, 10.0])  # Chronological order
        assert result.unwrap() == pytest.approx(expected)

    def test_success_window_size_1(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test with window size of 1 (returns 0)."""
        # Populate 1 day
        values = {0: 42.0}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate stddev
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = stddev(cache, metric_spec, 1, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == 0.0  # Standard deviation of single value

    def test_success_identical_values(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test with all identical values (stddev = 0)."""
        # Populate 5 days with same value
        values = {-i: 100.0 for i in range(5)}
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate stddev
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = stddev(cache, metric_spec, 5, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == 0.0

    def test_missing_dates(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test with missing dates in window."""
        # Populate only some days
        values = {0: 10.0, -2: 30.0, -4: 50.0}  # Missing -1 and -3
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate stddev for 5 days
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = stddev(cache, metric_spec, 5, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 2 dates with missing metrics" in result.failure()

    def test_no_data(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test when database returns no data."""
        # Empty database
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = stddev(cache, metric_spec, 5, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Failure)
        # DB returns empty TimeSeries, so _timeseries_check reports missing dates
        assert "There are 5 dates with missing metrics" in result.failure()

    def test_large_window(
        self,
        db: MetricDB,
        cache: MetricCache,
        metric_spec: MetricSpec,
        base_date: date,
        execution_id: ExecutionId,
    ) -> None:
        """Test with large window size."""
        # Populate 30 days
        values = {-i: float(i * i) for i in range(30)}  # Quadratic values
        populate_time_series(db, metric_spec, base_date, values, execution_id=execution_id)

        # Calculate stddev
        key = ResultKey(yyyy_mm_dd=base_date, tags={})
        result = stddev(cache, metric_spec, 30, "test_dataset", key, execution_id)

        # Verify
        assert isinstance(result, Success)
        # Calculate expected stddev
        expected_values = [float(i * i) for i in range(30)]
        expected = statistics.stdev(expected_values)
        assert result.unwrap() == pytest.approx(expected)


class TestTimeseriesCheck:
    """Tests for _timeseries_check helper function."""

    def test_success_all_dates_present(self) -> None:
        """Test with all expected dates present."""
        # Create timeseries with all dates
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,
            base + timedelta(days=1): 110.0,
            base + timedelta(days=2): 120.0,
        }

        # Check
        result = _timeseries_check(ts, base, 3)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == ts

    def test_success_extra_dates(self) -> None:
        """Test with extra dates beyond expected range."""
        # Create timeseries with extra dates
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base - timedelta(days=1): 90.0,  # Extra before
            base: 100.0,
            base + timedelta(days=1): 110.0,
            base + timedelta(days=2): 120.0,
            base + timedelta(days=3): 130.0,  # Extra after
        }

        # Check for 3 days starting from base
        result = _timeseries_check(ts, base, 3)

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == ts

    def test_failure_single_missing_date(self) -> None:
        """Test with one missing date."""
        # Create timeseries missing one date
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,
            # Missing: base + timedelta(days=1)
            base + timedelta(days=2): 120.0,
        }

        # Check
        result = _timeseries_check(ts, base, 3)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 1 dates with missing metrics" in result.failure()
        assert "2024-01-11" in result.failure()

    def test_failure_multiple_missing_dates(self) -> None:
        """Test with multiple missing dates."""
        # Create timeseries missing multiple dates
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,
            # Missing: base + 1, 2, 3, 4
            base + timedelta(days=5): 150.0,
        }

        # Check for 6 days
        result = _timeseries_check(ts, base, 6)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 4 dates with missing metrics" in result.failure()

    def test_failure_with_limit(self) -> None:
        """Test that limit parameter controls error message."""
        # Create timeseries missing many dates
        base = date(2024, 1, 1)
        ts: TimeSeries = {base: 100.0}  # Only first date

        # Check for 10 days with limit=3
        result = _timeseries_check(ts, base, 10, limit=3)

        # Verify
        assert isinstance(result, Failure)
        failure_msg = result.failure()
        assert "There are 9 dates with missing metrics" in failure_msg
        # Should only list 3 dates due to limit
        date_count = failure_msg.count("2024-01-")
        assert date_count == 3

    def test_empty_timeseries(self) -> None:
        """Test with empty timeseries."""
        # Empty timeseries
        ts: TimeSeries = {}
        base = date(2024, 1, 10)

        # Check
        result = _timeseries_check(ts, base, 3)

        # Verify
        assert isinstance(result, Failure)
        assert "There are 3 dates with missing metrics" in result.failure()


class TestSparseTimeseriesCheck:
    """Tests for _sparse_timeseries_check helper function."""

    def test_success_all_lag_points_present(self) -> None:
        """Test with all required lag points present."""
        # Create timeseries with specific lag points
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,  # lag 0
            base - timedelta(days=7): 70.0,  # lag 7
            base - timedelta(days=30): 30.0,  # lag 30
        }

        # Check for specific lag points
        result = _sparse_timeseries_check(ts, base, [0, 7, 30])

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == ts

    def test_success_extra_dates(self) -> None:
        """Test with extra dates beyond lag points."""
        # Create timeseries with extra dates
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,  # lag 0
            base - timedelta(days=1): 90.0,  # Extra
            base - timedelta(days=7): 70.0,  # lag 7
            base - timedelta(days=15): 50.0,  # Extra
        }

        # Check for specific lag points
        result = _sparse_timeseries_check(ts, base, [0, 7])

        # Verify
        assert isinstance(result, Success)
        assert result.unwrap() == ts

    def test_failure_missing_lag_points(self) -> None:
        """Test with missing lag points."""
        # Create timeseries missing some lag points
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,  # lag 0
            # Missing: lag 7
            base - timedelta(days=30): 30.0,  # lag 30
        }

        # Check
        result = _sparse_timeseries_check(ts, base, [0, 7, 30])

        # Verify
        assert isinstance(result, Failure)
        assert "There are 1 dates with missing metrics" in result.failure()
        assert "2024-01-03" in result.failure()  # base - 7 days

    def test_empty_lag_points(self) -> None:
        """Test with empty lag points list."""
        # Create timeseries with some data
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,
            base - timedelta(days=1): 90.0,
        }

        # Check with empty lag points
        result = _sparse_timeseries_check(ts, base, [])

        # Verify - should succeed as no lag points are required
        assert isinstance(result, Success)
        assert result.unwrap() == ts

    def test_duplicate_lag_points(self) -> None:
        """Test with duplicate lag points."""
        # Create timeseries
        base = date(2024, 1, 10)
        ts: TimeSeries = {
            base: 100.0,  # lag 0
            base - timedelta(days=7): 70.0,  # lag 7
        }

        # Check with duplicate lag points
        result = _sparse_timeseries_check(ts, base, [0, 7, 7, 0])

        # Verify - should succeed as duplicates resolve to same dates
        assert isinstance(result, Success)
        assert result.unwrap() == ts

    def test_with_limit(self) -> None:
        """Test that limit parameter controls error message."""
        # Create timeseries missing many dates
        base = date(2024, 1, 10)
        ts: TimeSeries = {}  # Empty timeseries

        # Check for many lag points with limit=2
        result = _sparse_timeseries_check(ts, base, [7, 14, 21, 28, 35], limit=2)

        # Verify
        assert isinstance(result, Failure)
        failure_msg = result.failure()
        assert "There are 5 dates with missing metrics" in failure_msg
        # Should only list 2 dates due to limit (expect dates like "2024-01-03", "2024-01-")
        dates_listed = failure_msg.split(": ")[1].count(",") + 1
        assert dates_listed == 2
