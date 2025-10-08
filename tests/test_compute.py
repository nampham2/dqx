import datetime as dt
from unittest.mock import MagicMock

import numpy as np
import pytest
from returns.maybe import Nothing, Some
from returns.result import Failure, Success

from dqx import compute
from dqx.common import ResultKey, ResultKeyProvider, TimeSeries
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec


@pytest.fixture
def mock_db() -> MetricDB:
    """Create a mock MetricDB for testing."""
    return MagicMock(spec=MetricDB)


@pytest.fixture
def mock_metric() -> MetricSpec:
    """Create a mock MetricSpec for testing."""
    metric = MagicMock(spec=MetricSpec)
    metric.name = "test_metric"
    return metric


@pytest.fixture
def mock_key_provider() -> ResultKeyProvider:
    """Create a mock ResultKeyProvider for testing."""
    return MagicMock(spec=ResultKeyProvider)


@pytest.fixture
def mock_result_key() -> ResultKey:
    """Create a mock ResultKey for testing."""
    key = MagicMock(spec=ResultKey)
    key.yyyy_mm_dd = dt.date(2023, 1, 1)
    key.lag.return_value.yyyy_mm_dd = dt.date(2022, 12, 31)
    return key


@pytest.fixture
def sample_timeseries() -> TimeSeries:
    """Create a sample TimeSeries for testing."""
    return {
        dt.date(2023, 1, 1): 100.0,
        dt.date(2022, 12, 31): 95.0,
        dt.date(2022, 12, 30): 90.0,
        dt.date(2022, 12, 29): 85.0,
    }


def test_timeseries_check_success() -> None:
    """Test _timeseries_check with complete data."""
    ts = {
        dt.date(2023, 1, 1): 100.0,
        dt.date(2023, 1, 2): 105.0,
    }
    from_date = dt.date(2023, 1, 1)
    window = 2

    result = compute._timeseries_check(ts, from_date, window)

    assert isinstance(result, Success)
    assert result.unwrap() == ts


def test_timeseries_check_missing_dates() -> None:
    """Test _timeseries_check with missing dates."""
    ts = {
        dt.date(2023, 1, 1): 100.0,
        # Missing dt.date(2023, 1, 2)
    }
    from_date = dt.date(2023, 1, 1)
    window = 2

    result = compute._timeseries_check(ts, from_date, window)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "There are 1 dates with missing metrics" in error_msg
    assert "2023-01-02" in error_msg


def test_timeseries_check_multiple_missing_dates() -> None:
    """Test _timeseries_check with multiple missing dates."""
    ts: dict[dt.date, float] = {
        dt.date(2023, 1, 1): 100.0,
        # Missing multiple dates
    }
    from_date = dt.date(2023, 1, 1)
    window = 5

    result = compute._timeseries_check(ts, from_date, window, limit=3)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "There are 4 dates with missing metrics" in error_msg


def test_timeseries_check_limit_exceeded() -> None:
    """Test _timeseries_check respects the limit parameter."""
    ts: dict[dt.date, float] = {}  # Empty timeseries
    from_date = dt.date(2023, 1, 1)
    window = 10
    limit = 2

    result = compute._timeseries_check(ts, from_date, window, limit)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "There are 10 dates with missing metrics" in error_msg
    # Should only show first 2 dates due to limit
    error_msg_parts = error_msg.split(": ")[1].split(".")
    listed_dates = error_msg_parts[0].split(", ")
    assert len(listed_dates) == limit


def test_simple_metric_success(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test simple_metric with successful retrieval."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    mock_db.get_metric_value.return_value = Some(42.5)  # type: ignore[attr-defined]

    result = compute.simple_metric(mock_db, mock_metric, mock_key_provider, mock_result_key)

    assert isinstance(result, Success)
    assert result.unwrap() == 42.5
    mock_key_provider.create.assert_called_once_with(mock_result_key)  # type: ignore[attr-defined]
    mock_db.get_metric_value.assert_called_once_with(mock_metric, mock_result_key)  # type: ignore[attr-defined]


def test_simple_metric_not_found(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test simple_metric when metric is not found."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    mock_db.get_metric_value.return_value = Nothing  # type: ignore[attr-defined]

    result = compute.simple_metric(mock_db, mock_metric, mock_key_provider, mock_result_key)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "Metric test_metric not found" in error_msg


def test_day_over_day_success(
    mock_db: MetricDB,
    mock_metric: MetricSpec,
    mock_key_provider: ResultKeyProvider,
    mock_result_key: ResultKey,
    sample_timeseries: TimeSeries,
) -> None:
    """Test day_over_day with successful calculation."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    mock_db.get_metric_window.return_value = Some(sample_timeseries)  # type: ignore[attr-defined]

    # Mock the lag method to return proper dates
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2022, 12, 31)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    result = compute.day_over_day(mock_db, mock_metric, mock_key_provider, mock_result_key)

    assert isinstance(result, Success)
    # Expected: 100.0 / 95.0 ≈ 1.0526
    assert abs(result.unwrap() - (100.0 / 95.0)) < 1e-6
    mock_db.get_metric_window.assert_called_once_with(mock_metric, mock_result_key, lag=0, window=2)  # type: ignore[attr-defined]


def test_day_over_day_no_data(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test day_over_day when no data is available."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    mock_db.get_metric_window.return_value = Nothing  # type: ignore[attr-defined]

    result = compute.day_over_day(mock_db, mock_metric, mock_key_provider, mock_result_key)

    assert isinstance(result, Failure)
    assert result.failure() == compute.METRIC_NOT_FOUND


def test_day_over_day_missing_dates(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test day_over_day with missing dates."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    incomplete_ts = {dt.date(2023, 1, 1): 100.0}  # Missing previous day
    mock_db.get_metric_window.return_value = Some(incomplete_ts)  # type: ignore[attr-defined]

    # Mock the lag method
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2022, 12, 31)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    result = compute.day_over_day(mock_db, mock_metric, mock_key_provider, mock_result_key)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "dates with missing metrics" in error_msg


def test_day_over_day_divide_by_zero(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test day_over_day when previous day value is zero."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    zero_ts = {
        dt.date(2023, 1, 1): 100.0,
        dt.date(2022, 12, 31): 0.0,  # Zero value causes division by zero
    }
    mock_db.get_metric_window.return_value = Some(zero_ts)  # type: ignore[attr-defined]

    # Mock the lag method
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2022, 12, 31)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    result = compute.day_over_day(mock_db, mock_metric, mock_key_provider, mock_result_key)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "is zero" in error_msg
    assert "2022-12-31" in error_msg


def test_stddev_success(
    mock_db: MetricDB,
    mock_metric: MetricSpec,
    mock_key_provider: ResultKeyProvider,
    mock_result_key: ResultKey,
    sample_timeseries: TimeSeries,
) -> None:
    """Test stddev with successful calculation."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    mock_db.get_metric_window.return_value = Some(sample_timeseries)  # type: ignore[attr-defined]

    # Mock the lag method - the sample_timeseries starts from 2022-12-29, so lag should start there
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2022, 12, 29)  # First available date in sample_timeseries
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    lag = 3
    size = 4

    result = compute.stddev(mock_db, mock_metric, lag, size, mock_key_provider, mock_result_key)

    assert isinstance(result, Success)
    # Calculate expected stddev
    values = list(sample_timeseries.values())
    expected_stddev = np.std(values).item()
    assert abs(result.unwrap() - expected_stddev) < 1e-6
    mock_db.get_metric_window.assert_called_once_with(mock_metric, mock_result_key, lag, size)  # type: ignore[attr-defined]


def test_stddev_no_data(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test stddev when no data is available."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    mock_db.get_metric_window.return_value = Nothing  # type: ignore[attr-defined]

    lag = 1
    size = 5

    result = compute.stddev(mock_db, mock_metric, lag, size, mock_key_provider, mock_result_key)

    assert isinstance(result, Failure)
    assert result.failure() == compute.METRIC_NOT_FOUND


def test_stddev_missing_dates(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test stddev with missing dates."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    incomplete_ts = {
        dt.date(2023, 1, 1): 100.0,
        dt.date(2022, 12, 30): 90.0,
        # Missing dates in between
    }
    mock_db.get_metric_window.return_value = Some(incomplete_ts)  # type: ignore[attr-defined]

    # Mock the lag method
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2022, 12, 29)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    lag = 3
    size = 5

    result = compute.stddev(mock_db, mock_metric, lag, size, mock_key_provider, mock_result_key)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "dates with missing metrics" in error_msg


def test_stddev_single_value(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test stddev with a single value (edge case)."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    single_value_ts = {dt.date(2023, 1, 1): 100.0}
    mock_db.get_metric_window.return_value = Some(single_value_ts)  # type: ignore[attr-defined]

    # Mock the lag method
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2023, 1, 1)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    lag = 0
    size = 1

    result = compute.stddev(mock_db, mock_metric, lag, size, mock_key_provider, mock_result_key)

    assert isinstance(result, Success)
    # Standard deviation of a single value should be 0
    assert result.unwrap() == 0.0


def test_stddev_empty_values(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test stddev with empty timeseries values."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    empty_ts: dict[dt.date, float] = {}
    mock_db.get_metric_window.return_value = Some(empty_ts)  # type: ignore[attr-defined]

    # Mock the lag method
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2023, 1, 1)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    lag = 0
    size = 1

    # This should fail at the timeseries check stage since empty ts means missing dates
    result = compute.stddev(mock_db, mock_metric, lag, size, mock_key_provider, mock_result_key)

    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "dates with missing metrics" in error_msg


def test_constants() -> None:
    """Test that constants are defined correctly."""
    assert compute.METRIC_NOT_FOUND == "Metric not found in the metric database"


def test_timeseries_check_with_different_limits() -> None:
    """Test _timeseries_check with different limit values."""
    ts: dict[dt.date, float] = {}  # Empty timeseries
    from_date = dt.date(2023, 1, 1)
    window = 6

    # Test with default limit (5)
    result = compute._timeseries_check(ts, from_date, window)
    assert isinstance(result, Failure)
    error_msg = result.failure()
    assert "There are 6 dates with missing metrics" in error_msg
    # Should only show first 5 dates due to default limit
    dates_part = error_msg.split(": ")[1].split(".")[0]
    listed_dates = dates_part.split(", ")
    assert len(listed_dates) == 5

    # Test with custom limit
    result_custom = compute._timeseries_check(ts, from_date, window, limit=3)
    assert isinstance(result_custom, Failure)
    error_msg_custom = result_custom.failure()
    dates_part_custom = error_msg_custom.split(": ")[1].split(".")[0]
    listed_dates_custom = dates_part_custom.split(", ")
    assert len(listed_dates_custom) == 3


def test_day_over_day_edge_cases(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test day_over_day with various edge cases."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]

    # Test with very small numbers (near zero but not zero)
    small_ts = {
        dt.date(2023, 1, 1): 0.001,
        dt.date(2022, 12, 31): 0.0001,
    }
    mock_db.get_metric_window.return_value = Some(small_ts)  # type: ignore[attr-defined]

    # Mock the lag method
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2022, 12, 31)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    result = compute.day_over_day(mock_db, mock_metric, mock_key_provider, mock_result_key)

    assert isinstance(result, Success)
    # Should be 0.001 / 0.0001 = 10.0
    assert abs(result.unwrap() - 10.0) < 1e-6


def test_stddev_with_identical_values(
    mock_db: MetricDB, mock_metric: MetricSpec, mock_key_provider: ResultKeyProvider, mock_result_key: ResultKey
) -> None:
    """Test stddev when all values are identical."""
    # Setup
    mock_key_provider.create.return_value = mock_result_key  # type: ignore[attr-defined]
    identical_ts = {
        dt.date(2023, 1, 1): 100.0,
        dt.date(2022, 12, 31): 100.0,
        dt.date(2022, 12, 30): 100.0,
    }
    mock_db.get_metric_window.return_value = Some(identical_ts)  # type: ignore[attr-defined]

    # Mock the lag method
    lag_key = MagicMock()
    lag_key.yyyy_mm_dd = dt.date(2022, 12, 30)
    mock_result_key.lag.return_value = lag_key  # type: ignore[attr-defined]

    lag = 2
    size = 3

    result = compute.stddev(mock_db, mock_metric, lag, size, mock_key_provider, mock_result_key)

    assert isinstance(result, Success)
    # Standard deviation of identical values should be 0
    assert result.unwrap() == 0.0
