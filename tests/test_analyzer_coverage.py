"""Additional tests to improve coverage for analyzer.py."""

from datetime import date
from unittest.mock import Mock

import pytest

from dqx.analyzer import AnalysisReport, Analyzer, analyze_batch_sql_ops
from dqx.common import DQXError, ResultKey, SqlDataSource
from dqx.ops import Sum
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec


def test_analyzer_metrics_without_analyzers() -> None:
    """Test analyzing metrics that have no analyzers."""
    # Create a mock data source
    ds = Mock(spec=SqlDataSource)
    ds.name = "test_dataset"  # Add dataset name

    # Create a mock metric with no analyzers
    metric = Mock(spec=MetricSpec)
    metric.analyzers = []  # Empty analyzers list

    # Mock the state() method to return a mock state
    mock_state = Mock()
    metric.state = Mock(return_value=mock_state)

    # Create analyzer
    analyzer = Analyzer()

    # Create result key
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})

    # Analyze with metrics that have no analyzers
    report = analyzer.analyze(ds, {key: [metric]})

    # Should return a report with the metric (even if no analyzers)
    # The analyzer now processes all metrics and creates states for them
    assert len(report) == 1
    # Use 3-tuple key format: (metric, key, dataset)
    assert (metric, key, "test_dataset") in report

    # Verify the metric in the report
    # Access using the metric_key variable to avoid mypy errors with Mock
    metric_key = (metric, key, "test_dataset")
    result_metric = report[metric_key]  # type: ignore[index]
    assert result_metric.spec == metric
    assert result_metric.state == mock_state
    assert result_metric.key == key


def test_persist_empty_report() -> None:
    """Test persisting an empty analysis report - covers line 115."""
    # Create empty report
    report = AnalysisReport()

    # Verify it's empty
    assert len(report) == 0

    # Create mock database
    db = Mock(spec=MetricDB)

    # Persist empty report
    report.persist(db)

    # persist should not be called on db
    db.persist.assert_not_called()


def test_analyze_batch_with_more_than_4_dates() -> None:
    """Test batch analysis with more than 4 dates - covers lines 319-321."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)
    ds.dialect = "duckdb"

    # Create 6 dates to trigger the special logging
    dates = [date(2024, 1, i) for i in range(1, 7)]
    keys = [ResultKey(yyyy_mm_dd=d, tags={}) for d in dates]

    # Create mock metrics
    metric = Mock(spec=MetricSpec)
    metric.analyzers = []
    mock_state = Mock()
    metric.state = Mock(return_value=mock_state)

    # Create metrics dict
    metrics = {key: [metric] for key in keys}

    # Create analyzer
    analyzer = Analyzer()

    # Analyze - should work without checking logs
    result = analyzer.analyze(ds, metrics)

    # Verify result has expected metrics
    assert len(result) == 6  # One metric per date


def test_analyze_batch_with_large_date_range() -> None:
    """Test batch analysis with more than DEFAULT_BATCH_SIZE dates - covers lines 326 and 343-344."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)
    ds.dialect = "duckdb"

    # Create 10 dates (more than DEFAULT_BATCH_SIZE=7)
    dates = [date(2024, 1, i) for i in range(1, 11)]
    keys = [ResultKey(yyyy_mm_dd=d, tags={}) for d in dates]

    # Create mock metrics without SqlOp analyzers to avoid SQL execution
    metrics = {}
    for key in keys:
        metric = Mock(spec=MetricSpec)
        metric.analyzers = []  # No analyzers to avoid SQL execution
        mock_state = Mock()
        metric.state = Mock(return_value=mock_state)
        metrics[key] = [metric]

    # Create analyzer
    analyzer = Analyzer()

    # Analyze - should work without checking logs
    result = analyzer.analyze(ds, metrics)

    # Verify result has expected metrics
    assert len(result) == 10  # One metric per date


def test_analyze_batch_sql_ops_value_retrieval_failure() -> None:
    """Test analyze_batch_sql_ops when value retrieval fails - covers lines 420-421."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)
    ds.dialect = "duckdb"
    ds.cte = Mock(return_value="WITH data AS (...)")

    # Create mock query result with MAP format - the op's sql_col won't be in the MAP
    mock_result = Mock()
    # Return a result that has date and MAP, but the MAP doesn't contain our metric
    mock_result.fetchall.return_value = [("2024-01-01", {"other_metric": 123.0})]
    ds.query = Mock(return_value=mock_result)

    # Create a real SqlOp and a metric that uses it
    sql_op = Sum(column="test_col")

    # Create a mock metric with the SqlOp as an analyzer
    metric = Mock(spec=MetricSpec)
    metric.analyzers = [sql_op]

    # Create analyzer
    analyzer = Analyzer()

    # Create metrics dict
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
    metrics: dict[ResultKey, list[MetricSpec]] = {key: [metric]}

    # Call _analyze_internal which will check for value retrieval
    # This should raise DQXError when it tries to get the value
    with pytest.raises(DQXError) as exc_info:
        analyzer._analyze_internal(ds, metrics)  # type: ignore[arg-type]

    # Check the error message
    assert "Failed to retrieve value for analyzer" in str(exc_info.value)
    assert "on date 2024-01-01" in str(exc_info.value)


def test_analyze_batch_sql_ops_with_empty_ops() -> None:
    """Test analyze_batch_sql_ops with empty ops_by_key - covers line 214."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)

    # Call analyze_batch_sql_ops with empty dict
    analyze_batch_sql_ops(ds, {})

    # Should return early without doing anything
    # No query should be made
    ds.query.assert_not_called()
    ds.cte.assert_not_called()
