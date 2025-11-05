"""Additional tests to improve coverage for analyzer.py."""

from datetime import date
from typing import Mapping, Sequence
from unittest.mock import Mock, patch

import pytest

from dqx.analyzer import AnalysisReport, Analyzer, analyze_sql_ops
from dqx.common import DQXError, ResultKey, SqlDataSource
from dqx.models import Metric
from dqx.ops import Sum
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider
from dqx.specs import MetricSpec
from dqx.states import SimpleAdditiveState


def test_analyzer_metrics_without_analyzers() -> None:
    """Test analyzing metrics that have no analyzers."""
    # Create a mock data source
    ds = Mock(spec=SqlDataSource)
    ds.name = "test_dataset"

    # Create a mock metric with no analyzers
    metric = Mock(spec=MetricSpec)
    metric.analyzers = []  # Empty analyzers list
    metric.name = "test_metric"

    # Mock the state() method to return a real state
    metric.state = Mock(return_value=SimpleAdditiveState(value=0.0))

    # Create analyzer with proper dependencies
    datasources: list[SqlDataSource] = [ds]
    mock_db = Mock()
    provider = MetricProvider(mock_db, execution_id="test-123")
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})

    analyzer = Analyzer(datasources, provider, key, "test-123")

    # Analyze with metrics that have no analyzers
    with patch.object(analyzer, "_analyze_internal") as mock_analyze:
        # Create a proper metric in the report
        result_metric = Metric.build(
            metric=metric, key=key, dataset="test_dataset", state=SimpleAdditiveState(value=0.0)
        )
        mock_analyze.return_value = AnalysisReport({(metric, key, "test_dataset"): result_metric})

        report = analyzer.analyze_simple_metrics(ds, {key: [metric]})

    # Should return a report with the metric
    assert len(report) == 1
    assert (metric, key, "test_dataset") in report


def test_persist_empty_report() -> None:
    """Test persisting an empty analysis report - covers line 115."""
    # Create empty report
    report = AnalysisReport()

    # Verify it's empty
    assert len(report) == 0

    # Create mock database
    db = Mock(spec=MetricDB)
    cache = Mock()

    # Persist empty report - should log warning and not call db.persist
    with patch("dqx.analyzer.logger") as mock_logger:
        report.persist(db, cache)
        mock_logger.warning.assert_called_once_with("Try to save an EMPTY analysis report!")

    # persist should not be called on db
    db.persist.assert_not_called()
    cache.put.assert_not_called()


def test_analyze_batch_with_more_than_4_dates() -> None:
    """Test batch analysis with more than 4 dates - covers lines 319-321."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)
    ds.dialect = "duckdb"
    ds.name = "test_ds"

    # Create 6 dates to trigger the special logging
    dates = [date(2024, 1, i) for i in range(1, 7)]
    keys = [ResultKey(yyyy_mm_dd=d, tags={}) for d in dates]

    # Create mock metrics
    metric = Mock(spec=MetricSpec)
    metric.analyzers = []
    metric.name = "test_metric"
    metric.state = Mock(return_value=SimpleAdditiveState(value=0.0))

    # Create metrics dict
    metrics = {key: [metric] for key in keys}

    # Create analyzer
    datasources: list[SqlDataSource] = [ds]
    mock_db = Mock()
    provider = MetricProvider(mock_db, execution_id="test-123")
    analyzer = Analyzer(datasources, provider, keys[0], "test-123")

    # Mock _analyze_internal to return proper report
    with patch.object(analyzer, "_analyze_internal") as mock_analyze:
        mock_analyze.return_value = AnalysisReport()
        result = analyzer.analyze_simple_metrics(ds, metrics)

    # Verify result
    assert isinstance(result, AnalysisReport)


def test_analyze_batch_with_large_date_range() -> None:
    """Test batch analysis with more than DEFAULT_BATCH_SIZE dates - covers lines 326 and 343-344."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)
    ds.dialect = "duckdb"
    ds.name = "test_ds"

    # Create 10 dates (more than DEFAULT_BATCH_SIZE=7)
    dates = [date(2024, 1, i) for i in range(1, 11)]
    keys = [ResultKey(yyyy_mm_dd=d, tags={}) for d in dates]

    # Create mock metrics without SqlOp analyzers to avoid SQL execution
    metrics = {}
    for key in keys:
        metric = Mock(spec=MetricSpec)
        metric.analyzers = []  # No analyzers to avoid SQL execution
        metric.name = "test_metric"
        metric.state = Mock(return_value=SimpleAdditiveState(value=0.0))
        metrics[key] = [metric]

    # Create analyzer
    datasources: list[SqlDataSource] = [ds]
    mock_db = Mock()
    provider = MetricProvider(mock_db, execution_id="test-123")
    analyzer = Analyzer(datasources, provider, keys[0], "test-123")

    # Mock _analyze_internal to return proper report
    with patch.object(analyzer, "_analyze_internal") as mock_analyze:
        mock_analyze.return_value = AnalysisReport()
        result = analyzer.analyze_simple_metrics(ds, metrics)

    # Verify result
    assert isinstance(result, AnalysisReport)


def test_analyze_batch_sql_ops_value_retrieval_failure() -> None:
    """Test analyze_batch_sql_ops when value retrieval fails - covers lines 420-421."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)
    ds.dialect = "duckdb"
    ds.cte = Mock(return_value="WITH data AS (...)")
    ds.name = "test_ds"

    # Create mock query result with array format - the op's sql_col won't be in the array
    mock_result = Mock()
    # Return a result that has date and array, but the array doesn't contain our metric
    mock_result.fetchall.return_value = [("2024-01-01", [{"key": "other_metric", "value": 123.0}])]
    ds.query = Mock(return_value=mock_result)

    # Create a real SqlOp and a metric that uses it
    sql_op = Sum(column="test_col")

    # Create a mock metric with the SqlOp as an analyzer
    metric = Mock(spec=MetricSpec)
    metric.analyzers = [sql_op]
    metric.name = "test_metric"
    metric.state = Mock(return_value=SimpleAdditiveState(value=0.0))

    # Create analyzer
    datasources: list[SqlDataSource] = [ds]
    mock_db = Mock()
    provider = MetricProvider(mock_db, execution_id="test-123")
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 1), tags={})
    analyzer = Analyzer(datasources, provider, key, "test-123")

    # Create metrics dict
    metrics: dict[ResultKey, list[MetricSpec]] = {key: [metric]}

    # Mock dialect to return proper SQL
    with patch("dqx.analyzer.get_dialect") as mock_get_dialect:
        mock_dialect = Mock()
        mock_dialect.build_batch_cte_query.return_value = "BATCH SQL"
        mock_get_dialect.return_value = mock_dialect

        # Call _analyze_internal which will check for value retrieval
        # This should raise DQXError when it tries to get the value
        # Cast to proper type for mypy
        metrics_for_analyze: Mapping[ResultKey, Sequence[MetricSpec]] = metrics
        with pytest.raises(DQXError) as exc_info:
            analyzer._analyze_internal(ds, dict(metrics_for_analyze))

        # Check the error message
        assert "Failed to retrieve value for analyzer" in str(exc_info.value)
        assert "on date 2024-01-01" in str(exc_info.value)


def test_analyze_batch_sql_ops_with_empty_ops() -> None:
    """Test analyze_batch_sql_ops with empty ops_by_key - covers line 214."""
    # Create mock data source
    ds = Mock(spec=SqlDataSource)

    # Call analyze_batch_sql_ops with empty dict
    analyze_sql_ops(ds, {})

    # Should return early without doing anything
    # No query should be made
    ds.query.assert_not_called()
    ds.cte.assert_not_called()
