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
    assert (metric, key) in report

    # Verify the metric in the report
    result_metric = report[(metric, key)]
    assert result_metric.spec == metric
    assert result_metric.state == mock_state
    assert result_metric.key == key


def test_persist_empty_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Test persisting an empty analysis report - covers line 115."""
    # Create empty report
    report = AnalysisReport()

    # Verify it's empty
    assert len(report) == 0

    # Create mock database
    db = Mock(spec=MetricDB)

    # Persist empty report
    report.persist(db)

    # Should log warning and return early
    captured = capsys.readouterr()
    assert "Try to save an EMPTY analysis report!" in captured.out
    # persist should not be called on db
    db.persist.assert_not_called()


def test_analyze_batch_with_more_than_4_dates(capsys: pytest.CaptureFixture[str]) -> None:
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

    # Analyze and capture output
    _ = analyzer.analyze(ds, metrics)

    # Check that the special log message was generated
    captured = capsys.readouterr()

    # Strip ANSI color codes for simpler matching
    import re

    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)

    # Should log with first 2 and last 2 dates - check for parts of the message
    # since the output may be formatted across multiple lines
    assert "Analyzing batch of 6 dates:" in clean_output
    assert "['2024-01-01', '2024-01-02']" in clean_output
    # The last part might be split across lines, so check for the dates separately
    assert "'2024-01-05'" in clean_output
    assert "'2024-01-06'" in clean_output


def test_analyze_batch_with_large_date_range(capsys: pytest.CaptureFixture[str]) -> None:
    """Test batch analysis with more than DEFAULT_BATCH_SIZE dates - covers lines 326 and 343-344."""
    # We need to create a simpler test that just checks for the log messages
    # without actually executing the SQL operations

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

    # Analyze and capture output
    _ = analyzer.analyze(ds, metrics)

    # Check debug logs for batch processing
    captured = capsys.readouterr()

    # Strip ANSI color codes for simpler matching
    import re

    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)

    # Should log the initial batch info
    assert "Processing 10 dates in batches of 7" in clean_output

    # Should log batch boundaries
    assert "Processing batch 1: 2024-01-01 to 2024-01-07 (7 dates)" in clean_output
    assert "Processing batch 2: 2024-01-08 to 2024-01-10 (3 dates)" in clean_output


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
