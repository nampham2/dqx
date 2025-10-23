"""Additional tests to improve coverage for analyzer.py."""

from datetime import date
from unittest.mock import Mock

from dqx.analyzer import Analyzer
from dqx.common import ResultKey, SqlDataSource
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
