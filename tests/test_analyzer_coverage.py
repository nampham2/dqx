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

    # Create analyzer
    analyzer = Analyzer()

    # Create result key
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})

    # Analyze with metrics that have no analyzers
    report = analyzer.analyze(ds, [metric], key)

    # Should return empty report
    assert len(report) == 0
    assert isinstance(report.data, dict)
    assert report.data == {}
