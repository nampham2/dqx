"""Test type updates for date exclusion feature."""

from typing import get_args

import sympy as sp
from returns.result import Success

from dqx import specs
from dqx.common import AssertionStatus
from dqx.provider import SymbolicMetric


class TestDateExclusionTypes:
    """Test core type updates for date exclusion."""

    def test_assertion_status_includes_skipped(self) -> None:
        """Verify AssertionStatus literal includes SKIPPED."""
        status_values = get_args(AssertionStatus)
        assert "OK" in status_values
        assert "FAILURE" in status_values
        assert "SKIPPED" in status_values
        assert len(status_values) == 3

    def test_symbolic_metric_has_data_av_ratio(self) -> None:
        """Verify SymbolicMetric has data_av_ratio field."""
        # Test default None
        metric = SymbolicMetric(
            name="test", symbol=sp.Symbol("x_1"), fn=lambda k: Success(1.0), metric_spec=specs.NumRows()
        )
        assert metric.data_av_ratio is None

        # Test explicit value
        metric2 = SymbolicMetric(
            name="test2",
            symbol=sp.Symbol("x_2"),
            fn=lambda k: Success(2.0),
            metric_spec=specs.Average("price"),
            data_av_ratio=0.75,
        )
        assert metric2.data_av_ratio == 0.75
