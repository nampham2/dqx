"""Test type updates for date exclusion feature."""

from typing import get_args

from dqx.common import AssertionStatus


class TestDateExclusionTypes:
    """Test core type updates for date exclusion."""

    def test_assertion_status_includes_skipped(self) -> None:
        """Verify AssertionStatus literal includes SKIPPED."""
        status_values = get_args(AssertionStatus)
        assert "OK" in status_values
        assert "FAILURE" in status_values
        assert "SKIPPED" in status_values
        assert len(status_values) == 3
