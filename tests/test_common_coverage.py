"""Additional tests to improve coverage for common.py."""

from datetime import date

from dqx.common import ResultKey


def test_result_key_str_method() -> None:
    """Test ResultKey.__str__ method - covers line 131."""
    # Create a ResultKey
    key = ResultKey(yyyy_mm_dd=date(2024, 1, 15), tags={"env": "prod", "region": "us-east"})

    # Test __str__ method
    str_repr = str(key)

    # Should call __repr__ internally
    assert str_repr == "ResultKey(2024-01-15, {'env': 'prod', 'region': 'us-east'})"

    # Verify that __str__ returns same as __repr__
    assert str(key) == repr(key)
