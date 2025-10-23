"""Test coverage for _validate_value function in analyzer.py."""

import numpy as np
import pytest

from dqx.analyzer import _validate_value
from dqx.common import DQXError


class TestValidateValue:
    """Test the _validate_value function."""

    def test_validate_value_with_masked(self) -> None:
        """Test that masked values raise DQXError."""
        masked_value = np.ma.masked

        with pytest.raises(DQXError, match="Masked value encountered for symbol 'test_col' on date 2024-01-01"):
            _validate_value(masked_value, "2024-01-01", "test_col")

    def test_validate_value_with_none(self) -> None:
        """Test that None values raise DQXError."""
        with pytest.raises(DQXError, match="Null value encountered for symbol 'test_col' on date 2024-01-01"):
            _validate_value(None, "2024-01-01", "test_col")

    def test_validate_value_with_invalid_type(self) -> None:
        """Test that non-convertible values raise DQXError."""
        # Test with a string that can't be converted to float
        with pytest.raises(DQXError) as exc_info:
            _validate_value("not_a_number", "2024-01-01", "test_col")

        assert "Cannot convert value to float for symbol 'test_col' on date 2024-01-01" in str(exc_info.value)
        assert "Value: 'not_a_number'" in str(exc_info.value)
        assert "could not convert string to float" in str(exc_info.value)

        # Test with an object that causes TypeError
        class BadObject:
            def __float__(self) -> None:
                raise TypeError("Cannot convert to float")

        with pytest.raises(DQXError) as exc_info:
            _validate_value(BadObject(), "2024-01-01", "test_col")

        assert "Cannot convert value to float for symbol 'test_col' on date 2024-01-01" in str(exc_info.value)
        assert "Cannot convert to float" in str(exc_info.value)

    def test_validate_value_with_nan(self) -> None:
        """Test that NaN values raise DQXError."""
        with pytest.raises(DQXError, match="NaN value encountered for symbol 'test_col' on date 2024-01-01"):
            _validate_value(np.nan, "2024-01-01", "test_col")

    def test_validate_value_success(self) -> None:
        """Test that valid values are returned as floats."""
        # Test with integer
        assert _validate_value(42, "2024-01-01", "test_col") == 42.0

        # Test with float
        assert _validate_value(3.14, "2024-01-01", "test_col") == 3.14

        # Test with string that can be converted
        assert _validate_value("123.45", "2024-01-01", "test_col") == 123.45

        # Test with numpy float
        assert _validate_value(np.float64(99.9), "2024-01-01", "test_col") == 99.9
