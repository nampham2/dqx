"""Tests for pct() helper function."""

from __future__ import annotations

import sympy as sp

from dqx.functions import pct


class TestPct:
    """Tests for pct() percentage conversion helper."""

    def test_pct_basic_conversion(self) -> None:
        """Test basic percentage conversion."""
        assert pct(5) == 0.05

    def test_pct_with_integer_input(self) -> None:
        """Test pct() with integer input."""
        result = pct(10)
        assert result == 0.1
        assert isinstance(result, float)

    def test_pct_with_float_input(self) -> None:
        """Test pct() with float input."""
        result = pct(5.5)
        assert result == 0.055
        assert isinstance(result, float)

    def test_pct_with_decimal_percentage(self) -> None:
        """Test pct() with sub-percent decimal values."""
        assert pct(0.5) == 0.005
        assert pct(0.1) == 0.001

    def test_pct_with_large_value(self) -> None:
        """Test pct() with values over 100 (e.g., growth rates)."""
        assert pct(150) == 1.5
        assert pct(200) == 2.0
        assert pct(100) == 1.0

    def test_pct_with_negative_value(self) -> None:
        """Test pct() with negative percentages (e.g., declines)."""
        assert pct(-10) == -0.1
        assert pct(-5) == -0.05

    def test_pct_with_zero(self) -> None:
        """Test pct() with zero."""
        assert pct(0) == 0.0
        assert isinstance(pct(0), float)

    def test_pct_returns_float_type(self) -> None:
        """Test that pct() returns exact float type, not subclass."""
        result = pct(5)
        assert type(result) is float  # Exact type check
        assert isinstance(result, float)

    def test_pct_not_sympy_type(self) -> None:
        """Test that pct() does NOT return SymPy type."""
        result = pct(5)
        assert not isinstance(result, sp.Basic)
        assert not isinstance(result, sp.Expr)
