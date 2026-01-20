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


class TestPctAPIExport:
    """Tests for pct() API export."""

    def test_pct_available_from_api(self) -> None:
        """Test that pct() can be imported from dqx.api."""
        from dqx.api import pct as pct_from_api  # type: ignore[attr-defined]

        assert pct_from_api(5) == 0.05
        assert type(pct_from_api(5)) is float

    def test_pct_import_from_api(self) -> None:
        """Test importing pct from dqx.api works correctly."""
        from dqx.api import pct as api_pct  # type: ignore[attr-defined]
        from dqx.functions import pct as func_pct

        # Both should be the same function
        assert api_pct is func_pct
        assert api_pct(10) == 0.1


class TestPctDocumentation:
    """Tests for pct() documentation examples."""

    def test_pct_docstring_examples_work(self) -> None:
        """Verify all examples in pct() docstring are correct."""
        # Basic conversions from docstring
        assert pct(5) == 0.05
        assert pct(0.5) == 0.005
        assert pct(100) == 1.0
        assert pct(150) == 1.5
        assert pct(-10) == -0.1

        # Type verification
        assert type(pct(5)) is float
