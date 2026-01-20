"""Tests for tunable bounds validation - numeric literals only.

This test module verifies that tunable bounds are restricted to numeric literals
(NUMBER tokens) only. Arithmetic expressions, percentages, tunable references, and
function calls should fail at parse time.
"""

from __future__ import annotations

import pytest

from dqx.dql import parse
from dqx.dql.errors import DQLSyntaxError


class TestTunableBoundsValidation:
    """Tests for tunable bounds validation rules."""

    def test_tunable_with_numeric_literal_bounds(self) -> None:
        """Test tunable with valid numeric literal bounds."""
        dql = """
        suite "test" {
            tunable X = 5 bounds [0, 10]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        ast = parse(dql)
        assert len(ast.tunables) == 1
        assert ast.tunables[0].name == "X"

    def test_tunable_with_float_literal_bounds(self) -> None:
        """Test tunable with float literal bounds."""
        dql = """
        suite "test" {
            tunable Y = 0.5 bounds [0.0, 1.0]

            check "simple" on my_table {
                assert Y > 0 name "positive"
            }
        }
        """
        ast = parse(dql)
        assert len(ast.tunables) == 1
        assert ast.tunables[0].name == "Y"

    def test_tunable_with_zero_lower_bound(self) -> None:
        """Test tunable with zero as lower bound."""
        dql = """
        suite "test" {
            tunable Z = 5 bounds [0, 10]

            check "simple" on my_table {
                assert Z > 0 name "positive"
            }
        }
        """
        ast = parse(dql)
        assert len(ast.tunables) == 1
        assert ast.tunables[0].name == "Z"

    def test_tunable_with_integer_literal_bounds(self) -> None:
        """Test tunable with integer literal bounds."""
        dql = """
        suite "test" {
            tunable COUNT = 100 bounds [0, 1000]

            check "simple" on my_table {
                assert COUNT > 0 name "positive"
            }
        }
        """
        ast = parse(dql)
        assert len(ast.tunables) == 1
        assert ast.tunables[0].name == "COUNT"

    def test_tunable_with_arithmetic_in_bounds_fails(self) -> None:
        """Test tunable with arithmetic in bounds fails at parse time."""
        dql = """
        suite "test" {
            tunable X = 5 bounds [1 + 1, 10]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_multiplication_in_bounds_fails(self) -> None:
        """Test tunable with multiplication in bounds fails."""
        dql = """
        suite "test" {
            tunable X = 5 bounds [2 * 3, 10]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_tunable_reference_in_bounds_fails(self) -> None:
        """Test tunable with tunable reference in bounds fails."""
        dql = """
        suite "test" {
            tunable BASE = 10 bounds [5, 20]
            tunable SCALED = 20 bounds [BASE * 2, 100]

            check "simple" on my_table {
                assert BASE > 0 name "positive"
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_identifier_in_bounds_fails(self) -> None:
        """Test tunable with identifier reference in bounds fails."""
        dql = """
        suite "test" {
            tunable X = 10 bounds [MIN_VAL, 100]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_function_call_in_bounds_fails(self) -> None:
        """Test tunable with function call in bounds fails."""
        dql = """
        suite "test" {
            tunable X = 5 bounds [min(1, 2), 10]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_percentage_in_value_fails(self) -> None:
        """Test tunable with percentage in value fails."""
        dql = """
        suite "test" {
            tunable X = 5% bounds [0, 10]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        # Percentage in value should fail - only NUMBER allowed for tunable values
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_percentage_in_bounds_fails(self) -> None:
        """Test tunable with percentage in bounds fails."""
        dql = """
        suite "test" {
            tunable X = 5 bounds [0%, 10%]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        # Percentage in bounds should fail - only NUMBER allowed
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_mixed_numeric_percent_bounds_fails(self) -> None:
        """Test tunable with mixed numeric and percent bounds fails."""
        dql = """
        suite "test" {
            tunable X = 5 bounds [0, 10%]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_tunable_with_complex_arithmetic_in_bounds_fails(self) -> None:
        """Test tunable with complex arithmetic expression in bounds fails."""
        dql = """
        suite "test" {
            tunable BASE = 10 bounds [5, 20]
            tunable DERIVED = 20 bounds [BASE + 5, BASE * 10]

            check "simple" on my_table {
                assert BASE > 0 name "positive"
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(dql)

    def test_multiple_tunables_with_numeric_literals(self) -> None:
        """Test multiple tunables all using numeric literals."""
        dql = """
        suite "test" {
            tunable BASE = 10 bounds [5, 20]
            tunable SCALED = 20 bounds [10, 40]

            check "simple" on my_table {
                assert BASE > 0 name "positive"
            }
        }
        """
        # All tunables use only numeric literals (no references, no arithmetic)
        ast = parse(dql)
        assert len(ast.tunables) == 2
        assert ast.tunables[0].name == "BASE"
        assert ast.tunables[1].name == "SCALED"

    def test_assertion_threshold_can_use_percentage(self) -> None:
        """Test that assertion thresholds can still use percentages."""
        dql = """
        suite "test" {
            tunable X = 0.05 bounds [0.0, 0.1]

            check "percentage test" on my_table {
                assert null_rate <= 5% name "null rate check"
            }
        }
        """
        # Percentages allowed in assertions, not in tunable bounds
        ast = parse(dql)
        assert len(ast.checks) == 1

    def test_availability_threshold_can_use_percentage(self) -> None:
        """Test that availability_threshold can still use percentages."""
        dql = """
        suite "test" {
            availability_threshold 85%

            tunable X = 0.05 bounds [0.0, 0.1]

            check "simple" on my_table {
                assert X > 0 name "positive"
            }
        }
        """
        # Percentages allowed in availability_threshold
        ast = parse(dql)
        assert ast.availability_threshold is not None
