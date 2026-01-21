"""Tests for DQL expression evaluation error handling in api.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from dqx.api import VerificationSuite
from dqx.dql.errors import DQLError, DQLSyntaxError
from dqx.orm.repositories import InMemoryMetricDB


class TestDQLEvalErrorHandling:
    """Tests for error handling in _eval_simple_expr method (covers api.py:1539-1540)."""

    def test_invalid_expression_syntax_error(self, tmp_path: Path) -> None:
        """Test error handling for invalid expression syntax during parsing."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_invalid_expr.dql"
        # Use invalid expression that will cause parser error
        dql_file.write_text("""
        suite "Invalid Expr Suite" {
            check "Invalid Check" on test_data {
                assert average(value) > @@@invalid@@@
                    name "Invalid expression"
            }
        }
        """)

        # Parser should raise DQLSyntaxError during suite creation
        with pytest.raises(DQLSyntaxError, match="Unexpected"):
            VerificationSuite(dql=dql_file, db=db)

    def test_undefined_symbol_error(self, tmp_path: Path) -> None:
        """Test error handling for undefined symbols in expressions (covers api.py:1539-1540)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_undefined_symbol.dql"
        # Using undefined symbol - will fail during evaluation
        dql_file.write_text("""
        suite "Undefined Symbol Suite" {
            check "Undefined Check" on test_data {
                assert average(value) > undefined_var
                    name "Undefined symbol"
            }
        }
        """)

        # Should raise DQLError during suite creation when building graph
        with pytest.raises(DQLError, match="Cannot evaluate expression"):
            VerificationSuite(dql=dql_file, db=db)

    def test_invalid_arithmetic_expression(self, tmp_path: Path) -> None:
        """Test error handling for invalid arithmetic expressions (covers api.py:1539-1540)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_invalid_arithmetic.dql"
        # Division by zero - SymPy evaluates to 'zoo' which can't be converted to float
        dql_file.write_text("""
        suite "Invalid Arithmetic Suite" {
            check "Invalid Arithmetic Check" on test_data {
                assert average(value) > (1/0)
                    name "Division by zero"
            }
        }
        """)

        # Should raise DQLError during suite creation
        with pytest.raises(DQLError, match="Cannot evaluate expression"):
            VerificationSuite(dql=dql_file, db=db)

    def test_valid_expression_evaluation(self, tmp_path: Path) -> None:
        """Test that valid expressions evaluate correctly during suite creation."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_valid_expr.dql"
        dql_file.write_text("""
        suite "Valid Expr Suite" {
            check "Valid Check" on test_data {
                assert average(value) > 100
                    name "Valid numeric expression"
            }
        }
        """)

        # Should succeed without raising DQLError
        suite = VerificationSuite(dql=dql_file, db=db)
        assert suite._name == "Valid Expr Suite"
