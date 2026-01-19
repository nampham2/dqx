"""Tests for DQL arithmetic expression evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dqx.api import VerificationSuite
from dqx.dql.errors import DQLError
from dqx.orm.repositories import InMemoryMetricDB


class TestDQLArithmeticEvaluation:
    """Test arithmetic expression evaluation in DQL tunables."""

    def test_tunable_with_arithmetic_in_bounds(self, tmp_path: Path) -> None:
        """Test tunable with arithmetic expressions in bounds."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Arithmetic Suite" {
            tunable BASE = 10 bounds [1, 100]
            tunable DERIVED = 20 bounds [BASE + 5, BASE * 10]

            check "Check" on dataset {
                assert num_rows() >= DERIVED
                    name "test"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        # Suite should be created successfully with arithmetic evaluation
        assert suite._name == "Arithmetic Suite"
        assert len(suite._checks) == 1

    def test_tunable_with_percentage(self, tmp_path: Path) -> None:
        """Test tunable with percentage value."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Percentage Suite" {
            tunable THRESHOLD = 50% bounds [0%, 100%]

            check "Check" on dataset {
                assert null_count(col) <= THRESHOLD
                    name "test"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert suite._name == "Percentage Suite"
        assert len(suite._checks) == 1

    def test_tunable_with_invalid_expression(self, tmp_path: Path) -> None:
        """Test that invalid expressions raise DQLError."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Invalid Expression" {
            tunable BAD = invalid_func(10) bounds [0, 100]

            check "Check" on dataset {
                assert num_rows() > 0
                    name "test"
            }
        }
        """)

        # Should raise DQLError during tunable evaluation
        with pytest.raises(DQLError, match="Cannot evaluate expression"):
            VerificationSuite(
                dql=dql_file,
                db=db,
            )

    def test_tunable_with_complex_arithmetic(self, tmp_path: Path) -> None:
        """Test tunable with complex arithmetic expressions."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Complex Math" {
            tunable X = 5 bounds [1, 10]
            tunable Y = (X * 2) + 3 bounds [X, X * 5]

            check "Check" on dataset {
                assert num_rows() >= Y
                    name "test"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        # Y should be evaluated as (5 * 2) + 3 = 13
        assert suite._name == "Complex Math"
        assert len(suite._checks) == 1
