"""E2E tests for DQL tolerance on all comparison operators in api.py."""

from __future__ import annotations

from pathlib import Path

from dqx.api import VerificationSuite
from dqx.orm.repositories import InMemoryMetricDB


class TestDQLToleranceIntegration:
    """E2E tests for tolerance on all comparison operators (>, >=, <, <=, !=, between)."""

    def test_tolerance_on_greater_than_operator(self, tmp_path: Path) -> None:
        """Test DQL assertion with tolerance on > operator (covers api.py:1807)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_gt_tolerance.dql"
        dql_file.write_text("""
        suite "GT Tolerance Suite" {
            check "GT Check" on test_data {
                assert average(price) > 100
                    tolerance 5
                    name "Price greater than 100 with tolerance"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        # Verify suite was created successfully with tolerance
        assert suite._name == "GT Tolerance Suite"
        assert len(suite._checks) == 1

    def test_tolerance_on_greater_or_equal_operator(self, tmp_path: Path) -> None:
        """Test DQL assertion with tolerance on >= operator (covers api.py:1813)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_geq_tolerance.dql"
        dql_file.write_text("""
        suite "GEQ Tolerance Suite" {
            check "GEQ Check" on test_data {
                assert average(value) >= 50
                    tolerance 3
                    name "Value greater or equal 50 with tolerance"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        assert suite._name == "GEQ Tolerance Suite"
        assert len(suite._checks) == 1

    def test_tolerance_on_less_than_operator(self, tmp_path: Path) -> None:
        """Test DQL assertion with tolerance on < operator (covers api.py:1819)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_lt_tolerance.dql"
        dql_file.write_text("""
        suite "LT Tolerance Suite" {
            check "LT Check" on test_data {
                assert average(amount) < 50
                    tolerance 10
                    name "Amount less than 50 with tolerance"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        assert suite._name == "LT Tolerance Suite"
        assert len(suite._checks) == 1

    def test_tolerance_on_less_or_equal_operator(self, tmp_path: Path) -> None:
        """Test DQL assertion with tolerance on <= operator (covers api.py:1825)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_leq_tolerance.dql"
        dql_file.write_text("""
        suite "LEQ Tolerance Suite" {
            check "LEQ Check" on test_data {
                assert average(score) <= 100
                    tolerance 5
                    name "Score less or equal 100 with tolerance"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        assert suite._name == "LEQ Tolerance Suite"
        assert len(suite._checks) == 1

    def test_tolerance_on_not_equal_operator(self, tmp_path: Path) -> None:
        """Test DQL assertion with tolerance on != operator (covers api.py:1837)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_neq_tolerance.dql"
        dql_file.write_text("""
        suite "NEQ Tolerance Suite" {
            check "NEQ Check" on test_data {
                assert average(value) != 0
                    tolerance 0.5
                    name "Value not equal zero with tolerance"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        assert suite._name == "NEQ Tolerance Suite"
        assert len(suite._checks) == 1

    def test_tolerance_on_between_operator(self, tmp_path: Path) -> None:
        """Test DQL assertion with tolerance on between operator (covers api.py:1849)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_between_tolerance.dql"
        dql_file.write_text("""
        suite "Between Tolerance Suite" {
            check "Between Check" on test_data {
                assert num_rows() between 3 and 10
                    tolerance 2
                    name "Row count in range with tolerance"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        assert suite._name == "Between Tolerance Suite"
        assert len(suite._checks) == 1

    def test_all_operators_with_tolerance_comprehensive(self, tmp_path: Path) -> None:
        """Comprehensive test of all operators with tolerance."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_all_tolerance.dql"
        dql_file.write_text("""
        suite "Comprehensive Tolerance Suite" {
            check "All Operators" on test_data {
                assert average(price) > 100
                    tolerance 5
                    name "GT with tolerance"

                assert average(price) >= 100
                    tolerance 5
                    name "GEQ with tolerance"

                assert average(price) < 200
                    tolerance 10
                    name "LT with tolerance"

                assert average(price) <= 200
                    tolerance 10
                    name "LEQ with tolerance"

                assert average(price) == 120
                    tolerance 5
                    name "EQ with tolerance"

                assert average(price) != 0
                    tolerance 1
                    name "NEQ with tolerance"

                assert num_rows() between 3 and 10
                    tolerance 2
                    name "Between with tolerance"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        assert suite._name == "Comprehensive Tolerance Suite"
        assert len(suite._checks) == 1

    def test_operators_without_tolerance_backward_compatibility(self, tmp_path: Path) -> None:
        """Test operators without tolerance to ensure backward compatibility."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test_no_tolerance.dql"
        dql_file.write_text("""
        suite "No Tolerance Suite" {
            check "No Tolerance Check" on test_data {
                assert average(value) > 10 name "GT"
                assert average(value) >= 15 name "GEQ"
                assert average(value) < 30 name "LT"
                assert average(value) <= 25 name "LEQ"
                assert average(value) != 0 name "NEQ"
                assert num_rows() between 2 and 5 name "Between"
            }
        }
        """)

        suite = VerificationSuite(dql=dql_file, db=db)

        assert suite._name == "No Tolerance Suite"
        assert len(suite._checks) == 1
