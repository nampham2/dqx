"""Tests for DQL tolerance modifier on all comparison operators."""

from __future__ import annotations

from dqx.dql import parse
from dqx.dql.ast import Assertion


class TestToleranceOnBasicComparisons:
    """Test tolerance modifier on basic comparison operators (>, >=, <, <=)."""

    def test_tolerance_parses_on_greater_than(self) -> None:
        """Test tolerance modifier parses on > operator."""
        dql = """
        suite "Test Suite" {
            check "GT Check" on test_data {
                assert average(price) > 100
                    tolerance 5
                    name "Price greater than"
            }
        }
        """
        suite = parse(dql)
        check = suite.checks[0]
        statement = check.assertions[0]
        assert isinstance(statement, Assertion), "Should be an Assertion"
        assert statement.tolerance == 5, "Tolerance should be parsed as 5"
        assert statement.condition == ">", "Condition should be >"

    def test_tolerance_parses_on_greater_or_equal(self) -> None:
        """Test tolerance modifier parses on >= operator."""
        dql = """
        suite "Test Suite" {
            check "GEQ Check" on test_data {
                assert average(price) >= 100
                    tolerance 5
                    name "Price greater or equal"
            }
        }
        """
        suite = parse(dql)
        check = suite.checks[0]
        statement = check.assertions[0]
        assert isinstance(statement, Assertion)
        assert statement.tolerance == 5
        assert statement.condition == ">="

    def test_tolerance_parses_on_less_than(self) -> None:
        """Test tolerance modifier parses on < operator."""
        dql = """
        suite "Test Suite" {
            check "LT Check" on test_data {
                assert average(price) < 100
                    tolerance 5
                    name "Price less than"
            }
        }
        """
        suite = parse(dql)
        check = suite.checks[0]
        statement = check.assertions[0]
        assert isinstance(statement, Assertion)
        assert statement.tolerance == 5
        assert statement.condition == "<"

    def test_tolerance_parses_on_less_or_equal(self) -> None:
        """Test tolerance modifier parses on <= operator."""
        dql = """
        suite "Test Suite" {
            check "LEQ Check" on test_data {
                assert average(price) <= 100
                    tolerance 5
                    name "Price less or equal"
            }
        }
        """
        suite = parse(dql)
        check = suite.checks[0]
        statement = check.assertions[0]
        assert isinstance(statement, Assertion)
        assert statement.tolerance == 5
        assert statement.condition == "<="

    def test_basic_operators_without_tolerance(self) -> None:
        """Verify operators without tolerance have None tolerance."""
        dql = """
        suite "Test Suite" {
            check "No Tolerance Check" on test_data {
                assert average(price) > 100 name "GT"
                assert average(price) >= 100 name "GEQ"
                assert average(price) < 200 name "LT"
                assert average(price) <= 200 name "LEQ"
            }
        }
        """
        suite = parse(dql)
        check = suite.checks[0]
        for statement in check.assertions:
            assert isinstance(statement, Assertion), f"Statement should be Assertion, got {type(statement)}"
            assert statement.tolerance is None, f"Assertion {statement.name} should have no tolerance"


class TestToleranceOnNotEqual:
    """Test tolerance modifier on != operator."""

    def test_tolerance_parses_on_not_equal(self) -> None:
        """Test tolerance modifier parses on != operator."""
        dql = """
        suite "Test Suite" {
            check "NEQ Check" on test_data {
                assert average(price) != 0
                    tolerance 1
                    name "Price not equal to zero"
            }
        }
        """
        suite = parse(dql)
        check = suite.checks[0]
        statement = check.assertions[0]
        assert isinstance(statement, Assertion)
        assert statement.tolerance == 1
        assert statement.condition == "!="

    def test_not_equal_without_tolerance(self) -> None:
        """Verify != operator without tolerance has None tolerance."""
        dql = """
        suite "Test Suite" {
            check "NEQ Check" on test_data {
                assert average(price) != 0 name "NEQ"
            }
        }
        """
        suite = parse(dql)
        check = suite.checks[0]
        statement = check.assertions[0]
        assert isinstance(statement, Assertion)
        assert statement.tolerance is None
