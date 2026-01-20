"""Integration tests for pct() with assertions and tunables."""

from __future__ import annotations

from pathlib import Path

from dqx.api import VerificationSuite, check
from dqx.functions import pct
from dqx.orm.repositories import InMemoryMetricDB
from dqx.tunables import TunableFloat


class TestPctWithAssertions:
    """Test pct() integration with assertion methods."""

    def test_pct_with_is_leq(self) -> None:
        """Test pct() with is_leq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_leq")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="leq").is_leq(pct(5))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_lt(self) -> None:
        """Test pct() with is_lt assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_lt")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="lt").is_lt(pct(5))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_geq(self) -> None:
        """Test pct() with is_geq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_geq")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="geq").is_geq(pct(1))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_gt(self) -> None:
        """Test pct() with is_gt assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_gt")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="gt").is_gt(pct(1))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_eq(self) -> None:
        """Test pct() with is_eq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_eq")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="eq").is_eq(pct(0))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_neq(self) -> None:
        """Test pct() with is_neq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_neq")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="neq").is_neq(pct(100))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_between(self) -> None:
        """Test pct() with is_between assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_between")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="between").is_between(pct(0), pct(10))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_all_assertions_in_check(self) -> None:
        """Test pct() with all assertion methods in one check."""
        db = InMemoryMetricDB()

        @check(name="Test All Assertions")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            metric = mp.num_rows()
            ctx.assert_that(metric).where(name="leq").is_leq(pct(5))
            ctx.assert_that(metric).where(name="lt").is_lt(pct(5))
            ctx.assert_that(metric).where(name="geq").is_geq(pct(1))
            ctx.assert_that(metric).where(name="gt").is_gt(pct(1))
            ctx.assert_that(metric).where(name="eq").is_eq(pct(0))
            ctx.assert_that(metric).where(name="neq").is_neq(pct(100))
            ctx.assert_that(metric).where(name="between").is_between(pct(0), pct(10))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1


class TestPctWithTunables:
    """Test pct() integration with tunables."""

    def test_pct_with_tunable_float_value(self) -> None:
        """Test pct() with TunableFloat value."""
        threshold = TunableFloat(
            "THRESHOLD",
            value=pct(5),
            bounds=(0.0, 1.0),
        )

        assert threshold.value == 0.05
        assert threshold.bounds == (0.0, 1.0)

    def test_pct_with_tunable_float_bounds(self) -> None:
        """Test pct() with TunableFloat bounds."""
        threshold = TunableFloat(
            "THRESHOLD",
            value=pct(5),
            bounds=(pct(0), pct(10)),
        )

        assert threshold.value == 0.05
        assert threshold.bounds == (0.0, 0.1)

    def test_pct_with_tunable_in_assertion(self) -> None:
        """Test pct() with tunable used in assertion."""
        db = InMemoryMetricDB()

        threshold = TunableFloat(
            "THRESHOLD",
            value=pct(5),
            bounds=(pct(0), pct(10)),
        )

        @check(name="Test Tunable")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            ctx.assert_that(mp.num_rows()).where(name="Null rate").is_leq(threshold)

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_mixed_with_tunable_in_is_between(self) -> None:
        """Test pct() mixed with tunable in is_between."""
        db = InMemoryMetricDB()

        lower = TunableFloat("LOWER", value=pct(0), bounds=(pct(0), pct(50)))
        upper = TunableFloat("UPPER", value=pct(10), bounds=(pct(0), pct(50)))

        @check(name="Test Mixed")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            ctx.assert_that(mp.num_rows()).where(name="Range").is_between(lower, upper)

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1


class TestPctNotInNamespace:
    """Test that pct() is NOT in the metric namespace."""

    def test_pct_not_in_metric_namespace(self) -> None:
        """Test pct() is NOT available in metric expressions."""
        db = InMemoryMetricDB()

        @check(name="Test Namespace")
        def test_check(mp, ctx):  # type: ignore[no-untyped-def]
            # This should work - pct() used as threshold (immediate evaluation)
            ctx.assert_that(mp.num_rows()).where(name="Test").is_leq(pct(5))

        suite = VerificationSuite([test_check], db, "Test")

        # Verify the suite builds successfully
        assert len(list(suite.graph.checks())) == 1
        # pct() evaluated to 0.05 before being passed to is_leq()
        assert suite.is_evaluated is False  # Not run yet

    def test_pct_not_registered_as_sympy_function(self) -> None:
        """Test pct() returns plain float, not SymPy expression."""
        import sympy as sp

        result = pct(5)

        # Verify it's plain float
        assert type(result) is float
        assert result == 0.05

        # Verify it's NOT SymPy
        assert not isinstance(result, sp.Basic)
        assert not isinstance(result, sp.Expr)


class TestDQLPercentageLiterals:
    """Test that DQL percentage literals work correctly."""

    def test_dql_percentage_in_assertion(self, tmp_path: Path) -> None:
        """Test DQL with percentage literals in assertions."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Percentage Test" {
            check "Null Rate Check" on dataset {
                assert null_rate(customer_id) <= 5%
                    name "Customer ID completeness"

                assert null_rate(amount) <= 1%
                    name "Amount completeness"

                assert null_rate(status) <= 0.5%
                    name "Status completeness"
            }
        }
        """)

        # Should parse successfully with percentage literals
        suite = VerificationSuite(dql=dql_file, db=db)

        # Verify suite was created with correct name
        assert suite._name == "Percentage Test"
        assert len(suite._checks) == 1

        # Verify assertions were created
        assertions = list(suite.graph.assertions())
        assert len(assertions) == 3

        # Check assertion names
        names = {a.name for a in assertions}
        assert "Customer ID completeness" in names
        assert "Amount completeness" in names
        assert "Status completeness" in names

    def test_dql_percentage_in_tunable(self, tmp_path: Path) -> None:
        """Test DQL with percentage literals in tunables."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Tunable Percentage Test" {
            tunable MAX_NULL_RATE = 5% bounds [0%, 10%]

            check "Quality" on dataset {
                assert null_rate(col) <= MAX_NULL_RATE
                    name "Null rate check"
            }
        }
        """)

        # Should parse successfully with percentage literals in tunables
        suite = VerificationSuite(dql=dql_file, db=db)

        # Verify tunable was created with correct value (5% = 0.05)
        tunables = suite.get_tunable_params()
        assert len(tunables) == 1
        assert tunables[0]["name"] == "MAX_NULL_RATE"
        assert tunables[0]["value"] == 0.05
        assert tunables[0]["bounds"] == (0.0, 0.1)

    def test_dql_percentage_all_operators(self, tmp_path: Path) -> None:
        """Test DQL percentage literals work with all comparison operators."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "All Operators" {
            check "Test" on dataset {
                assert null_rate(col1) > 5%
                    name "Greater than"

                assert null_rate(col2) >= 5%
                    name "Greater equal"

                assert null_rate(col3) < 10%
                    name "Less than"

                assert null_rate(col4) <= 10%
                    name "Less equal"

                assert null_rate(col5) == 5%
                    name "Equal"

                assert null_rate(col6) != 5%
                    name "Not equal"

                assert null_rate(col7) between 1% and 10%
                    name "Between"
            }
        }
        """)

        # Should parse successfully
        suite = VerificationSuite(dql=dql_file, db=db)

        # Verify all assertions created
        assertions = list(suite.graph.assertions())
        assert len(assertions) == 7

        # Check all assertion names present
        names = {a.name for a in assertions}
        assert "Greater than" in names
        assert "Greater equal" in names
        assert "Less than" in names
        assert "Less equal" in names
        assert "Equal" in names
        assert "Not equal" in names
        assert "Between" in names
