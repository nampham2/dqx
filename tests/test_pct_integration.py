"""Integration tests for pct() with assertions and tunables."""

from __future__ import annotations


from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.functions import pct
from dqx.orm.repositories import InMemoryMetricDB
from dqx.tunables import TunableFloat


class TestPctWithAssertions:
    """Test pct() integration with assertion methods."""

    def test_pct_with_is_leq(self) -> None:
        """Test pct() with is_leq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_leq")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="leq").is_leq(pct(5))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_lt(self) -> None:
        """Test pct() with is_lt assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_lt")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="lt").is_lt(pct(5))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_geq(self) -> None:
        """Test pct() with is_geq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_geq")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="geq").is_geq(pct(1))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_gt(self) -> None:
        """Test pct() with is_gt assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_gt")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="gt").is_gt(pct(1))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_eq(self) -> None:
        """Test pct() with is_eq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_eq")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="eq").is_eq(pct(0))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_neq(self) -> None:
        """Test pct() with is_neq assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_neq")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="neq").is_neq(pct(100))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_is_between(self) -> None:
        """Test pct() with is_between assertion."""
        db = InMemoryMetricDB()

        @check(name="Test is_between")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="between").is_between(pct(0), pct(10))

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_with_all_assertions_in_check(self) -> None:
        """Test pct() with all assertion methods in one check."""
        db = InMemoryMetricDB()

        @check(name="Test All Assertions")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ctx.assert_that(metric).config(name="leq").is_leq(pct(5))
            ctx.assert_that(metric).config(name="lt").is_lt(pct(5))
            ctx.assert_that(metric).config(name="geq").is_geq(pct(1))
            ctx.assert_that(metric).config(name="gt").is_gt(pct(1))
            ctx.assert_that(metric).config(name="eq").is_eq(pct(0))
            ctx.assert_that(metric).config(name="neq").is_neq(pct(100))
            ctx.assert_that(metric).config(name="between").is_between(pct(0), pct(10))

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
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).config(name="Null rate").is_leq(threshold)

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1

    def test_pct_mixed_with_tunable_in_is_between(self) -> None:
        """Test pct() mixed with tunable in is_between."""
        db = InMemoryMetricDB()

        lower = TunableFloat("LOWER", value=pct(0), bounds=(pct(0), pct(50)))
        upper = TunableFloat("UPPER", value=pct(10), bounds=(pct(0), pct(50)))

        @check(name="Test Mixed")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            ctx.assert_that(mp.num_rows()).config(name="Range").is_between(lower, upper)

        suite = VerificationSuite([test_check], db, "Test")
        assert len(list(suite.graph.checks())) == 1


class TestPctNotInNamespace:
    """Test that pct() is NOT in the metric namespace."""

    def test_pct_not_in_metric_namespace(self) -> None:
        """Test pct() is NOT available in metric expressions."""
        db = InMemoryMetricDB()

        @check(name="Test Namespace")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            # This should work - pct() used as threshold (immediate evaluation)
            ctx.assert_that(mp.num_rows()).config(name="Test").is_leq(pct(5))

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
