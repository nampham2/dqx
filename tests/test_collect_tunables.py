"""Tests for collect_tunables_from_graph function."""

from __future__ import annotations

from dqx.api import Context, MetricProvider, VerificationSuite, check, collect_tunables_from_graph
from dqx.orm.repositories import InMemoryMetricDB
from dqx.tunables import TunableInt, TunableFloat


class TestCollectTunablesFromGraph:
    """Tests for collect_tunables_from_graph function."""

    def test_collect_single_tunable(self) -> None:
        """Can collect a single tunable from the graph."""
        db = InMemoryMetricDB()
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            # Tunable must be in the expression, not passed to is_gt
            expr = x - threshold
            ctx.assert_that(expr).config(name="Test").is_gt(0)

        suite = VerificationSuite([test_check], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        assert len(tunables) == 1
        assert "THRESHOLD" in tunables
        assert tunables["THRESHOLD"] is threshold

    def test_collect_multiple_tunables(self) -> None:
        """Can collect multiple tunables from the graph."""
        db = InMemoryMetricDB()
        threshold1 = TunableFloat("THRESHOLD1", value=0.05, bounds=(0.0, 0.20))
        threshold2 = TunableInt("THRESHOLD2", value=100, bounds=(0, 1000))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold1).config(name="Test1").is_gt(0)
            ctx.assert_that(x + threshold2).config(name="Test2").is_lt(1000)

        suite = VerificationSuite([test_check], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        assert len(tunables) == 2
        assert "THRESHOLD1" in tunables
        assert "THRESHOLD2" in tunables
        assert tunables["THRESHOLD1"] is threshold1
        assert tunables["THRESHOLD2"] is threshold2

    def test_collect_tunable_from_multiple_assertions(self) -> None:
        """Same tunable used in multiple assertions is collected once."""
        db = InMemoryMetricDB()
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            y = mp.null_count("col")
            # Use same tunable in two different assertions
            ctx.assert_that(x - threshold).config(name="Test1").is_gt(0)
            ctx.assert_that(y + threshold).config(name="Test2").is_lt(1)

        suite = VerificationSuite([test_check], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        # Should only have one instance
        assert len(tunables) == 1
        assert "THRESHOLD" in tunables
        assert tunables["THRESHOLD"] is threshold

    def test_collect_tunable_from_expression(self) -> None:
        """Can collect tunable used in complex expression."""
        db = InMemoryMetricDB()
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            # Use tunable in an expression
            expr = x + threshold * 2
            ctx.assert_that(expr).config(name="Test").is_gt(0)

        suite = VerificationSuite([test_check], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        assert len(tunables) == 1
        assert "THRESHOLD" in tunables
        assert tunables["THRESHOLD"] is threshold

    def test_collect_no_tunables(self) -> None:
        """Returns empty dict when no tunables are used."""
        db = InMemoryMetricDB()

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            # No tunable, just constant
            ctx.assert_that(x).config(name="Test").is_gt(100)

        suite = VerificationSuite([test_check], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        assert len(tunables) == 0
        assert tunables == {}

    def test_collect_tunables_different_types(self) -> None:
        """Can collect tunables of different types."""
        db = InMemoryMetricDB()
        percent = TunableFloat("PCT", value=0.05, bounds=(0.0, 1.0))
        int_val = TunableInt("INT", value=100, bounds=(0, 1000))
        float_val = TunableFloat("FLT", value=0.5, bounds=(0.0, 1.0))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - percent).config(name="Test1").is_gt(0)
            ctx.assert_that(x + int_val).config(name="Test2").is_lt(1000)
            ctx.assert_that(x - float_val).config(name="Test3").is_gt(0)

        suite = VerificationSuite([test_check], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        assert len(tunables) == 3
        assert tunables["PCT"] is percent
        assert tunables["INT"] is int_val
        assert tunables["FLT"] is float_val

    def test_duplicate_tunable_name_uses_last_instance(self) -> None:
        """When different tunable instances share the same name, SymPy uses the last one.

        Due to SymPy's symbol caching, creating TunableSymbol instances with the same
        name will result in the same Symbol object, and the last Tunable reference wins.

        This is acceptable behavior since users should not create multiple tunables
        with the same name in the first place.
        """
        db = InMemoryMetricDB()
        # Create two different tunables with same name (this shouldn't happen in practice)
        threshold1 = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        threshold2 = TunableFloat("THRESHOLD", value=0.10, bounds=(0.0, 0.20))

        @check(name="Test Check")
        def test_check(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            # Use different instances with same name
            ctx.assert_that(x - threshold1).config(name="Test1").is_gt(0)
            ctx.assert_that(x + threshold2).config(name="Test2").is_lt(1000)

        suite = VerificationSuite([test_check], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables - will get the last instance (threshold2)
        tunables = collect_tunables_from_graph(context._graph)

        assert len(tunables) == 1
        assert "THRESHOLD" in tunables
        # The last tunable instance is used due to SymPy symbol caching
        assert tunables["THRESHOLD"] is threshold2

    def test_collect_tunables_from_multiple_checks(self) -> None:
        """Can collect tunables from multiple checks."""
        db = InMemoryMetricDB()
        t1 = TunableFloat("T1", value=0.05, bounds=(0.0, 0.20))
        t2 = TunableInt("T2", value=100, bounds=(0, 1000))

        @check(name="Check 1")
        def check1(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - t1).config(name="Test1").is_gt(0)

        @check(name="Check 2")
        def check2(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x + t2).config(name="Test2").is_lt(1000)

        suite = VerificationSuite([check1, check2], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        assert len(tunables) == 2
        assert tunables["T1"] is t1
        assert tunables["T2"] is t2

    def test_collect_same_tunable_across_checks(self) -> None:
        """Same tunable used across multiple checks is collected once."""
        db = InMemoryMetricDB()
        threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        @check(name="Check 1")
        def check1(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x - threshold).config(name="Test1").is_gt(0)

        @check(name="Check 2")
        def check2(mp: MetricProvider, ctx: Context) -> None:
            x = mp.num_rows()
            ctx.assert_that(x + threshold * 2).config(name="Test2").is_lt(1000)

        suite = VerificationSuite([check1, check2], db, "Test Suite")
        context = Context("test", db, execution_id="test-exec", data_av_threshold=0.9)
        suite.build_graph(context)

        # Collect tunables
        tunables = collect_tunables_from_graph(context._graph)

        # Should only have one instance
        assert len(tunables) == 1
        assert tunables["THRESHOLD"] is threshold
