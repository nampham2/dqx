"""Tests for profiles module."""

from datetime import date

import pytest
import sympy as sp
from returns.result import Failure, Result, Success

from dqx.common import ExecutionId, ResultKey, SymbolicValidator
from dqx.evaluator import Evaluator
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.orm.repositories import InMemoryMetricDB
from dqx.profiles import (
    AssertionSelector,
    HolidayProfile,
    Profile,
    Rule,
    TagSelector,
    assertion,
    check,
    resolve_overrides,
    tag,
)
from dqx.provider import MetricProvider, SymbolicMetric


class TestAssertionSelector:
    """Tests for AssertionSelector."""

    def test_matches_check_and_assertion(self) -> None:
        selector = AssertionSelector(check="Volume Check", assertion="Daily orders")
        assert selector.matches("Volume Check", "Daily orders")
        assert not selector.matches("Volume Check", "Other assertion")
        assert not selector.matches("Other Check", "Daily orders")

    def test_matches_check_only(self) -> None:
        selector = AssertionSelector(check="Volume Check", assertion=None)
        assert selector.matches("Volume Check", "Daily orders")
        assert selector.matches("Volume Check", "Any assertion")
        assert not selector.matches("Other Check", "Daily orders")


class TestTagSelector:
    """Tests for TagSelector."""

    def test_matches_tag(self) -> None:
        selector = TagSelector(tag="xmas")
        assert selector.matches(frozenset({"xmas", "volume"}))
        assert selector.matches(frozenset({"xmas"}))
        assert not selector.matches(frozenset({"volume"}))
        assert not selector.matches(frozenset())


class TestRule:
    """Tests for Rule dataclass."""

    def test_defaults(self) -> None:
        rule = Rule(selector=TagSelector(tag="test"))
        assert not rule.disabled
        assert rule.metric_multiplier == 1.0

    def test_disabled(self) -> None:
        rule = Rule(selector=TagSelector(tag="test"), disabled=True)
        assert rule.disabled

    def test_metric_multiplier(self) -> None:
        rule = Rule(selector=TagSelector(tag="test"), metric_multiplier=2.0)
        assert rule.metric_multiplier == 2.0


class TestHolidayProfile:
    """Tests for HolidayProfile."""

    def test_is_active_within_range(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[],
        )
        assert profile.is_active(date(2024, 12, 20))  # Start date
        assert profile.is_active(date(2024, 12, 25))  # Middle
        assert profile.is_active(date(2025, 1, 5))  # End date

    def test_is_active_outside_range(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[],
        )
        assert not profile.is_active(date(2024, 12, 19))  # Before
        assert not profile.is_active(date(2025, 1, 6))  # After

    def test_rules_property(self) -> None:
        rules = [
            Rule(selector=TagSelector(tag="xmas"), metric_multiplier=2.0),
            Rule(selector=AssertionSelector(check="Check"), disabled=True),
        ]
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=rules,
        )
        assert profile.rules == rules


class TestRuleBuilder:
    """Tests for RuleBuilder and builder functions."""

    def test_tag_builder_disable(self) -> None:
        rule = tag("xmas").disable()
        assert isinstance(rule.selector, TagSelector)
        assert rule.selector.tag == "xmas"
        assert rule.disabled
        assert rule.metric_multiplier == 1.0

    def test_tag_builder_set(self) -> None:
        rule = tag("xmas").set(metric_multiplier=2.0)
        assert isinstance(rule.selector, TagSelector)
        assert rule.selector.tag == "xmas"
        assert not rule.disabled
        assert rule.metric_multiplier == 2.0

    def test_check_builder(self) -> None:
        rule = check("Volume Check").disable()
        assert isinstance(rule.selector, AssertionSelector)
        assert rule.selector.check == "Volume Check"
        assert rule.selector.assertion is None
        assert rule.disabled

    def test_assertion_builder_with_name(self) -> None:
        rule = assertion("Volume Check", "Daily orders").set(metric_multiplier=1.5)
        assert isinstance(rule.selector, AssertionSelector)
        assert rule.selector.check == "Volume Check"
        assert rule.selector.assertion == "Daily orders"
        assert rule.metric_multiplier == 1.5

    def test_assertion_builder_without_name(self) -> None:
        rule = assertion("Volume Check").set(metric_multiplier=1.5)
        assert isinstance(rule.selector, AssertionSelector)
        assert rule.selector.check == "Volume Check"
        assert rule.selector.assertion is None


class MockAssertionNode:
    """Mock AssertionNode for testing resolve_overrides."""

    def __init__(self, name: str, tags: set[str] | None = None) -> None:
        self.name = name
        self.tags = tags or set()


class TestResolveOverrides:
    """Tests for resolve_overrides function."""

    def test_no_profiles(self) -> None:
        node = MockAssertionNode("Test Assertion", {"xmas"})
        result = resolve_overrides(
            check_name="Test Check",
            assertion=node,  # type: ignore
            profiles=[],
            target_date=date(2024, 12, 25),
        )
        assert not result.disabled
        assert result.metric_multiplier == 1.0

    def test_inactive_profile(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )
        node = MockAssertionNode("Test Assertion", {"xmas"})
        result = resolve_overrides(
            check_name="Test Check",
            assertion=node,  # type: ignore
            profiles=[profile],
            target_date=date(2024, 11, 1),  # Outside profile range
        )
        assert not result.disabled
        assert result.metric_multiplier == 1.0

    def test_active_profile_with_matching_tag(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )
        node = MockAssertionNode("Test Assertion", {"xmas"})
        result = resolve_overrides(
            check_name="Test Check",
            assertion=node,  # type: ignore
            profiles=[profile],
            target_date=date(2024, 12, 25),
        )
        assert not result.disabled
        assert result.metric_multiplier == 2.0

    def test_active_profile_with_non_matching_tag(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )
        node = MockAssertionNode("Test Assertion", {"volume"})  # No xmas tag
        result = resolve_overrides(
            check_name="Test Check",
            assertion=node,  # type: ignore
            profiles=[profile],
            target_date=date(2024, 12, 25),
        )
        assert not result.disabled
        assert result.metric_multiplier == 1.0

    def test_disabled_by_check_selector(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[check("Volume Check").disable()],
        )
        node = MockAssertionNode("Daily orders")
        result = resolve_overrides(
            check_name="Volume Check",
            assertion=node,  # type: ignore
            profiles=[profile],
            target_date=date(2024, 12, 25),
        )
        assert result.disabled

    def test_disabled_by_assertion_selector(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[assertion("Volume Check", "Daily orders").disable()],
        )
        node = MockAssertionNode("Daily orders")
        result = resolve_overrides(
            check_name="Volume Check",
            assertion=node,  # type: ignore
            profiles=[profile],
            target_date=date(2024, 12, 25),
        )
        assert result.disabled

    def test_multipliers_compound(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[
                tag("volume").set(metric_multiplier=1.5),
                tag("xmas").set(metric_multiplier=2.0),
            ],
        )
        node = MockAssertionNode("Test Assertion", {"volume", "xmas"})
        result = resolve_overrides(
            check_name="Test Check",
            assertion=node,  # type: ignore
            profiles=[profile],
            target_date=date(2024, 12, 25),
        )
        assert result.metric_multiplier == 3.0  # 1.5 * 2.0

    def test_multiple_profiles_compound(self) -> None:
        profile1 = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )
        profile2 = HolidayProfile(
            name="Year End",
            start_date=date(2024, 12, 25),
            end_date=date(2024, 12, 31),
            rules=[tag("xmas").set(metric_multiplier=1.5)],
        )
        node = MockAssertionNode("Test Assertion", {"xmas"})
        result = resolve_overrides(
            check_name="Test Check",
            assertion=node,  # type: ignore
            profiles=[profile1, profile2],
            target_date=date(2024, 12, 25),
        )
        assert result.metric_multiplier == 3.0  # 2.0 * 1.5

    def test_disabled_takes_precedence(self) -> None:
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[
                tag("xmas").set(metric_multiplier=2.0),
                check("Volume Check").disable(),
            ],
        )
        node = MockAssertionNode("Daily orders", {"xmas"})
        result = resolve_overrides(
            check_name="Volume Check",
            assertion=node,  # type: ignore
            profiles=[profile],
            target_date=date(2024, 12, 25),
        )
        assert result.disabled
        assert result.metric_multiplier == 2.0  # Still computed


class TestEvaluatorWithProfiles:
    """Integration tests for Evaluator with profiles."""

    def _create_evaluator_with_metric(
        self,
        symbol: sp.Symbol,
        metric_name: str,
        metric_value: float,
        target_date: date,
        profiles: list[Profile] | None = None,
    ) -> tuple[Evaluator, MetricProvider]:
        """Helper to create evaluator with a single metric."""
        db = InMemoryMetricDB()
        execution_id = ExecutionId("test-exec-123")
        provider = MetricProvider(db, execution_id, data_av_threshold=0.9)

        def metric_fn(k: ResultKey, v: float = metric_value) -> Result[float, str]:
            return Success(v)

        metric = SymbolicMetric(
            name=metric_name,
            symbol=symbol,
            fn=metric_fn,
            metric_spec=None,  # type: ignore
            dataset="test_data",
            data_av_ratio=0.95,
        )
        provider.registry._metrics.append(metric)
        provider.registry.index[symbol] = metric

        key = ResultKey(yyyy_mm_dd=target_date, tags={})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.8, profiles=profiles or [])
        evaluator._metrics = {symbol: Success(metric_value)}

        return evaluator, provider

    def test_evaluator_without_profiles(self) -> None:
        """Test evaluator works normally without profiles."""
        x = sp.Symbol("x")
        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=60.0,
            target_date=date(2024, 12, 25),
            profiles=[],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 50", lambda v: v > 50)
        assertion_node = AssertionNode(check_node, actual=x, name="Orders above 50", validator=validator, severity="P1")

        evaluator.visit(assertion_node)

        assert assertion_node._result == "PASSED"
        assert assertion_node._metric == Success(60.0)

    def test_evaluator_with_metric_multiplier(self) -> None:
        """Test evaluator applies metric_multiplier from profile."""
        x = sp.Symbol("x")
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )

        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=60.0,  # Without multiplier: 60 > 100 = FAILED
            target_date=date(2024, 12, 25),
            profiles=[profile],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 100", lambda v: v > 100)
        assertion_node = AssertionNode(
            check_node,
            actual=x,
            name="Orders above 100",
            validator=validator,
            severity="P1",
            tags=frozenset({"xmas"}),  # Tag matches profile rule
        )

        evaluator.visit(assertion_node)

        # 60 * 2.0 = 120 > 100 = PASSED
        assert assertion_node._result == "PASSED"

    def test_evaluator_with_disabled_assertion(self) -> None:
        """Test evaluator skips disabled assertions."""
        x = sp.Symbol("x")
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[check("Volume Check").disable()],
        )

        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=60.0,
            target_date=date(2024, 12, 25),
            profiles=[profile],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 100", lambda v: v > 100)
        assertion_node = AssertionNode(
            check_node, actual=x, name="Orders above 100", validator=validator, severity="P1"
        )

        evaluator.visit(assertion_node)

        assert assertion_node._result == "SKIPPED"
        assert isinstance(assertion_node._metric, Failure)
        failures = assertion_node._metric.failure()
        assert failures[0].error_message == "Disabled by profile"

    def test_evaluator_inactive_profile_no_effect(self) -> None:
        """Test inactive profile has no effect on evaluation."""
        x = sp.Symbol("x")
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )

        # Date is outside profile range
        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=60.0,
            target_date=date(2024, 11, 1),  # Before Christmas profile
            profiles=[profile],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 100", lambda v: v > 100)
        assertion_node = AssertionNode(
            check_node,
            actual=x,
            name="Orders above 100",
            validator=validator,
            severity="P1",
            tags=frozenset({"xmas"}),
        )

        evaluator.visit(assertion_node)

        # Without multiplier: 60 > 100 = FAILED
        assert assertion_node._result == "FAILED"

    def test_evaluator_non_matching_tag_no_effect(self) -> None:
        """Test profile rule with non-matching tag has no effect."""
        x = sp.Symbol("x")
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )

        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=60.0,
            target_date=date(2024, 12, 25),
            profiles=[profile],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 100", lambda v: v > 100)
        assertion_node = AssertionNode(
            check_node,
            actual=x,
            name="Orders above 100",
            validator=validator,
            severity="P1",
            tags=frozenset({"volume"}),  # Different tag, no match
        )

        evaluator.visit(assertion_node)

        # No multiplier applied: 60 > 100 = FAILED
        assert assertion_node._result == "FAILED"

    def test_evaluator_multiple_profiles_compound(self) -> None:
        """Test multiple active profiles compound their multipliers."""
        x = sp.Symbol("x")
        profile1 = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[tag("xmas").set(metric_multiplier=2.0)],
        )
        profile2 = HolidayProfile(
            name="Year End",
            start_date=date(2024, 12, 25),
            end_date=date(2024, 12, 31),
            rules=[tag("xmas").set(metric_multiplier=1.5)],
        )

        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=40.0,  # 40 * 2.0 * 1.5 = 120 > 100
            target_date=date(2024, 12, 25),
            profiles=[profile1, profile2],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 100", lambda v: v > 100)
        assertion_node = AssertionNode(
            check_node,
            actual=x,
            name="Orders above 100",
            validator=validator,
            severity="P1",
            tags=frozenset({"xmas"}),
        )

        evaluator.visit(assertion_node)

        # 40 * 2.0 * 1.5 = 120 > 100 = PASSED
        assert assertion_node._result == "PASSED"

    def test_evaluator_disabled_by_assertion_selector(self) -> None:
        """Test disabling specific assertion by name."""
        x = sp.Symbol("x")
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[assertion("Volume Check", "Daily orders").disable()],
        )

        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=60.0,
            target_date=date(2024, 12, 25),
            profiles=[profile],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 50", lambda v: v > 50)

        # This assertion matches the rule
        disabled_assertion = AssertionNode(
            check_node, actual=x, name="Daily orders", validator=validator, severity="P1"
        )
        # This assertion does NOT match the rule
        enabled_assertion = AssertionNode(
            check_node, actual=x, name="Weekly orders", validator=validator, severity="P1"
        )

        evaluator.visit(disabled_assertion)
        evaluator.visit(enabled_assertion)

        assert disabled_assertion._result == "SKIPPED"
        assert enabled_assertion._result == "PASSED"

    @pytest.mark.asyncio
    async def test_evaluator_visit_async_with_profiles(self) -> None:
        """Test async visit also applies profile overrides."""
        x = sp.Symbol("x")
        profile = HolidayProfile(
            name="Christmas",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[check("Volume Check").disable()],
        )

        evaluator, _ = self._create_evaluator_with_metric(
            symbol=x,
            metric_name="count(orders)",
            metric_value=60.0,
            target_date=date(2024, 12, 25),
            profiles=[profile],
        )

        root = RootNode("test")
        check_node = CheckNode(root, "Volume Check")
        validator = SymbolicValidator("> 50", lambda v: v > 50)
        assertion_node = AssertionNode(check_node, actual=x, name="Orders", validator=validator, severity="P1")

        await evaluator.visit_async(assertion_node)

        assert assertion_node._result == "SKIPPED"
