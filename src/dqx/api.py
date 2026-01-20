from __future__ import annotations

import functools
import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast, overload, runtime_checkable

import pyarrow as pa
import sympy as sp

from dqx import functions, setup_logger
from dqx.analyzer import AnalysisReport, Analyzer
from dqx.common import (
    AssertionResult,
    DQXError,
    ResultKey,
    SeverityLevel,
    SqlDataSource,
    SymbolicValidator,
    validate_tags,
)
from dqx.evaluator import Evaluator
from dqx.graph.nodes import CheckNode, RootNode
from dqx.graph.traversal import Graph

# import moved to local scope(s) to avoid cyclic dependency
from dqx.plugins import PluginExecutionContext, PluginManager
from dqx.profiles import Profile
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.timer import Registry
from dqx.tunables import Tunable, TunableChange
from dqx.validator import SuiteValidator

if TYPE_CHECKING:
    from dqx.orm.repositories import MetricDB, MetricStats

CheckProducer = Callable[[MetricProvider, "Context"], None]
CheckCreator = Callable[[CheckProducer], CheckProducer]

# Type alias for numeric tunables (TunableFloat, TunableInt)
NumericTunable: TypeAlias = "Tunable[float] | Tunable[int]"


logger = logging.getLogger(__name__)

# Sentinel value to detect if name parameter was explicitly provided
_NAME_NOT_PROVIDED = object()
timer_registry = Registry()


def collect_tunables_from_graph(graph: Graph) -> dict[str, Tunable[Any]]:
    """
    Extract all Tunable objects referenced in assertion expressions.

    Uses graph traversal with TunableCollectorVisitor to scan the graph
    and collect TunableSymbol instances from assertion expressions,
    returning a mapping of tunable names to their Tunable objects.

    Note: If multiple Tunable instances with the same name are used,
    only the last instance encountered during traversal will be captured.
    Users should avoid creating multiple tunables with the same name.

    Args:
        graph: The verification graph to scan for tunables

    Returns:
        dict[str, Tunable[Any]]: Mapping of tunable names to Tunable objects

    Example:
        >>> # After building the graph
        >>> tunables = collect_tunables_from_graph(suite.graph)
        >>> print(tunables.keys())  # {'THRESHOLD', 'MIN_ROWS', ...}
    """
    from dqx.graph.visitors import TunableCollectorVisitor

    visitor = TunableCollectorVisitor()
    graph.dfs(visitor)
    return visitor.tunables


class AssertionDraft:
    """
    Initial assertion builder that requires a name before making assertions.

    This is the first stage of assertion building. You must call where()
    with a name before you can make any assertions.

    Example:
        draft = ctx.assert_that(mp.average("price"))
        ready = draft.where(name="Price is positive")
        ready.is_positive()
    """

    def __init__(self, actual: sp.Expr, context: Context | None = None) -> None:
        """
        Create an AssertionDraft that holds the symbolic expression to be asserted and an optional execution Context.

        Args:
            actual: The symbolic expression representing the value or predicate to evaluate.
            context: The execution Context used to register the assertion when finalized; may be None for deferred registration.
        """
        self._actual = actual
        self._context = context

    def where(
        self,
        *,
        name: str,
        severity: SeverityLevel = "P1",
        tags: frozenset[str] | set[str] | None = None,
        experimental: bool = False,
        required: bool = False,
        cost: dict[str, float] | None = None,
    ) -> AssertionReady:
        """
        Create an AssertionReady bound to this expression with the given name and metadata.

        Args:
            name: Descriptive name for the assertion (1–255 characters).
            severity: Severity level for the assertion (e.g., "P0", "P1", "P2", "P3").
            tags: Optional set of tags; tags must contain only alphanumerics, dashes, and underscores.
            experimental: If True, marks the assertion as proposed/experimental and removable by algorithms.
            required: If True, marks the assertion as required and not removable by algorithms.
            cost: Optional cost dictionary for RL with exactly the keys "fp" and "fn"; values must be numeric and >= 0.

        Returns:
            AssertionReady: A ready-to-use assertion object with assertion methods available.

        Raises:
            ValueError: If name is empty or longer than 255 characters, if tags are invalid, or if cost is not a dict with numeric, non-negative "fp" and "fn" values.
        """
        if not name or not name.strip():
            raise ValueError("Assertion name cannot be empty")
        if len(name) > 255:
            raise ValueError("Assertion name is too long (max 255 characters)")

        validated_tags = validate_tags(tags)

        # Validate cost if provided
        cost_fp = None
        cost_fn = None
        if cost is not None:
            if not isinstance(cost, dict):
                raise ValueError("cost must be a dict with 'fp' and 'fn' keys")
            if set(cost.keys()) != {"fp", "fn"}:
                raise ValueError("cost must have exactly 'fp' and 'fn' keys")
            if (
                not isinstance(cost["fp"], (int, float))
                or isinstance(cost["fp"], bool)
                or not isinstance(cost["fn"], (int, float))
                or isinstance(cost["fn"], bool)
            ):
                raise ValueError("cost values must be numeric (int or float)")
            if cost["fp"] < 0 or cost["fn"] < 0:
                raise ValueError("cost values must be non-negative")
            cost_fp = cost["fp"]
            cost_fn = cost["fn"]

        return AssertionReady(
            actual=self._actual,
            name=name.strip(),
            severity=severity,
            tags=validated_tags,
            experimental=experimental,
            required=required,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            context=self._context,
        )


class AssertionReady:
    """
    Named assertion ready to perform validations.

    This assertion has been properly named and can now use any of the
    validation methods like is_gt(), is_eq(), etc.
    """

    def __init__(
        self,
        actual: sp.Expr,
        name: str,
        severity: SeverityLevel = "P1",
        tags: frozenset[str] | None = None,
        experimental: bool = False,
        required: bool = False,
        cost_fp: float | None = None,
        cost_fn: float | None = None,
        context: Context | None = None,
    ) -> None:
        """
        Create an assertion ready to be registered with a named check, carrying its expression, metadata, and optional execution context.

        Args:
            actual: Symbolic expression representing the assertion target.
            name: Human-readable identifier for the assertion (max 255 chars).
            severity: Severity label, one of "P0", "P1", "P2", "P3".
            tags: Optional tags used for profile-based selection.
            experimental: If True, marks the assertion as algorithm-proposed.
            required: If True, prevents automated removal of the assertion.
            cost_fp: Cost assigned to a false positive for reward calculations.
            cost_fn: Cost assigned to a false negative for reward calculations.
            context: Execution context that will own the assertion.
        """
        self._actual = actual
        self._name = name
        self._severity = severity
        self._tags = tags
        self._experimental = experimental
        self._required = required
        self._cost_fp = cost_fp
        self._cost_fn = cost_fn
        self._context = context

    @overload
    def is_geq(self, other: float, tol: float = functions.EPSILON) -> None: ...

    @overload
    def is_geq(self, other: NumericTunable, tol: float = functions.EPSILON) -> None: ...

    def is_geq(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
        """
        Create an assertion that the expression is greater than or equal to the specified threshold.

        When a tunable is provided, the comparison is evaluated lazily at evaluation time.

        Args:
            other: Threshold to compare against (float or NumericTunable).
            tol: Comparison tolerance; values within `tol` of `other` are treated as equal.

        Example:
            >>> MAX_NULL = TunableFloat("MAX_NULL", value=0.05, bounds=(0.0, 0.20))
            >>> ctx.assert_that(mp.null_count("col")).where(name="Nulls").is_geq(MAX_NULL)
        """
        from dqx.tunables import Tunable

        if isinstance(other, Tunable):
            validator = SymbolicValidator(f"≥ {other.name}", lambda x: functions.is_geq(x, other.value, tol))
            self._create_assertion_node_with_tunables(validator, {other.name: other})
        else:
            validator = SymbolicValidator(f"≥ {other}", lambda x: functions.is_geq(x, other, tol))
            self._create_assertion_node(validator)

    @overload
    def is_gt(self, other: float, tol: float = functions.EPSILON) -> None: ...

    @overload
    def is_gt(self, other: NumericTunable, tol: float = functions.EPSILON) -> None: ...

    def is_gt(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is greater than the given value or tunable."""
        from dqx.tunables import Tunable

        if isinstance(other, Tunable):
            validator = SymbolicValidator(f"> {other.name}", lambda x: functions.is_gt(x, other.value, tol))
            self._create_assertion_node_with_tunables(validator, {other.name: other})
        else:
            validator = SymbolicValidator(f"> {other}", lambda x: functions.is_gt(x, other, tol))
            self._create_assertion_node(validator)

    @overload
    def is_leq(self, other: float, tol: float = functions.EPSILON) -> None: ...

    @overload
    def is_leq(self, other: NumericTunable, tol: float = functions.EPSILON) -> None: ...

    def is_leq(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
        """
        Assert that the expression is less than or equal to the given value or tunable.

        When a tunable is provided, the comparison is evaluated lazily - the tunable's
        current value is used at evaluation time, not at graph-building time. This
        enables dynamic threshold tuning via suite.set_param() and suite.reset().

        Args:
            other: Threshold to compare against. Can be:
                   - float: Static threshold value
                   - NumericTunable: Dynamic tunable parameter (TunableFloat, TunableInt)
            tol: Comparison tolerance for floating-point comparisons.

        Example:
            >>> # Static threshold
            >>> ctx.assert_that(mp.minimum("qty")).where(name="Min qty").is_leq(10.0)
            >>>
            >>> # Dynamic tunable threshold
            >>> MIN_QTY = TunableFloat("MIN_QTY", value=10.0, bounds=(1.0, 100.0))
            >>> ctx.assert_that(mp.minimum("qty")).where(name="Min qty").is_leq(MIN_QTY)
            >>>
            >>> # Later: tune and re-run
            >>> suite.set_param("MIN_QTY", 20.0)
            >>> suite.reset()
            >>> suite.run([datasource], key)
        """
        from dqx.tunables import Tunable

        if isinstance(other, Tunable):
            # Tunable path - closure captures tunable reference for lazy evaluation
            validator = SymbolicValidator(f"≤ {other.name}", lambda x: functions.is_leq(x, other.value, tol))
            self._create_assertion_node_with_tunables(validator, {other.name: other})
        else:
            # Static float path - same as current implementation
            validator = SymbolicValidator(f"≤ {other}", lambda x: functions.is_leq(x, other, tol))
            self._create_assertion_node(validator)

    @overload
    def is_lt(self, other: float, tol: float = functions.EPSILON) -> None: ...

    @overload
    def is_lt(self, other: NumericTunable, tol: float = functions.EPSILON) -> None: ...

    def is_lt(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is less than the given value or tunable."""
        from dqx.tunables import Tunable

        if isinstance(other, Tunable):
            validator = SymbolicValidator(f"< {other.name}", lambda x: functions.is_lt(x, other.value, tol))
            self._create_assertion_node_with_tunables(validator, {other.name: other})
        else:
            validator = SymbolicValidator(f"< {other}", lambda x: functions.is_lt(x, other, tol))
            self._create_assertion_node(validator)

    @overload
    def is_eq(self, other: float, tol: float = functions.EPSILON) -> None: ...

    @overload
    def is_eq(self, other: NumericTunable, tol: float = functions.EPSILON) -> None: ...

    def is_eq(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
        """
        Assert that the expression equals the given value or tunable within tolerance.

        Args:
            other: Target value to compare against (float or NumericTunable).
            tol: Absolute tolerance for the comparison; defaults to functions.EPSILON.

        Example:
            >>> TARGET = TunableInt("TARGET", value=100, bounds=(50, 200))
            >>> ctx.assert_that(mp.count("id")).where(name="Count").is_eq(TARGET)
        """
        from dqx.tunables import Tunable

        if isinstance(other, Tunable):
            validator = SymbolicValidator(f"= {other.name}", lambda x: functions.is_eq(x, other.value, tol))
            self._create_assertion_node_with_tunables(validator, {other.name: other})
        else:
            validator = SymbolicValidator(f"= {other}", lambda x: functions.is_eq(x, other, tol))
            self._create_assertion_node(validator)

    @overload
    def is_neq(self, other: float, tol: float = functions.EPSILON) -> None: ...

    @overload
    def is_neq(self, other: NumericTunable, tol: float = functions.EPSILON) -> None: ...

    def is_neq(self, other: float | NumericTunable, tol: float = functions.EPSILON) -> None:
        """
        Assert that the expression is not equal to a specified value or tunable.

        Args:
            other: The value to compare against (float or NumericTunable).
            tol: Allowed tolerance; values within `tol` of `other` are considered equal.
        """
        from dqx.tunables import Tunable

        if isinstance(other, Tunable):
            validator = SymbolicValidator(f"≠ {other.name}", lambda x: functions.is_neq(x, other.value, tol))
            self._create_assertion_node_with_tunables(validator, {other.name: other})
        else:
            validator = SymbolicValidator(f"≠ {other}", lambda x: functions.is_neq(x, other, tol))
            self._create_assertion_node(validator)

    @overload
    def is_between(self, lower: float, upper: float, tol: float = functions.EPSILON) -> None: ...

    @overload
    def is_between(
        self,
        lower: float | NumericTunable,
        upper: float | NumericTunable,
        tol: float = functions.EPSILON,
    ) -> None: ...

    def is_between(
        self,
        lower: float | NumericTunable,
        upper: float | NumericTunable,
        tol: float = functions.EPSILON,
    ) -> None:
        """
        Assert that the expression lies within the inclusive interval [lower, upper].

        Both bounds support static floats or dynamic tunables for runtime evaluation.

        Args:
            lower: Lower bound (float or NumericTunable).
            upper: Upper bound (float or NumericTunable).
            tol: Numeric tolerance applied to the comparison.

        Raises:
            ValueError: If lower > upper when both are static floats.
                        For tunables, bound validation occurs at evaluation time.

        Example:
            >>> # Static bounds
            >>> ctx.assert_that(mp.average("score")).where(name="Range").is_between(0, 100)
            >>>
            >>> # Dynamic tunable bounds
            >>> LOWER = TunableFloat("LOWER", value=10.0, bounds=(0, 50))
            >>> UPPER = TunableFloat("UPPER", value=90.0, bounds=(50, 100))
            >>> ctx.assert_that(mp.average("score")).where(name="Range").is_between(LOWER, UPPER)
        """
        from dqx.tunables import Tunable

        # Determine if we have tunables
        lower_is_tunable = isinstance(lower, Tunable)
        upper_is_tunable = isinstance(upper, Tunable)

        if not lower_is_tunable and not upper_is_tunable:
            # Both static - validate bounds immediately
            if lower > upper:  # type: ignore[operator]
                raise ValueError(
                    f"Invalid range: lower bound ({lower}) must be less than or equal to upper bound ({upper})"
                )

            # Static validator
            validator = SymbolicValidator(
                f"in [{lower}, {upper}]",
                lambda x: functions.is_between(x, lower, upper, tol),  # type: ignore[arg-type]
            )
            self._create_assertion_node(validator)
        else:
            # At least one tunable - use closure to capture references
            # Build display name and collect tunables
            tunables_dict: dict[str, Tunable[Any]] = {}

            lower_name = lower.name if lower_is_tunable else str(lower)  # type: ignore[union-attr]
            upper_name = upper.name if upper_is_tunable else str(upper)  # type: ignore[union-attr]

            # Register tunables
            if lower_is_tunable:
                tunables_dict[lower.name] = lower  # type: ignore[union-attr, assignment]
            if upper_is_tunable:
                tunables_dict[upper.name] = upper  # type: ignore[union-attr, assignment]

            # Create validator with closure that evaluates tunables at validation time
            def between_validator(x: float) -> bool:
                """Validate that x is between lower and upper bounds, evaluating tunables at runtime."""
                # Get current values (tunable or static)
                lower_val = lower.value if lower_is_tunable else lower  # type: ignore[union-attr]
                upper_val = upper.value if upper_is_tunable else upper  # type: ignore[union-attr]

                # Runtime validation for tunable bounds
                if lower_val > upper_val:
                    lower_display = lower.name if lower_is_tunable else lower  # type: ignore[union-attr]
                    upper_display = upper.name if upper_is_tunable else upper  # type: ignore[union-attr]
                    raise ValueError(
                        f"Invalid range for {lower_display}/{upper_display}: "
                        f"lower bound ({lower_val}) > upper bound ({upper_val})"
                    )

                # Now both are floats, use standard is_between
                return functions.is_between(x, lower_val, upper_val, tol)  # type: ignore[arg-type]

            validator = SymbolicValidator(f"in [{lower_name}, {upper_name}]", between_validator)
            self._create_assertion_node_with_tunables(validator, tunables_dict)

    def is_negative(self, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is negative."""
        validator = SymbolicValidator("< 0", lambda x: functions.is_negative(x, tol))
        self._create_assertion_node(validator)

    def is_positive(self, tol: float = functions.EPSILON) -> None:
        """
        Create an assertion that the expression is greater than zero.

        Args:
            tol: Comparison tolerance; values greater than `tol` are considered positive.
        """
        validator = SymbolicValidator("> 0", lambda x: functions.is_positive(x, tol))
        self._create_assertion_node(validator)

    def is_zero(self, tol: float = functions.EPSILON) -> None:
        """
        Create an assertion that the expression equals 0.

        Args:
            tol: Comparison tolerance; values with absolute value less than `tol` are considered zero.
        """
        validator = SymbolicValidator("== 0", lambda x: functions.is_zero(x, tol))
        self._create_assertion_node(validator)

    def noop(self) -> None:
        """
        Create an assertion that records the metric for the current check without performing any validation.

        This assertion collects the underlying metric value but does not evaluate or change the check's pass/fail status.
        """
        validator = SymbolicValidator("", lambda x: True)
        self._create_assertion_node(validator)

    def _create_assertion_node(self, validator: SymbolicValidator) -> None:
        """
        Attach the given SymbolicValidator as a new assertion node to the currently active check.

        If the context is not set, this call is a no-op. If there is no active check, a DQXError is raised.

        Args:
            validator: The validator that defines the assertion to attach.

        Raises:
            DQXError: If no active check is present in the current context.
        """
        if self._context is None:
            return

        current = self._context.current_check
        if not current:
            raise DQXError(
                "Cannot create assertion outside of check context. "
                "Assertions must be created within a @check decorated function."
            )

        # Use the check node's factory method to create and add the assertion
        current.add_assertion(
            actual=self._actual,
            name=self._name,  # Always has a name now!
            severity=self._severity,
            tags=self._tags,
            experimental=self._experimental,
            required=self._required,
            cost_fp=self._cost_fp,
            cost_fn=self._cost_fn,
            validator=validator,
        )

    def _create_assertion_node_with_tunables(
        self, validator: SymbolicValidator, tunables: dict[str, Tunable[Any]]
    ) -> None:
        """
        Attach a validator as a new assertion node, registering tunables used in closures.

        This is used internally when comparison methods capture tunables in validator
        closures that won't be visible in the symbolic expression.

        Args:
            validator: The validator that defines the assertion to attach.
            tunables: Mapping of tunable names to Tunable objects used in validator closure.

        Raises:
            DQXError: If no active check is present in the current context.
        """
        if self._context is None:  # pragma: no cover
            return

        current = self._context.current_check
        if not current:  # pragma: no cover
            raise DQXError(
                "Cannot create assertion outside of check context. "
                "Assertions must be created within a @check decorated function."
            )

        # Use the check node's factory method to create and add the assertion
        current.add_assertion(
            actual=self._actual,
            name=self._name,
            severity=self._severity,
            tags=self._tags,
            experimental=self._experimental,
            required=self._required,
            cost_fp=self._cost_fp,
            cost_fn=self._cost_fn,
            validator=validator,
            tunables=tunables,
        )


@runtime_checkable
class DecoratedCheck(Protocol):
    """Protocol for check functions."""

    __name__: str

    def __call__(self, mp: MetricProvider, ctx: "Context") -> None: ...


class Context:
    """
    Execution context for data quality checks containing the dependency graph and assertion utilities.

    Provides methods to create assertions and manage the verification workflow.
    The Context owns the SymbolTable and provides factory methods for creating
    graph nodes that need access to the symbol table.
    """

    def __init__(self, suite: str, db: "MetricDB", execution_id: str, data_av_threshold: float) -> None:
        """
        Initialize the context with a root graph node.

        Args:
            suite: Name of the verification suite
            db: Database for storing and retrieving metrics
            execution_id: Unique identifier for this execution
            data_av_threshold: Minimum data availability threshold for metrics
        """
        self._graph = Graph(RootNode(name=suite))
        # MetricProvider now creates its own cache internally
        self._provider = MetricProvider(db, execution_id=execution_id, data_av_threshold=data_av_threshold)
        self._local = threading.local()

        # Track the start time of the suite execution
        self._start_time: float = time.time()

    @property
    def _check_stack(self) -> list[CheckNode]:
        if not hasattr(self._local, "check_stack"):
            self._local.check_stack = []
        return self._local.check_stack

    @property
    def start_time(self) -> float:
        """
        The start time of the data quality check suite execution.

        Returns:
            float: The start time in seconds since the epoch.
        """
        return self._start_time

    def tick(self) -> None:
        """
        Reset the start time of the data quality check suite execution.

        This method is usually called between suites to track the total execution time.
        """
        self._start_time = time.time()

    def _push_check(self, check_node: CheckNode) -> None:
        """Push a check onto the thread-local stack."""
        self._check_stack.append(check_node)

    def _pop_check(self) -> CheckNode | None:
        """Pop a check from the thread-local stack."""
        stack = self._check_stack
        return stack.pop() if stack else None

    @property
    def current_check(self) -> CheckNode | None:
        """Get the currently active check node for this thread."""
        stack = self._check_stack
        return stack[-1] if stack else None

    @contextmanager
    def check_context(self, check_node: CheckNode) -> Any:
        """Context manager for check execution."""
        self._push_check(check_node)
        try:
            yield check_node
        finally:
            self._pop_check()

    @property
    def provider(self) -> MetricProvider:
        """Get the metric provider for this context."""
        return self._provider

    def assert_that(self, expr: sp.Expr) -> AssertionDraft:
        """
        Create an assertion draft for the given expression.

        You must provide a name using where() before making assertions:

        Example:
            ctx.assert_that(mp.average("price"))
               .where(name="Average price is positive")
               .is_positive()

        Args:
            expr: Symbolic expression to assert on

        Returns:
            AssertionDraft that requires where() to be called
        """
        return AssertionDraft(actual=expr, context=self)

    def pending_metrics(self, dataset: str | None = None) -> Sequence[SymbolicMetric]:
        """
        Get pending metrics for the specified dataset or all datasets if none specified.

        Returns SymbolicMetric objects that contain both the metric specification
        and the key provider with lag information.
        """
        all_metrics = self.provider.metrics
        if dataset:
            return [metric for metric in all_metrics if metric.dataset == dataset]
        return all_metrics


class VerificationSuite:
    """
    A suite of data quality verification checks that can be executed against multiple data sources.

    The suite collects symbolic assertions through check functions and builds a dependency graph
    of metrics, symbols, and analyzers required to evaluate those assertions. The graph is built
    immediately upon suite initialization, and any Tunable objects used in assertions are
    automatically discovered and made available for tuning.

    Example:
        Basic usage (single run):

        >>> db = MetricDB()
        >>> suite = VerificationSuite([my_check], db, "My Suite")
        >>> datasource = DuckRelationDataSource.from_arrow(data, "dataset")
        >>> result = suite.run([datasource], key)

        Advanced usage with tunables and reset (for AI agents):

        >>> from dqx.tunables import TunableFloat
        >>> threshold = TunableFloat("THRESHOLD", value=0.05, bounds=(0.0, 0.50))
        >>>
        >>> @check(name="My Check")
        >>> def my_check(mp, ctx):
        >>>     null_rate = mp.null_count("col") / mp.num_rows()
        >>>     # Use tunable directly in expression (no .value needed!)
        >>>     ctx.assert_that(null_rate - threshold).where(name="Null rate check").is_lt(0)
        >>>
        >>> suite = VerificationSuite([my_check], db, "My Suite")
        >>> # Tunables are automatically discovered from the checks
        >>> print(suite.get_tunable_params())  # Shows discovered tunables
        >>>
        >>> # Run 1
        >>> suite.run([datasource], key)
        >>> results = suite.collect_results()
        >>>
        >>> # Adjust threshold based on results
        >>> suite.set_param("THRESHOLD", 0.30, agent="rl_optimizer", reason="Tuning iteration")
        >>> suite.reset()  # Rebuild graph with new threshold
        >>>
        >>> # Run 2 with new threshold
        >>> suite.run([datasource], key)
        >>> updated_results = suite.collect_results()
    """

    def __init__(
        self,
        checks: Sequence[CheckProducer | DecoratedCheck] | None = None,
        db: "MetricDB | None" = None,
        name: str | object = _NAME_NOT_PROVIDED,
        dql: Path | None = None,
        log_level: int | str = logging.INFO,
        data_av_threshold: float = 0.9,
        profiles: Sequence[Profile] | None = None,
        config: Path | None = None,
    ) -> None:
        """
        Initialize a VerificationSuite that orchestrates and evaluates a set of data quality checks.

        The suite can be initialized in two ways:
        1. Python API: Pass check functions via `checks` parameter
        2. DQL: Pass DQL file path via `dql` parameter

        Exactly one of `checks` or `dql` must be provided.

        Args:
            checks: Python check functions decorated with @check. Mutually exclusive with `dql`.
            dql: Path to .dql file. Mutually exclusive with `checks`.
            db: Storage backend for producing and retrieving metrics used by checks and analysis.
            name: Human-readable name for the suite.
                  - **Required** when using 'checks' parameter (Python API)
                  - **Must NOT be specified** when using 'dql' parameter
                  - DQL suites must define the suite name in the DQL file itself
            log_level: Logging level for the suite (default: logging.INFO).
            data_av_threshold: Minimum fraction of available data required to evaluate assertions (default: 0.9).
                              Can be overridden by DQL availability_threshold.
            profiles: Optional profiles that alter assertion evaluation behavior.
                     Profiles from config file (if provided) are loaded first, then profiles from
                     this parameter are appended. All profiles are active; there is no override behavior.
            config: Optional path to YAML configuration file for setting initial tunable values and profiles.

        Raises:
            DQXError: If both or neither of checks/dql provided, if name is specified with dql,
                      if name is missing/empty with checks, if DQL suite name is empty, or if config file is invalid.
            DQLSyntaxError: If DQL syntax is invalid or suite name is empty in DQL.

        Examples:
            Python API (existing usage):

            >>> @check(name="Completeness", datasets=["orders"])
            >>> def check_completeness(mp, ctx):
            >>>     ctx.assert_that(mp.null_count("id")).where(
            >>>         name="ID not null"
            >>>     ).is_eq(0)
            >>>
            >>> suite = VerificationSuite(
            >>>     checks=[check_completeness],
            >>>     db=db,
            >>>     name="My Suite",
            >>> )

            DQL with file:

            >>> suite = VerificationSuite(
            >>>     dql=Path("suites/orders.dql"),
            >>>     db=db,
            >>>     config=Path("config/prod.yaml"),
            >>> )
        """
        # Setting up the logger
        setup_logger(level=log_level)

        # Validation: exactly one of checks or dql must be provided
        if (checks is None) == (dql is None):
            raise DQXError(
                "Exactly one of 'checks' or 'dql' must be provided. Use 'checks' for Python API or 'dql' for DQL files."
            )

        # Parse DQL if provided
        if dql is not None:
            # Validate name parameter was NOT provided with dql
            if name is not _NAME_NOT_PROVIDED:
                raise DQXError(
                    "'name' parameter cannot be specified when using 'dql'. "
                    'The suite name must be defined in the DQL file: suite "Name" { ... }'
                )

            from dqx.dql.parser import parse_file

            if db is None:
                raise DQXError("'db' parameter is required when using 'dql'")

            # Parse DQL file (only Path supported now)
            suite_ast = parse_file(dql)

            # Parser validates non-empty suite name, but double-check
            if not suite_ast.name or not suite_ast.name.strip():  # pragma: no cover
                raise DQXError("DQL suite must define a non-empty name. This should have been caught by the parser.")

            self._name = suite_ast.name.strip()

            if suite_ast.availability_threshold is not None:
                self._data_av_threshold = suite_ast.availability_threshold
            else:
                self._data_av_threshold = data_av_threshold

            logger.info("Building checks from DQL...")
            self._checks, self._dql_tunables = self._build_dql_checks(suite_ast)
        else:
            # Python API path - name is REQUIRED
            if not checks:
                raise DQXError("At least one check must be provided")
            if db is None:
                raise DQXError("'db' parameter is required")

            # Validate name was provided and is non-empty
            if name is _NAME_NOT_PROVIDED:
                raise DQXError("'name' parameter is required when using 'checks'")
            if not isinstance(name, str):  # pragma: no cover
                raise DQXError("'name' must be a string")

            name_str = str(name)  # Cast for type checker
            if not name_str or not name_str.strip():
                raise DQXError("Suite name cannot be empty")

            self._checks = list(checks)
            self._name = name_str.strip()
            self._data_av_threshold = data_av_threshold
            self._dql_tunables = {}  # Python API doesn't have DQL tunables

        # Generate unique execution ID
        self._execution_id = str(uuid.uuid4())

        # Create a context with execution_id and data availability threshold
        self._context = Context(
            suite=self._name,
            db=db,
            execution_id=self._execution_id,
            data_av_threshold=self._data_av_threshold,  # type: ignore[arg-type]
        )

        # State tracking for result collection
        self._is_evaluated = False  # Track if assertions have been evaluated
        self._key: ResultKey | None = None  # Store the key used during run()

        # Lazy-loaded plugin manager
        self._plugin_manager: PluginManager | None = None

        # Caching for collect_results
        self._cached_results: list[AssertionResult] | None = None

        # Timer for analyzing phase
        self._analyze_ms = timer_registry.timer("analyzing.time_ms")

        # Store analysis reports by datasource name
        self._analysis_reports: AnalysisReport

        # Cache for metrics stats
        self._metrics_stats: "MetricStats | None" = None

        # Store config path for potential reloading
        self._config_path = config

        # Load profiles from config file if provided
        config_profiles: list[Profile] = []
        config_data: dict[str, Any] | None = None
        if self._config_path is not None:
            logger.info(f"Loading configuration from {self._config_path}")
            from dqx.config import load_config, load_profiles_from_config

            config_data = load_config(self._config_path)

            # Load profiles from config
            config_profiles = load_profiles_from_config(config_data)
            if config_profiles:
                logger.info(f"Loaded {len(config_profiles)} profile(s) from config")

        # Merge profiles: config + API (both active)
        # Check for duplicate names between config and API profiles
        if config_profiles and profiles:
            config_names = {p.name for p in config_profiles}
            api_names = {p.name for p in profiles}
            duplicates = config_names & api_names
            if duplicates:
                raise DQXError(
                    f"Duplicate profile name(s) {sorted(duplicates)} found in both config and API parameters. "
                    f"Profile names must be unique. Rename one of the profiles."
                )

        all_profiles = list(config_profiles)
        if profiles is not None:
            all_profiles.extend(profiles)
        self._profiles = all_profiles

        # Build the dependency graph immediately
        logger.info("Building dependency graph for suite '%s'...", self._name)
        self.build_graph(self._context)

        # Collect tunables from the graph automatically
        graph_tunables = collect_tunables_from_graph(self._context._graph)

        # Merge DQL tunables with graph-collected tunables
        # DQL tunables take precedence (they are the source of truth)
        self._tunables = {**graph_tunables, **self._dql_tunables}

        if self._tunables:
            logger.info(f"Discovered {len(self._tunables)} tunable(s): {list(self._tunables.keys())}")

        # Load tunable values from config file if provided
        if self._config_path is not None and config_data is not None:
            from dqx.config import apply_tunables_from_config

            apply_tunables_from_config(config_data, self._tunables)
            logger.info("Applied tunable values from config")

    @property
    def execution_id(self) -> str:
        """
        Unique identifier for this suite execution.

        Returns a UUID string that uniquely identifies this instance of the
        VerificationSuite. This ID is generated when the suite is created
        and remains constant throughout its lifetime.

        Returns:
            str: UUID string for this execution
        """
        return self._execution_id

    @property
    def graph(self) -> Graph:
        """
        Access the dependency graph for the verification suite.

        The graph is built during suite initialization and is immediately
        available after the VerificationSuite is constructed.

        Returns:
            Graph: The dependency graph containing checks and assertions

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> print(f"Graph has {len(list(suite.graph.checks()))} checks")
            >>> # Can access graph immediately after construction
        """
        return self._context._graph

    @property
    def analysis_reports(self) -> AnalysisReport:
        """
        Access the analysis report generated by the suite.

        This property provides read-only access to the analysis report
        generated after the suite has been run.

        Returns:
            AnalysisReport: The analysis report containing metric values and symbol evaluations.

        Raises:
            DQXError: If accessed before the suite has been run.
        """
        self.assert_is_evaluated()
        return self._analysis_reports

    @property
    def provider(self) -> MetricProvider:
        """
        The metric provider instance used by the verification suite.

        This property returns the MetricProvider instance that is used by the verification suite to access and manage metrics.

        Returns:
            MetricProvider instance used by the verification suite
        """
        return self._context.provider

    @property
    def metrics_stats(self) -> "MetricStats":
        """
        Retrieve cached metrics statistics for the suite.

        Returns:
            MetricStats: Total and expired metric counts.

        Raises:
            DQXError: If the suite has not been evaluated or metrics stats are unavailable.
        """
        self.assert_is_evaluated()
        if self._metrics_stats is None:
            raise DQXError(
                "Metrics stats not available. This should not happen after successful run()."
            )  # pragma: no cover
        return self._metrics_stats

    @property
    def plugin_manager(self) -> PluginManager:
        """
        Lazy-load plugin manager on first access.

        Returns:
            PluginManager instance, creating one if it doesn't exist
        """
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager()
        return self._plugin_manager

    @property
    def key(self) -> ResultKey:
        """
        Get the ResultKey produced by the last successful run of the suite.

        Returns:
            ResultKey: The ResultKey for the most recent run.

        Raises:
            DQXError: If the suite has not been run yet and no ResultKey is available.
        """
        if self._key is None:
            raise DQXError("No ResultKey available. This should not happen after successful run().")  # pragma: no cover
        return self._key

    @property
    def data_av_threshold(self) -> float:
        """
        Minimum data availability threshold for assertion evaluation.

        Assertions depending on metrics with availability below this
        threshold will be marked as SKIPPED rather than evaluated.

        Returns:
            Float between 0.0 and 1.0 (default: 0.9)
        """
        return self._data_av_threshold

    @property
    def is_evaluated(self) -> bool:
        """
        Check if the suite has been evaluated.

        Returns:
            True if the suite has been executed, False otherwise.
        """
        return self._is_evaluated

    def assert_is_evaluated(self) -> None:
        """
        Ensure the suite has been evaluated before proceeding.

        Raises:
            DQXError: If the suite has not been evaluated yet
        """
        if not self._is_evaluated:
            raise DQXError("Verification suite has not been executed yet!")

    # -------------------------------------------------------------------------
    # Tunable API for RL agent integration
    # -------------------------------------------------------------------------

    def get_tunable_params(self) -> list[dict[str, Any]]:
        """
        List all tunable parameters available for the suite's reinforcement-learning action space.

        Returns:
            list[dict[str, Any]]: A list of dictionaries where each dictionary describes a tunable and includes keys such as:
                - "name": the tunable's identifier
                - "type": the tunable's data or semantic type (e.g., "percent", "int", "categorical")
                - "value": the current value
                - "bounds" or "choices": numeric bounds as a (min, max) tuple for continuous tunables or an iterable of allowed choices for categorical tunables
        """
        return [t.to_dict() for t in self._tunables.values()]

    def get_param(self, name: str) -> Any:
        """
        Retrieve the current value of a tunable parameter.

        Args:
            name: Name of the tunable parameter.

        Returns:
            The current value of the tunable.

        Raises:
            KeyError: If a tunable with the given name does not exist.
        """
        if name not in self._tunables:
            raise KeyError(f"Tunable '{name}' not found. Available: {list(self._tunables.keys())}")
        return self._tunables[name].value

    def set_param(self, name: str, value: Any, agent: str = "human", reason: str | None = None) -> None:
        """
        Update the value of a tunable and record the change in its history.

        Args:
            name: Name of the tunable parameter to update.
            value: New value to assign to the tunable; must satisfy the tunable's constraints.
            agent: Identifier of who made the change (e.g., "human", "rl_optimizer", "autotuner").
            reason: Optional human-readable explanation for the change.

        Raises:
            KeyError: If no tunable with the given name exists.
            ValueError: If the provided value violates the tunable's validation rules or bounds.
        """
        if name not in self._tunables:
            raise KeyError(f"Tunable '{name}' not found. Available: {list(self._tunables.keys())}")
        self._tunables[name].set(value, agent=agent, reason=reason)

    def get_param_history(self, name: str) -> list[TunableChange]:
        """
        Return the change history for a named tunable parameter.

        Args:
            name: Name of the tunable parameter.

        Returns:
            list[TunableChange]: List of TunableChange records for the specified tunable.

        Raises:
            KeyError: If a tunable with the given name does not exist.
        """
        if name not in self._tunables:
            raise KeyError(f"Tunable '{name}' not found. Available: {list(self._tunables.keys())}")
        return self._tunables[name].history

    def reset(self) -> None:
        """
        Reset the verification suite to allow running it again with modified tunables.

        This method clears all execution state (results, analysis reports) while
        preserving the suite configuration and tunable values. The graph is rebuilt
        to ensure all assertions reflect the current tunable values. Each reset
        generates a new execution_id to distinguish tuning iterations in the metrics
        database.

        Primary use case: AI agents tuning threshold parameters via Tunables.

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> suite.run([datasource], key)
            >>> result1 = suite.collect_results()[0].status  # "FAILED"
            >>>
            >>> # Tune threshold and try again
            >>> suite.set_param("THRESHOLD", 0.30, agent="rl_optimizer")
            >>> suite.reset()
            >>> suite.run([datasource], key)
            >>> result2 = suite.collect_results()[0].status  # "PASSED"

        Note:
            - Generates a new execution_id for the next run
            - Preserves tunables, profiles, checks, and suite name
            - Preserves current tunable values (does NOT reload config file)
            - Tunables retain any values set via set_param() after config was loaded
            - Rebuilds the graph to reflect current tunable values
            - Clears cached results, analysis reports
            - Clears plugin manager (will be lazy-loaded on next use)
        """
        # Generate new execution_id for the next run
        self._execution_id = str(uuid.uuid4())

        # Create fresh context with new execution_id
        self._context = Context(
            suite=self._name,
            db=self._context.provider.db,
            execution_id=self._execution_id,
            data_av_threshold=self._data_av_threshold,
        )

        # Rebuild the graph with updated tunable values
        logger.info("Rebuilding dependency graph after reset...")
        self.build_graph(self._context)

        # Re-collect tunables from the rebuilt graph
        # Merge both graph-discovered tunables and DQL tunables to preserve all
        graph_tunables = collect_tunables_from_graph(self._context._graph)
        self._tunables = {**graph_tunables, **self._dql_tunables}

        # Clear execution state
        self._is_evaluated = False
        self._key = None
        self._cached_results = None
        self._analysis_reports = None  # type: ignore[assignment]
        self._metrics_stats = None

        # Clear plugin manager (will be lazy-loaded on next use)
        self._plugin_manager = None

    def _build_dql_checks(self, suite_ast: Any) -> tuple[list[CheckProducer], dict[str, Tunable[Any]]]:
        """
        Convert DQL Suite AST to Python check functions.

        Args:
            suite_ast: Parsed DQL Suite AST

        Returns:
            Tuple of (list of check functions, dict of tunables defined in DQL)
        """
        # Build tunables dict for expression substitution
        tunables_dict = self._build_tunables_from_ast(suite_ast.tunables)

        # Build each check
        checks: list[CheckProducer] = []
        for check_ast in suite_ast.checks:
            check_func = self._build_check_from_ast(check_ast, tunables_dict)
            checks.append(check_func)

        return checks, tunables_dict

    def _build_tunables_from_ast(self, tunables_ast: tuple[Any, ...]) -> dict[str, Tunable[Any]]:
        """Convert DQL Tunable AST nodes to Tunable objects.

        Creates TunableInt or TunableFloat based on type inference.
        Type inference: if all components (value, min, max) are integers → TunableInt,
        otherwise → TunableFloat.

        Initial value is validated against bounds by Tunable.__post_init__().

        Args:
            tunables_ast: Tuple of Tunable AST nodes from DQL parser

        Returns:
            dict mapping tunable names to Tunable objects

        Raises:
            DQLError: If initial value is outside bounds or bounds are invalid
            DQLSyntaxError: If tunable name conflicts with built-in metric function
        """
        from dqx.dql.errors import DQLError, DQLSyntaxError
        from dqx.tunables import TunableFloat, TunableInt

        # Reserved names (built-in metric function names that should not be shadowed)
        # NOTE: Keep in sync with _build_metric_namespace()
        RESERVED_NAMES = {
            # Math functions (sympy builtins)
            "abs",
            "sqrt",
            "log",
            "exp",
            "min",
            "max",
            # Base aggregate metrics
            "num_rows",
            "null_count",
            "null_rate",
            "average",
            "sum",
            "minimum",
            "maximum",
            "variance",
            "stddev",
            # Completeness/value metrics
            "unique_count",
            "duplicate_count",
            "distinct_count",
            "count_values",
            "first",
            # Extension metrics (temporal comparisons)
            "day_over_day",
            "week_over_week",
            # Utility functions
            "coalesce",
            "custom_sql",
        }

        tunables_dict: dict[str, Tunable[Any]] = {}
        for t in tunables_ast:
            # Check for reserved names
            if t.name in RESERVED_NAMES:
                raise DQLSyntaxError(
                    f"Tunable name '{t.name}' conflicts with built-in metric function",
                    loc=t.loc,
                )

            # Evaluate bounds and value using simple expressions
            # Can reference previously defined tunables
            min_val = self._eval_simple_expr(t.bounds[0], tunables_dict)
            max_val = self._eval_simple_expr(t.bounds[1], tunables_dict)
            value = self._eval_simple_expr(t.value, tunables_dict)

            # Type inference: if all are ints → TunableInt, otherwise TunableFloat
            tunable: Tunable[Any]
            if isinstance(value, int) and isinstance(min_val, int) and isinstance(max_val, int):
                try:
                    tunable = TunableInt(name=t.name, value=int(value), bounds=(int(min_val), int(max_val)))
                except ValueError as e:
                    # TunableInt validation failed (value outside bounds or invalid bounds)
                    raise DQLError(f"Tunable '{t.name}': {e}", loc=t.loc) from None
            else:
                # Promote to float if any component is float
                try:
                    tunable = TunableFloat(name=t.name, value=float(value), bounds=(float(min_val), float(max_val)))
                except ValueError as e:
                    raise DQLError(f"Tunable '{t.name}': {e}", loc=t.loc) from None

            # Add to dict immediately so subsequent tunables can reference it
            tunables_dict[t.name] = tunable

        return tunables_dict

    def _build_check_from_ast(self, check_ast: Any, tunables: dict[str, Tunable[Any]]) -> CheckProducer:
        """Convert DQL Check AST to Python check function."""
        check_name = check_ast.name
        assertions = check_ast.assertions
        tunables_copy = dict(tunables)

        @check(name=check_name, datasets=list(check_ast.datasets))
        def dynamic_check(mp: MetricProvider, ctx: Context) -> None:
            """Generated check function from DQL."""
            # Execute each assertion
            for assertion_ast in assertions:
                self._build_statement(assertion_ast, mp, ctx, tunables_copy)

        return dynamic_check

    def _build_statement(
        self,
        statement: Any,
        mp: MetricProvider,
        ctx: Context,
        tunables: dict[str, Tunable[Any]],
    ) -> None:
        """Convert DQL Assertion or Collection to ctx.assert_that() call."""
        from dqx.dql.ast import Collection

        # Dispatch based on type
        if isinstance(statement, Collection):
            self._build_collection(statement, mp, ctx, tunables)
        else:
            self._build_assertion(statement, mp, ctx, tunables)

    def _build_assertion(
        self,
        assertion_ast: Any,
        mp: MetricProvider,
        ctx: Context,
        tunables: dict[str, Tunable[Any]],
    ) -> None:
        """Convert DQL Assertion to ctx.assert_that() call."""
        ready = self._setup_assertion_ready(assertion_ast, mp, ctx, tunables)
        if ready is None:  # pragma: no cover
            return

        # Apply condition
        self._apply_condition(ready, assertion_ast, tunables)

    def _build_collection(
        self,
        collection_ast: Any,
        mp: MetricProvider,
        ctx: Context,
        tunables: dict[str, Tunable[Any]],
    ) -> None:
        """Convert DQL Collection to ctx.assert_that(...).noop() call."""
        ready = self._setup_assertion_ready(collection_ast, mp, ctx, tunables)
        if ready is None:  # pragma: no cover
            return

        # Call noop() instead of applying a condition
        ready.noop()

    def _setup_assertion_ready(
        self,
        statement: Any,
        mp: MetricProvider,
        ctx: Context,
        tunables: dict[str, Tunable[Any]],
    ) -> Any | None:
        """Common setup for assertions and collections.

        Returns AssertionReady object or None if statement is disabled.

        Args:
            statement: Assertion or Collection AST node
            mp: MetricProvider for evaluating metric expressions
            ctx: Context for creating assertions
            tunables: Tunable values for expression substitution

        Returns:
            AssertionReady object if statement should be processed, None if disabled
        """
        # Evaluate metric expression
        metric_value = self._eval_metric_expr(statement.expr, mp, tunables)

        # Use severity from DQL statement
        severity = statement.severity

        # Validate name is present
        if statement.name is None:  # pragma: no cover
            # Parser should have caught this
            from dqx.dql.errors import DQLError

            raise DQLError("Statement must have a name", loc=statement.loc)

        # Extract cost annotation
        cost_annotation = self._get_cost_annotation(statement)
        # cost_annotation already contains floats, no conversion needed
        cost_dict = cost_annotation if cost_annotation else None

        # Build and return assertion ready object
        return ctx.assert_that(metric_value).where(
            name=statement.name,
            severity=severity.value,
            tags=set(statement.tags),
            experimental=self._has_annotation(statement, "experimental"),
            required=self._has_annotation(statement, "required"),
            cost=cost_dict,
        )

    # === DQL Expression Evaluation Methods ===

    def _eval_metric_expr(self, expr: Any, mp: MetricProvider, tunables: dict[str, Tunable[Any]]) -> Any:
        """Parse metric expression using sympy, injecting tunables as TunableSymbol.

        Args:
            expr: Expression AST node with .text attribute
            mp: MetricProvider for accessing metric functions
            tunables: Dictionary of Tunable objects (injected as TunableSymbol)

        Returns:
            SymPy expression with tunables as TunableSymbol instances
        """
        from dqx.tunables import TunableSymbol

        expr_text = expr.text

        # Handle stddev specially since it has named params that sympy doesn't understand
        if "stddev(" in expr_text and (", n=" in expr_text or ", offset=" in expr_text):
            return self._handle_stddev_extension(expr_text, mp, tunables)

        # Build namespace with metric functions
        namespace = self._build_metric_namespace(mp)

        # Inject tunables as TunableSymbol
        for name, tunable in tunables.items():
            namespace[name] = TunableSymbol(tunable)

        # Parse with sympy - tunables will appear as TunableSymbol in expression
        try:
            return sp.sympify(expr_text, locals=namespace, evaluate=False)
        except (sp.SympifyError, TypeError, ValueError) as e:  # pragma: no cover
            # Sympy parsing error (very rare with valid DQL)
            from dqx.dql.errors import DQLError

            raise DQLError(
                f"Failed to parse metric expression: {expr.text}\n{e}",
                loc=expr.loc,
            ) from e

    def _eval_simple_expr(self, expr: Any, tunables: dict[str, Tunable[Any]]) -> float:
        """Evaluate numeric expressions with full SymPy arithmetic support.

        Supports:
        - Numeric literals: 42, 3.14
        - Percentages: 50% (converted to 0.5)
        - Arithmetic: +, -, *, /, **
        - Tunable references: MIN_VAL, MAX_VAL (extracts .value)
        - Complex expressions: (MIN_VAL + 10) * 2

        Args:
            expr: Expression AST node with .text attribute
            tunables: Dictionary mapping tunable names to Tunable objects

        Returns:
            Evaluated numeric result (int or float)

        Raises:
            DQLError: If expression cannot be evaluated
        """
        import sympy as sp

        from dqx.dql.errors import DQLError
        from dqx.tunables import TunableSymbol

        text = expr.text.strip()

        # Handle percentages (convert to decimal)
        # Note: Parser already converts % tokens, this is defensive
        if text.endswith("%"):  # pragma: no cover
            text = str(float(text[:-1]) / 100)

        try:
            # Inject tunables as numeric values for immediate evaluation
            namespace = {name: tunable.value for name, tunable in tunables.items()}
            result = sp.sympify(text, locals=namespace, evaluate=True)

            # Handle TunableSymbol case (defensive - shouldn't happen with evaluate=True)
            if isinstance(result, TunableSymbol):  # pragma: no cover
                return float(result.value)

            # Handle raw Python int/float (when sympify returns the namespace value directly)
            if isinstance(result, (int, float)):
                return result

            # Preserve int vs float type for SymPy objects
            if result.is_Integer:
                return int(result)
            return float(result)
        except (sp.SympifyError, Exception) as e:
            raise DQLError(f"Cannot evaluate expression: {text}", loc=expr.loc) from e

    def _handle_stddev_extension(self, expr_text: str, mp: MetricProvider, tunables: dict[str, Tunable[Any]]) -> Any:
        """Handle stddev extension function with named parameters.

        Parses stddev calls with optional offset and required n parameters in any order:
        - stddev(expr, n=7)
        - stddev(expr, offset=1, n=7)
        - stddev(expr, n=7, offset=1)

        Args:
            expr_text: Expression string containing stddev call
            mp: MetricProvider for accessing extension functions
            tunables: Dictionary of Tunable objects (injected as TunableSymbol)

        Returns:
            Result of mp.ext.stddev() call via sp.sympify()
        """
        import re

        from dqx.tunables import TunableSymbol

        # Find the matching closing paren for stddev( by counting parentheses
        # This properly handles nested function calls like stddev(day_over_day(avg(x)), n=7)
        stddev_start = expr_text.find("stddev(")
        if stddev_start == -1:  # pragma: no cover
            # Should not happen - caller checks for stddev( first
            namespace = self._build_metric_namespace(mp)
            # Inject tunables
            for name, tunable in tunables.items():
                namespace[name] = TunableSymbol(tunable)
            return sp.sympify(expr_text, locals=namespace, evaluate=False)

        # Start after "stddev("
        pos = stddev_start + 7
        paren_count = 1
        inner_start = pos

        # Find the matching closing paren
        while pos < len(expr_text) and paren_count > 0:
            if expr_text[pos] == "(":
                paren_count += 1
            elif expr_text[pos] == ")":
                paren_count -= 1
            pos += 1

        if paren_count != 0:  # pragma: no cover - defensive fallback
            # Malformed expression - fallback to normal parsing
            namespace = self._build_metric_namespace(mp)
            # Inject tunables
            for name, tunable in tunables.items():
                namespace[name] = TunableSymbol(tunable)
            return sp.sympify(expr_text, locals=namespace, evaluate=False)

        # Extract everything inside stddev(...)
        inner_content = expr_text[inner_start : pos - 1]

        # Split inner content by commas, being careful about nested functions
        # Look for the first comma that's at the top level (not inside nested parens)
        parts = []
        current_part = []
        paren_depth = 0

        for char in inner_content:
            if char == "(":
                paren_depth += 1
                current_part.append(char)
            elif char == ")":
                paren_depth -= 1
                current_part.append(char)
            elif char == "," and paren_depth == 0:
                parts.append("".join(current_part).strip())
                current_part = []
            else:
                current_part.append(char)

        # Don't forget the last part
        if current_part:
            parts.append("".join(current_part).strip())

        # First part is the inner expression
        inner_expr_text = parts[0] if parts else ""

        # Extract offset and n parameters from remaining parts
        offset = 0  # Default offset
        n = None  # Will be required if params exist

        for part in parts[1:]:
            # Match offset=N or n=N with optional whitespace and decimal support
            # Pattern matches: offset=1, offset=1.0, n=7, n=7.0, etc.
            num_pattern = r"([0-9]+(?:\.[0-9]+)?)"
            offset_match = re.search(rf"offset\s*=\s*{num_pattern}", part)
            n_match = re.search(rf"n\s*=\s*{num_pattern}", part)

            if offset_match:
                offset_val = float(offset_match.group(1))
                if not offset_val.is_integer():
                    raise ValueError(f"stddev offset must be an integer, got: {offset_val}")
                offset = int(offset_val)
            if n_match:
                n_val = float(n_match.group(1))
                if not n_val.is_integer():
                    raise ValueError(f"stddev n must be an integer, got: {n_val}")
                n = int(n_val)

        # If params exist but n is not found, this shouldn't happen with valid DQL
        # The grammar should enforce n parameter when using stddev with params
        if len(parts) > 1 and n is None:  # pragma: no cover
            raise ValueError(f"stddev requires 'n' parameter: {expr_text}")

        # Parse the inner expression via _build_metric_namespace and sp.sympify
        # Don't recursively call _eval_metric_expr to avoid infinite loop
        namespace = self._build_metric_namespace(mp)
        # Inject tunables
        for name, tunable in tunables.items():  # pragma: no cover
            namespace[name] = TunableSymbol(tunable)  # pragma: no cover
        inner_metric = sp.sympify(inner_expr_text, locals=namespace, evaluate=False)

        # Call mp.ext.stddev with parsed parameters
        # Note: stddev without params is handled by normal sympy evaluation
        if n is not None:
            result = mp.ext.stddev(inner_metric, offset=offset, n=n)
        else:  # pragma: no cover - stddev without params shouldn't reach here
            # This path shouldn't be reached - stddev without params won't match
            # the condition in _eval_metric_expr that calls this method
            namespace = self._build_metric_namespace(mp)
            # Inject tunables
            for name, tunable in tunables.items():
                namespace[name] = TunableSymbol(tunable)
            return sp.sympify(expr_text, locals=namespace, evaluate=False)

        return result

    def _build_metric_namespace(self, mp: MetricProvider) -> dict[str, Any]:
        """Build sympy namespace with all metric and math functions."""

        def _to_str(arg: Any) -> str:
            """Convert sympy Symbol to string."""
            return str(arg) if isinstance(arg, sp.Symbol) else arg

        def _convert_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
            """Convert sympy types in kwargs to Python primitives.

            Handles:
            - sp.Integer -> int (lag=1, n=7, offset=1)
            - sp.Float -> float
            - sp.Symbol -> str (dataset=ds1)
            - Other types pass through unchanged
            """
            result: dict[str, Any] = {}
            for key, value in kw.items():
                if isinstance(value, sp.Basic):
                    # Convert sympy numbers to Python int/float
                    if value.is_Integer:
                        result[key] = int(value)  # type: ignore[arg-type]
                    elif value.is_Float or value.is_Rational:  # pragma: no cover - rare sympy output
                        result[key] = float(value)  # type: ignore[arg-type]  # pragma: no cover
                    elif isinstance(value, sp.Symbol):
                        # For symbols (like dataset names), convert to string
                        result[key] = str(value)
                    else:  # pragma: no cover - defensive fallback for unknown sympy types
                        # For other sympy types, try to extract value
                        try:  # pragma: no cover
                            result[key] = float(value)  # type: ignore[arg-type]  # pragma: no cover
                        except (TypeError, AttributeError):  # pragma: no cover
                            result[key] = str(value)  # pragma: no cover
                else:
                    result[key] = value  # pragma: no cover - passthrough of non-sympy values
            return result

        def _convert_list_arg(cols: Any) -> list[str]:
            """Convert list of Symbols/tokens to list of strings.

            Handles:
            - [Symbol('name')] -> ['name']
            - [Symbol('id'), Symbol('date')] -> ['id', 'date']
            - Symbol('name') -> ['name'] (single column case)
            """
            if isinstance(cols, list):
                return [_to_str(item) for item in cols]
            elif isinstance(cols, tuple):  # pragma: no cover - tuples not produced by parser
                return [_to_str(item) for item in cols]  # pragma: no cover
            else:  # pragma: no cover - single column fallback
                # Single column passed without list brackets
                return [_to_str(cols)]  # pragma: no cover

        def _convert_value(val: Any) -> int | str | bool:
            """Convert value argument to proper Python type for count_values.

            Handles:
            - sp.Integer/Zero/One -> int
            - sp.Float -> int (rounded)
            - sp.Symbol -> str
            - bool/int/str -> unchanged
            - float -> int (rounded)
            """
            if isinstance(val, sp.Basic):
                # Convert sympy types
                if val.is_Integer:
                    return int(val)  # type: ignore[arg-type]
                elif val.is_Float or val.is_Rational:  # pragma: no cover - rare sympy output
                    # Convert float to int for count_values
                    return int(float(val))  # type: ignore[arg-type]  # pragma: no cover
                elif isinstance(val, sp.Symbol):  # pragma: no cover - symbols handled elsewhere
                    return str(val)  # pragma: no cover
                else:  # pragma: no cover - defensive fallback for unknown sympy types
                    # Try to extract numeric value
                    try:  # pragma: no cover
                        return int(val)  # type: ignore[arg-type]  # pragma: no cover
                    except (TypeError, AttributeError):  # pragma: no cover
                        return str(val)  # pragma: no cover
            elif isinstance(val, float):
                # Convert float to int for count_values
                return int(val)  # pragma: no cover - float values converted at parse time
            elif isinstance(val, (int, str, bool)):
                return val
            else:  # pragma: no cover - defensive fallback for unknown types
                return str(val)  # pragma: no cover

        namespace = {
            # Math functions
            "abs": sp.Abs,
            "sqrt": sp.sqrt,
            "log": sp.log,
            "exp": sp.exp,
            "min": sp.Min,
            "max": sp.Max,
            # Base metrics - convert Symbol args to strings and kwargs to Python types
            "num_rows": lambda **kw: mp.num_rows(**_convert_kwargs(kw)),
            "null_count": lambda col, **kw: mp.null_count(_to_str(col), **_convert_kwargs(kw)),
            "average": lambda col, **kw: mp.average(_to_str(col), **_convert_kwargs(kw)),
            "sum": lambda col, **kw: mp.sum(_to_str(col), **_convert_kwargs(kw)),
            "minimum": lambda col, **kw: mp.minimum(_to_str(col), **_convert_kwargs(kw)),
            "maximum": lambda col, **kw: mp.maximum(_to_str(col), **_convert_kwargs(kw)),
            "variance": lambda col, **kw: mp.variance(_to_str(col), **_convert_kwargs(kw)),
            "unique_count": lambda col, **kw: mp.unique_count(_to_str(col), **_convert_kwargs(kw)),
            "duplicate_count": lambda cols, **kw: mp.duplicate_count(_convert_list_arg(cols), **_convert_kwargs(kw)),
            "count_values": lambda col, val, **kw: mp.count_values(
                _to_str(col), _convert_value(val), **_convert_kwargs(kw)
            ),
            "first": lambda col, **kw: mp.first(_to_str(col), **_convert_kwargs(kw)),
            # Utility functions
            "coalesce": lambda *args: functions.Coalesce(*args),
            # Extension functions
            "day_over_day": lambda metric, **kw: mp.ext.day_over_day(metric, **_convert_kwargs(kw)),
            "week_over_week": lambda metric, **kw: mp.ext.week_over_week(metric, **_convert_kwargs(kw)),
            # Note: stddev with n parameter is handled specially in _handle_stddev_extension
            # RAW SQL ESCAPE HATCH
            "custom_sql": lambda expr: mp.custom_sql(_to_str(expr)),
        }
        return namespace

    def _apply_condition(self, ready: Any, assertion_ast: Any, tunables: dict[str, Tunable[Any]]) -> None:
        """Apply the assertion condition to AssertionReady."""
        cond = assertion_ast.condition

        # Validate threshold is present for conditions that require it
        requires_threshold = cond in (">", ">=", "<", "<=", "==", "!=", "between")
        if requires_threshold and assertion_ast.threshold is None:  # pragma: no cover
            # Parser ensures threshold is present for these conditions
            from dqx.dql.errors import DQLError

            raise DQLError(f"Condition '{cond}' requires a threshold", assertion_ast.loc)

        if cond == ">":
            threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
            if assertion_ast.tolerance:
                ready.is_gt(threshold, tol=assertion_ast.tolerance)
            else:
                ready.is_gt(threshold)
        elif cond == ">=":
            threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
            if assertion_ast.tolerance:
                ready.is_geq(threshold, tol=assertion_ast.tolerance)
            else:
                ready.is_geq(threshold)
        elif cond == "<":
            threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
            if assertion_ast.tolerance:
                ready.is_lt(threshold, tol=assertion_ast.tolerance)
            else:
                ready.is_lt(threshold)
        elif cond == "<=":
            threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
            if assertion_ast.tolerance:
                ready.is_leq(threshold, tol=assertion_ast.tolerance)
            else:
                ready.is_leq(threshold)
        elif cond == "==":
            threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
            if assertion_ast.tolerance:
                ready.is_eq(threshold, tol=assertion_ast.tolerance)
            else:
                ready.is_eq(threshold)
        elif cond == "!=":
            threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)
            ready.is_neq(threshold)
        elif cond == "between":
            if assertion_ast.threshold_upper is None:  # pragma: no cover
                # Parser ensures upper threshold is present for between
                from dqx.dql.errors import DQLError

                raise DQLError("Condition 'between' requires upper threshold", assertion_ast.loc)
            lower = self._eval_simple_expr(assertion_ast.threshold, tunables)
            upper = self._eval_simple_expr(assertion_ast.threshold_upper, tunables)
            ready.is_between(lower, upper)
        elif cond == "is":
            if assertion_ast.keyword == "positive":
                ready.is_positive()
            elif assertion_ast.keyword == "negative":
                ready.is_negative()
            else:  # pragma: no cover
                # Parser ensures only 'positive' or 'negative' keywords
                from dqx.dql.errors import DQLError

                raise DQLError(f"Unknown keyword: {assertion_ast.keyword}", assertion_ast.loc)
        else:  # pragma: no cover
            # Parser validates all conditions
            from dqx.dql.errors import DQLError

            raise DQLError(f"Unknown condition: {cond}", assertion_ast.loc)

    def _has_annotation(self, statement_ast: Any, name: str) -> bool:
        """Check if assertion or collection has a specific annotation."""
        return any(ann.name == name for ann in statement_ast.annotations)

    def _get_cost_annotation(self, statement_ast: Any) -> dict[str, float] | None:
        """Extract cost annotation args if present."""
        for ann in statement_ast.annotations:
            if ann.name == "cost":
                # Convert args to expected format (preserve floats for fractional costs)
                return {
                    "fp": float(ann.args.get("false_positive", 1.0)),
                    "fn": float(ann.args.get("false_negative", 1.0)),
                }
        return None

    def build_graph(self, context: Context) -> None:
        """
        Populate the execution graph by running all registered checks and validate it.

        Runs each check to add nodes and assertions into the provided Context's graph, then validates the assembled graph using SuiteValidator. If validation reports errors a DQXError is raised; validation warnings are emitted to the logger.

        Args:
            context: Execution context that holds the graph and provider.

        Raises:
            DQXError: If the graph validation reports errors.
        """
        # Execute all checks to collect assertions
        for check in self._checks:
            check(self.provider, context)

        # Create validator locally instead of using instance attribute
        validator = SuiteValidator()
        report = validator.validate(context._graph, context.provider)

        # Only raise on errors, log warnings
        if report.has_errors():
            raise DQXError(f"Suite validation failed:\n{report}")
        elif report.has_warnings():
            logger.debug(f"Suite validation warnings:\n{report}")

    def _analyze(self, datasources: list[SqlDataSource], key: ResultKey) -> None:
        """Analyze datasources using the new analyze_all function."""

        # Call analyze_all and store the results
        analyzer = Analyzer(
            datasources,
            self.provider,
            key,
            execution_id=self._execution_id,
            data_av_threshold=self._data_av_threshold,
        )
        self._analysis_reports = analyzer.analyze()

    def run(self, datasources: list[SqlDataSource], key: ResultKey, *, enable_plugins: bool = True) -> None:
        """
        Run the verification suite against the given data sources and produce evaluation results stored on the suite.

        Args:
            datasources: Data sources to analyze.
            key: Result key that defines the time period and associated tags for this run.
            enable_plugins: If True, execute registered plugins after evaluation (default True).

        Raises:
            DQXError: If no data sources are provided or the suite has already been executed.
        """

        # Prevent multiple runs
        if self.is_evaluated:
            raise DQXError(
                "Verification suite has already been executed. "
                "Call reset() to clear state and run again, or create a new suite instance."
            )

        # Validate the datasources
        if not datasources:
            raise DQXError("No data sources provided!")

        logger.info(f"Running verification suite '{self._name}' with datasets: {[ds.name for ds in datasources]}")
        logger.info("Execution id: %s", self.execution_id)
        active_profiles = [p.name for p in self._profiles if p.is_active(key.yyyy_mm_dd)]
        logger.info("Active profiles: %s", active_profiles if active_profiles else None)

        # Log tunable values being used for this evaluation
        if self._tunables:
            from dqx.tunables import TunableChoice, TunableFloat, TunableInt

            logger.info(f"Evaluating with {len(self._tunables)} tunable(s):")
            for name, tunable in sorted(self._tunables.items()):
                if isinstance(tunable, TunableFloat):
                    logger.info(f"  - {name} (float) = {tunable.value} [{tunable.lower_bound}-{tunable.upper_bound}]")
                elif isinstance(tunable, TunableInt):
                    logger.info(f"  - {name} (int) = {tunable.value} [{tunable.lower_bound}-{tunable.upper_bound}]")
                elif isinstance(tunable, TunableChoice):
                    choices_str = ", ".join(map(str, tunable.choices))
                    logger.info(f"  - {name} (choice) = {tunable.value!r} [{choices_str}]")

        # Store the key for later use in collect_results
        self._key = key

        # Reset the run timer
        self._context.tick()

        # Graph is already built in __init__(), no need to build again

        # 1. Impute datasets using visitor pattern
        # Use graph in the context to avoid the check if the suite has been evaluated
        logger.info("Imputing datasets...")
        self._context._graph.impute_datasets([ds.name for ds in datasources], self._context.provider)

        # Apply symbol deduplication BEFORE analysis
        self._context.provider.symbol_deduplication(self._context._graph, key)

        # Calculate data availability ratios for date exclusion
        # Create datasources dict for calculate_data_av_ratios
        datasources_dict = {ds.name: ds for ds in datasources}
        logger.info("Calculating data availability ratios for datasets")
        self._context.provider.registry.calculate_data_av_ratios(datasources_dict, key)

        # Collect metrics stats and cleanup expired metrics BEFORE analysis
        self._metrics_stats = self.provider.db.get_metrics_stats()
        logger.info(
            f"Metrics stats: {self._metrics_stats.expired_metrics} expired "
            f"out of {self._metrics_stats.total_metrics} total"
        )

        # Cleanup expired metrics before analysis
        if self._metrics_stats.expired_metrics > 0:  # pragma: no cover
            logger.info("Cleaning up expired metrics...")
            self.cleanup_expired_metrics()

        # 2. Analyze by datasources
        with self._analyze_ms:
            self._analyze(datasources, key)

        # 3. Evaluate assertions
        # Use graph in the context to avoid the check if the suite has been evaluated
        evaluator = Evaluator(self.provider, key, self._name, self._data_av_threshold, self._profiles)
        self._context._graph.bfs(evaluator)

        # Mark suite as evaluated only after successful completion
        self._is_evaluated = True

        # 4. Process results through plugins if enabled
        if enable_plugins:
            self._process_plugins(datasources)

    def collect_results(self) -> list[AssertionResult]:
        """
        Collect all assertion results produced by the most recent run of the suite.

        Traverses the suite's evaluation graph and returns a list of AssertionResult objects—one per assertion—using the ResultKey from the last run. The returned list is cached and subsequent calls return the same list object.

        Returns:
            List[AssertionResult]: AssertionResult instances for each assertion in graph traversal order.

        Raises:
            DQXError: If the suite has not been evaluated (run) yet.
        """
        # Only collect results after evaluation
        self.assert_is_evaluated()

        # Return cached results if available
        if self._cached_results is not None:
            return self._cached_results

        key = self.key
        results = []

        # Use the graph's built-in method to get all assertions
        for assertion in self.graph.assertions():
            # Extract parent hierarchy
            check_node = assertion.parent  # Parent is always a CheckNode

            # Use effective severity (profile override) if set, otherwise original
            effective_severity = assertion._effective_severity or assertion.severity

            # Build expression, substituting tunable names with their actual values
            # Step 1: Substitute TunableSymbols in the actual expression
            from dqx.tunables import TunableSymbol

            actual_expr = assertion.actual
            tunable_symbols = actual_expr.atoms(TunableSymbol)
            if tunable_symbols:
                # Create substitution dict: TunableSymbol -> numeric value
                subs_dict = {ts: ts.value for ts in tunable_symbols}
                actual_expr = actual_expr.subs(subs_dict)

            # Step 2: Substitute tunable names in the validator
            validator_name = assertion.validator.name
            for tunable_name, tunable in assertion.tunables.items():
                validator_name = validator_name.replace(tunable_name, str(tunable.value))

            # Step 3: Build final expression string
            expression_str = f"{actual_expr} {validator_name}"

            result = AssertionResult(
                yyyy_mm_dd=key.yyyy_mm_dd,
                suite=self._name,
                check=check_node.name,
                assertion=assertion.name,
                severity=effective_severity,
                status=assertion._result,
                metric=assertion._metric,
                expression=expression_str,
                tags=key.tags,
                assertion_tags=assertion.tags,
                experimental=assertion.experimental,
                required=assertion.required,
                cost_fp=assertion.cost_fp,
                cost_fn=assertion.cost_fn,
            )
            results.append(result)

        # Cache the results
        self._cached_results = results
        return self._cached_results

    def is_critical(self) -> bool:
        """
        Determine if the suite has critical failures (P0 severity).

        A suite is considered critical if it contains at least one assertion
        with P0 severity that has failed.

        This method uses the cached collect_results() for efficiency.

        Returns:
            True if any P0 assertion failed, False otherwise.

        Raises:
            DQXError: If called before run() has been executed successfully.

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> datasources = [DuckRelationDataSource.from_arrow(data, "my_data")]
            >>> suite.run(datasources, key)
            >>> if suite.is_critical():
            ...     print("CRITICAL: P0 failures detected!")
            ...     send_alert()
        """
        # Ensure suite has been evaluated
        self.assert_is_evaluated()

        # Use collect_results to get all assertion results
        # This will use cached results if available
        results = self.collect_results()

        # Check if any P0 assertion has failed
        for result in results:
            if result.severity == "P0" and result.status == "FAILED":
                return True

        return False

    def metric_trace(self, db: "MetricDB") -> pa.Table:
        """
        Generate a metric trace table showing how metrics flow through the system.

        This method creates a trace showing metric values from the database,
        through analysis reports, to final symbol values. It performs joins
        to match metrics across these different stages.

        Args:
            db: The MetricDB instance to query for persisted metrics

        Returns:
            PyArrow table with columns: date, metric, symbol, type, dataset,
            value_db, value_analysis, value_final, error, tags, is_extended

        Raises:
            DQXError: If called before run() has been executed

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> datasources = [DuckRelationDataSource.from_arrow(data, "my_data")]
            >>> suite.run(datasources, key)
            >>> trace = suite.metric_trace(db)
            >>> # Analyze the trace
            >>> from dqx.data import metric_trace_stats
            >>> stats = metric_trace_stats(trace)
            >>> print(f"Found {stats.discrepancy_count} discrepancies")
        """
        from dqx import data

        # Ensure suite has been evaluated
        self.assert_is_evaluated()

        # Get metrics from database for this execution
        metrics = db.get_by_execution_id(self.execution_id)

        # Get symbols from the provider
        symbols = self.provider.collect_symbols(self.key)

        # Generate and return the trace table
        return data.metric_trace(
            metrics=metrics,
            execution_id=self.execution_id,
            reports=self._analysis_reports,
            symbols=symbols,
            symbol_lookup=self.provider.registry.symbol_lookup_table(self.key),
        )

    def _process_plugins(self, datasources: list[SqlDataSource]) -> None:
        """
        Process results through all loaded plugins.

        Args:
            datasources: List of data sources used in the suite execution
        """
        # Raise error if the suite hasn't been properly executed
        self.assert_is_evaluated()
        duration_ms = self._analyze_ms.elapsed_ms()

        # Create plugin execution context
        context = PluginExecutionContext(
            suite_name=self._name,
            execution_id=self.execution_id,
            datasources=[ds.name for ds in datasources],
            key=self.key,
            timestamp=self._context.start_time,
            duration_ms=duration_ms,
            results=self.collect_results(),
            symbols=self.provider.collect_symbols(self.key),
            trace=self.metric_trace(self.provider.db),
            metrics_stats=self.metrics_stats,
            cache_stats=self.provider.cache.get_stats(),
        )

        # Process through all plugins
        self.plugin_manager.process_all(context)

    def cleanup_expired_metrics(self) -> None:
        """
        Delete expired metrics from the database.

        This method should be called periodically to remove metrics that have
        exceeded their TTL (time-to-live). It uses UTC time for consistency
        across different timezones.

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> # After running the suite...
            >>> suite.cleanup_expired_metrics()
            >>> print("Expired metrics deleted")
        """
        self.provider.db.delete_expired_metrics()


def _create_check(
    provider: MetricProvider,
    context: Context,
    _check: CheckProducer,
    name: str,
    datasets: list[str] | None = None,
) -> None:
    """
    Internal function to create and register a check node in the context graph.

    Args:
        provider: Metric provider for the check
        context: Execution context
        _check: Check function to execute
        name: Name for the check
        datasets: Optional list of datasets the check applies to

    Raises:
        DQXError: If a check with the same name already exists
    """
    # Create the check node using root's factory method
    # This will automatically add it to the root and set the parent
    node = context._graph.root.add_check(name=name, datasets=datasets)

    # Call the symbolic check to collect assertions for this check node
    with context.check_context(node):
        _check(provider, context)


def check(
    *,
    name: str,
    datasets: list[str] | None = None,
) -> Callable[[CheckProducer], DecoratedCheck]:
    """
    Decorator for creating data quality check functions.

    Must be used with parentheses and a name:

    @check(name="Important Check", datasets=["ds1"])
    def my_labeled_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    Args:
        name: Human-readable name for the check (required)
        datasets: Optional list of datasets the check applies to

    Returns:
        Decorated check function

    Raises:
        TypeError: If called without the required 'name' parameter
    """

    def decorator(fn: CheckProducer) -> DecoratedCheck:
        """Wrap the check function with metadata and return a decorated check."""
        wrapped = functools.wraps(fn)(functools.partial(_create_check, _check=fn, name=name, datasets=datasets))
        # No metadata storage needed anymore
        return cast(DecoratedCheck, wrapped)

    return decorator
