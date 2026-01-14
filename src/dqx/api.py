from __future__ import annotations

import functools
import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

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
from dqx.tunables import Tunable, TunableChange
from dqx.timer import Registry
from dqx.validator import SuiteValidator

if TYPE_CHECKING:
    from dqx.orm.repositories import MetricDB, MetricStats

CheckProducer = Callable[[MetricProvider, "Context"], None]
CheckCreator = Callable[[CheckProducer], CheckProducer]


logger = logging.getLogger(__name__)
timer_registry = Registry()


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

    def is_geq(self, other: float, tol: float = functions.EPSILON) -> None:
        """
        Create an assertion that the expression is greater than or equal to the specified threshold.

        Args:
            other: Threshold value to compare the expression against.
            tol: Comparison tolerance; values within `tol` of `other` are treated as equal.
        """
        validator = SymbolicValidator(f"≥ {other}", lambda x: functions.is_geq(x, other, tol))
        self._create_assertion_node(validator)

    def is_gt(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is greater than the given value."""
        validator = SymbolicValidator(f"> {other}", lambda x: functions.is_gt(x, other, tol))
        self._create_assertion_node(validator)

    def is_leq(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is less than or equal to the given value."""
        validator = SymbolicValidator(f"≤ {other}", lambda x: functions.is_leq(x, other, tol))
        self._create_assertion_node(validator)

    def is_lt(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is less than the given value."""
        validator = SymbolicValidator(f"< {other}", lambda x: functions.is_lt(x, other, tol))
        self._create_assertion_node(validator)

    def is_eq(self, other: float, tol: float = functions.EPSILON) -> None:
        """
        Assert that the expression equals the given value within tolerance.

        Args:
            other: Target value to compare the expression against.
            tol: Absolute tolerance for the comparison; defaults to functions.EPSILON.
        """
        validator = SymbolicValidator(f"= {other}", lambda x: functions.is_eq(x, other, tol))
        self._create_assertion_node(validator)

    def is_neq(self, other: float, tol: float = functions.EPSILON) -> None:
        """
        Assert that the expression is not equal to a specified value, allowing for a tolerance.

        Args:
            other: The value to compare against.
            tol: Allowed tolerance; values within `tol` of `other` are considered equal.
        """
        validator = SymbolicValidator(f"≠ {other}", lambda x: functions.is_neq(x, other, tol))
        self._create_assertion_node(validator)

    def is_between(self, lower: float, upper: float, tol: float = functions.EPSILON) -> None:
        """
        Assert that the expression lies within the inclusive interval [lower, upper].

        Args:
            lower: Lower bound of the allowed interval.
            upper: Upper bound of the allowed interval.
            tol: Numeric tolerance applied to the comparison; values within `tol` of a boundary are considered inside.

        Raises:
            ValueError: If `lower` is greater than `upper`.
        """
        if lower > upper:
            raise ValueError(
                f"Invalid range: lower bound ({lower}) must be less than or equal to upper bound ({upper})"
            )

        validator = SymbolicValidator(f"in [{lower}, {upper}]", lambda x: functions.is_between(x, lower, upper, tol))
        self._create_assertion_node(validator)

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
        Create an assertion that the expression equals zero.

        Args:
            tol: Comparison tolerance; values with absolute value less than `tol` are considered zero.
        """
        validator = SymbolicValidator("== 0", lambda x: functions.is_zero(x, tol))
        self._create_assertion_node(validator)

    def is_none(self) -> None:
        """
        Create an assertion that the expression is None.
        """
        validator = SymbolicValidator("is None", lambda x: x is None)
        self._create_assertion_node(validator)

    def is_not_none(self) -> None:
        """Assert that the expression does not evaluate to None."""
        validator = SymbolicValidator("is not None", lambda x: x is not None)
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
    of metrics, symbols, and analyzers required to evaluate those assertions.

    Example:
        Basic usage (single run):

        >>> db = MetricDB()
        >>> suite = VerificationSuite([my_check], db, "My Suite")
        >>> datasource = DuckRelationDataSource.from_arrow(data, "dataset")
        >>> result = suite.run([datasource], key)

        Advanced usage with tunables and reset (for AI agents):

        >>> from dqx.tunables import TunablePercent
        >>> threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.50))
        >>> suite = VerificationSuite([my_check], db, "My Suite", tunables=[threshold])
        >>>
        >>> # Run 1
        >>> suite.run([datasource], key)
        >>> results = suite.collect_results()
        >>>
        >>> # Adjust threshold based on results
        >>> suite.set_param("THRESHOLD", 0.30, agent="rl_optimizer", reason="Tuning iteration")
        >>> suite.reset()  # Clear state for next run
        >>>
        >>> # Run 2 with new threshold
        >>> suite.run([datasource], key)
        >>> updated_results = suite.collect_results()
    """

    def __init__(
        self,
        checks: Sequence[CheckProducer | DecoratedCheck],
        db: "MetricDB",
        name: str,
        log_level: int | str = logging.INFO,
        data_av_threshold: float = 0.9,
        profiles: Sequence[Profile] | None = None,
        tunables: Sequence["Tunable"] | None = None,
    ) -> None:
        """
        Initialize a VerificationSuite that orchestrates and evaluates a set of data quality checks.

        Args:
            checks: Sequence of check callables to execute; each will be invoked to populate the suite's verification graph.
            db: Storage backend for producing and retrieving metrics used by checks and analysis.
            name: Human-readable name for the suite; must be non-empty.
            log_level: Logging level for the suite (default: logging.INFO).
            data_av_threshold: Minimum fraction of available data required to evaluate assertions (default: 0.9).
            profiles: Optional profiles that alter assertion evaluation behavior.
            tunables: Optional tunable parameters exposed for external agents; names must be unique.

        Raises:
            DQXError: If no checks are provided, the suite name is empty, or duplicate tunable names are supplied.
        """
        # Setting up the logger
        setup_logger(level=log_level)

        if not checks:
            raise DQXError("At least one check must be provided")
        if not name.strip():
            raise DQXError("Suite name cannot be empty")

        self._checks: Sequence[CheckProducer | DecoratedCheck] = checks
        self._name = name.strip()

        # Generate unique execution ID
        self._execution_id = str(uuid.uuid4())

        # Store data availability threshold
        self._data_av_threshold = data_av_threshold

        # Create a context with execution_id and data availability threshold
        self._context = Context(
            suite=self._name, db=db, execution_id=self._execution_id, data_av_threshold=self._data_av_threshold
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

        # Store profiles for evaluation
        self._profiles: list[Profile] = list(profiles) if profiles else []

        # Store tunables for RL agent integration
        self._tunables: dict[str, Tunable] = {}
        if tunables:
            for t in tunables:
                if t.name in self._tunables:
                    raise DQXError(f"Duplicate tunable name: {t.name}")
                self._tunables[t.name] = t

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

        This property provides read-only access to the internal Graph instance
        after the graph has been built via build_graph() or run().

        Returns:
            Graph: The dependency graph containing checks and assertions

        Raises:
            DQXError: If accessed before the graph has been built
                     (i.e., before build_graph() or run() has been called)

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> datasources = [DuckRelationDataSource.from_arrow(data, "my_data")]
            >>> suite.run(datasources, key)
            >>> graph = suite.graph  # Now accessible
            >>> print(f"Graph has {len(list(graph.checks()))} checks")
        """
        self.assert_is_evaluated()
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

        This method clears all execution state (results, analysis reports, graph) while
        preserving the suite configuration and tunable values. Each reset generates a
        new execution_id to distinguish tuning iterations in the metrics database.

        Primary use case: AI agents tuning threshold parameters via Tunables.

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite", tunables=[threshold])
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
            - Clears cached results, analysis reports, and dependency graph
            - Clears plugin manager (will be lazy-loaded on next use)
        """
        # Generate new execution_id for the next run
        self._execution_id = str(uuid.uuid4())

        # Create fresh context with new execution_id
        # Preserve the db reference from the old context's provider
        self._context = Context(
            suite=self._name,
            db=self._context.provider.db,
            execution_id=self._execution_id,
            data_av_threshold=self._data_av_threshold,
        )

        # Clear execution state
        self._is_evaluated = False
        self._key = None
        self._cached_results = None
        self._analysis_reports = None  # type: ignore[assignment]
        self._metrics_stats = None

        # Clear plugin manager (will be lazy-loaded on next use)
        self._plugin_manager = None

    def build_graph(self, context: Context, key: ResultKey) -> None:
        """
        Populate the execution graph by running all registered checks and validate it.

        Runs each check to add nodes and assertions into the provided Context's graph, then validates the assembled graph using SuiteValidator. If validation reports errors a DQXError is raised; validation warnings are emitted to the logger.

        Args:
            context: Execution context that holds the graph and provider.
            key: Result key identifying the run (reserved for future use).

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

        # Store the key for later use in collect_results
        self._key = key

        # Reset the run timer
        self._context.tick()

        # Build the dependency graph
        logger.info("Building dependency graph...")
        self.build_graph(self._context, key)

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

            result = AssertionResult(
                yyyy_mm_dd=key.yyyy_mm_dd,
                suite=self._name,
                check=check_node.name,
                assertion=assertion.name,
                severity=effective_severity,
                status=assertion._result,
                metric=assertion._metric,
                expression=f"{assertion.actual} {assertion.validator.name}",
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
        wrapped = functools.wraps(fn)(functools.partial(_create_check, _check=fn, name=name, datasets=datasets))
        # No metadata storage needed anymore
        return cast(DecoratedCheck, wrapped)

    return decorator
