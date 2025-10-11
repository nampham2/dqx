from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, Literal, Protocol, Self, cast, runtime_checkable

import sympy as sp

from dqx import functions, get_logger
from dqx.analyzer import Analyzer
from dqx.common import DQXError, ResultKey, ResultKeyProvider, SeverityLevel, SqlDataSource, SymbolicValidator
from dqx.evaluator import Evaluator
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.graph.traversal import Graph
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider
from dqx.specs import MetricSpec
from dqx.validator import SuiteValidator, ValidationReport

CheckProducer = Callable[[MetricProvider, "Context"], None]
CheckCreator = Callable[[CheckProducer], CheckProducer]


logger = get_logger(__name__)


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
        Initialize assertion draft.

        Args:
            actual: The symbolic expression to evaluate
            context: The Context instance (needed to create assertion nodes)
        """
        self._actual = actual
        self._context = context

    def where(self, *, name: str, severity: SeverityLevel = "P1") -> AssertionReady:
        """
        Provide a descriptive name for this assertion.

        Args:
            name: Required description of what this assertion validates
            severity: Severity level (P0, P1, P2, P3). Defaults to "P1".
                     All assertions must have a severity level.

        Returns:
            AssertionReady instance with all assertion methods available

        Raises:
            ValueError: If name is empty or too long
        """
        if not name or not name.strip():
            raise ValueError("Assertion name cannot be empty")
        if len(name) > 255:
            raise ValueError("Assertion name is too long (max 255 characters)")

        return AssertionReady(actual=self._actual, name=name.strip(), severity=severity, context=self._context)


class AssertionReady:
    """
    Named assertion ready to perform validations.

    This assertion has been properly named and can now use any of the
    validation methods like is_gt(), is_eq(), etc.
    """

    def __init__(
        self, actual: sp.Expr, name: str, severity: SeverityLevel = "P1", context: Context | None = None
    ) -> None:
        """
        Initialize ready assertion.

        Args:
            actual: The symbolic expression to evaluate
            name: Required description of the assertion
            severity: Severity level (P0, P1, P2, P3). Defaults to "P1".
            context: The Context instance
        """
        self._actual = actual
        self._name = name
        self._severity = severity
        self._context = context

    def is_geq(self, other: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is greater than or equal to the given value."""
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
        """Assert that the expression equals the given value within tolerance."""
        validator = SymbolicValidator(f"= {other}", lambda x: functions.is_eq(x, other, tol))
        self._create_assertion_node(validator)

    def is_negative(self, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is negative."""
        validator = SymbolicValidator("< 0", lambda x: functions.is_negative(x, tol))
        self._create_assertion_node(validator)

    def is_positive(self, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is positive."""
        validator = SymbolicValidator("> 0", lambda x: functions.is_positive(x, tol))
        self._create_assertion_node(validator)

    def _create_assertion_node(self, validator: SymbolicValidator) -> None:
        """Create a new assertion node and attach it to the current check."""
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
            validator=validator,
        )


@runtime_checkable
class DecoratedCheck(Protocol):
    """Protocol for check functions."""

    __name__: str

    def __call__(self, mp: MetricProvider, ctx: "Context") -> None: ...


# Graph node state types
GraphState = Literal["PENDING", "SUCCESS", "FAILED"]


class Context:
    """
    Execution context for data quality checks containing the dependency graph and assertion utilities.

    Provides methods to create assertions and manage the verification workflow.
    The Context owns the SymbolTable and provides factory methods for creating
    graph nodes that need access to the symbol table.
    """

    def __init__(self, suite: str, db: MetricDB) -> None:
        """
        Initialize the context with a root graph node.

        Args:
            suite: Name of the verification suite
        """
        self._graph = Graph(RootNode(name=suite))
        self._provider = MetricProvider(db)
        self._local = threading.local()

    @property
    def _check_stack(self) -> list[CheckNode]:
        if not hasattr(self._local, "check_stack"):
            self._local.check_stack = []
        return self._local.check_stack

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
    def key(self) -> ResultKeyProvider:
        """Get a result key provider for creating time-based metric keys."""
        return ResultKeyProvider()

    @property
    def provider(self) -> MetricProvider:
        return self._provider

    def create_check(
        self,
        name: str,
        tags: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> CheckNode:
        """
        Factory method to create a check node.

        Args:
            name: Name for the check (either user-provided or function name)
            tags: Optional tags for categorizing the check
            datasets: Optional list of datasets the check applies to

        Returns:
            CheckNode that can access context through its root node
        """
        # Use the root node's factory method
        return self._graph.root.add_check(name=name, tags=tags, datasets=datasets)

    def create_assertion(
        self,
        actual: sp.Expr,
        name: str | None = None,
        severity: SeverityLevel = "P1",
        validator: SymbolicValidator | None = None,
    ) -> AssertionNode:
        """
        Factory method to create an assertion node.

        Args:
            actual: Symbolic expression to evaluate
            name: Optional human-readable description
            severity: Severity level for failures (P0, P1, P2, P3). Defaults to "P1".
            validator: Optional validator function

        Returns:
            AssertionNode that can access context through its root node
        """
        # Get the current check node
        current = self.current_check
        if not current:
            raise DQXError(
                "Cannot create assertion outside of check context. "
                "Assertions must be created within a @check decorated function."
            )

        # Use the check node's factory method
        return current.add_assertion(actual=actual, name=name, severity=severity, validator=validator)

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

    def pending_metrics(self, dataset: str | None = None) -> Sequence[MetricSpec]:
        """
        Get pending metrics for the specified dataset or all datasets if none specified.

        Args:
            dataset: Optional dataset name. If None, returns metrics for all datasets.

        Returns:
            Sequence of pending metric specifications

        TODO: Read the database to filter out metrics based on TTL
        """
        # Get metrics from symbol table
        all_metrics = self.provider.symbolic_metrics
        if dataset:
            return [metric.metric_spec for metric in all_metrics if metric.dataset == dataset]
        return [metric.metric_spec for metric in all_metrics]


class VerificationSuite:
    """
    A suite of data quality verification checks that can be executed against multiple data sources.

    The suite collects symbolic assertions through check functions and builds a dependency graph
    of metrics, symbols, and analyzers required to evaluate those assertions.

    Example:
        >>> db = MetricDB()
        >>> suite = VerificationSuite([my_check], db, "My Suite")
        >>> result = suite.run({"dataset": datasource}, key)
    """

    def __init__(
        self,
        checks: Sequence[CheckProducer | DecoratedCheck],
        db: MetricDB,
        name: str,
    ) -> None:
        """
        Initialize the verification suite.

        Args:
            checks: Sequence of check functions to execute
            db: Database for storing and retrieving metrics
            name: Human-readable name for the suite

        Raises:
            DQXError: If no checks provided or name is empty
        """
        if not checks:
            raise DQXError("At least one check must be provided")
        if not name.strip():
            raise DQXError("Suite name cannot be empty")

        self._checks: Sequence[CheckProducer | DecoratedCheck] = checks
        self._name = name.strip()

        # Create a context
        self._context = Context(suite=self._name, db=db)

        # Create validator instance
        self._validator = SuiteValidator()

    @property
    def provider(self) -> MetricProvider:
        """
        The metric provider instance used by the verification suite.

        This property returns the MetricProvider instance that is used by the verification suite to access and manage metrics.

        Returns:
            MetricProvider instance used by the verification suite
        """
        return self._context.provider

    def validate(self) -> ValidationReport:
        """
        Explicitly validate the suite configuration.

        Returns:
            ValidationReport containing any issues found
        """
        # Create temporary context to collect checks
        temp_context = Context(suite=self._name, db=self.provider._db)

        # Execute all checks to build graph
        for check_fn in self._checks:
            check_fn(self.provider, temp_context)

        # Run validation on the graph
        return self._validator.validate(temp_context._graph, temp_context.provider)

    def collect(self, context: Context, key: ResultKey) -> None:
        """
        Collect all checks and build the dependency graph without executing analysis.

        Args:
            key: The result key defining the time period and tags for analysis

        Returns:
            Context containing the collected checks and dependency graph

        Raises:
            DQXError: If check collection fails or duplicate checks are found
        """
        # Execute all checks to collect assertions
        for check in self._checks:
            check(self.provider, context)

        # Run validation
        report = self._validator.validate(context._graph, context.provider)

        # Only raise on errors, log warnings
        if report.has_errors():
            raise DQXError(f"Suite validation failed:\n{report}")
        elif report.has_warnings():
            logger.warning(f"Suite validation warnings:\n{report}")

    def run(self, datasources: dict[str, SqlDataSource], key: ResultKey, threading: bool = False) -> None:
        """
        Execute the verification suite against the provided data sources.

        Args:
            datasources: Dictionary mapping dataset names to data sources
            key: Result key defining the time period and tags
            threading: Whether to use threading for analysis

        Returns:
            Context containing the execution results

        Raises:
            DQXError: If no data sources provided
        """
        logger.info(f"Running verification suite '{self._name}' with datasets: {list(datasources.keys())}")

        # Validate the datasources
        if not datasources:
            raise DQXError("No data sources provided!")

        # Build the dependency graph
        logger.info("Collecting checks and building dependency graph...")
        self.collect(self._context, key)

        # 1. Impute datasets using visitor pattern
        logger.info("Imputing datasets...")
        self._context._graph.impute_datasets(list(datasources.keys()), self._context.provider)

        # 2. Analyze by datasources
        for ds in datasources.keys():
            analyzer = Analyzer()
            metrics = self._context.pending_metrics(ds)
            # TODO: Check the metrics and logging
            analyzer.analyze(datasources[ds], metrics, key, threading=threading)
            analyzer.persist(self.provider._db)

        # 3. Evaluate assertions
        evaluator = Evaluator(self.provider, key)
        self._context._graph.bfs(evaluator)


class VerificationSuiteBuilder:
    """
    Builder pattern for creating VerificationSuite instances with fluent configuration.

    Example:
        >>> builder = VerificationSuiteBuilder("My Suite", db)
        >>> suite = builder.add_check(check1).add_checks([check2, check3]).build()
    """

    def __init__(self, name: str, db: MetricDB) -> None:
        """
        Initialize the builder.

        Args:
            name: Name for the verification suite
            db: Database for metrics storage
        """
        self._name = name
        self._db = db
        self._checks: list[CheckProducer | DecoratedCheck] = []

    def add_check(self, check: CheckProducer | DecoratedCheck) -> Self:
        """Add a single check to the suite."""
        self._checks.append(check)
        return self

    def add_checks(self, checks: Sequence[CheckProducer | DecoratedCheck]) -> Self:
        """Add multiple checks to the suite."""
        self._checks.extend(checks)
        return self

    def build(self) -> VerificationSuite:
        """Build and return the configured VerificationSuite."""
        return VerificationSuite(self._checks, self._db, self._name)


def _create_check(
    provider: MetricProvider,
    context: Context,
    _check: CheckProducer,
    name: str,
    tags: list[str] = [],
    datasets: list[str] | None = None,
) -> None:
    """
    Internal function to create and register a check node in the context graph.

    Args:
        provider: Metric provider for the check
        context: Execution context
        _check: Check function to execute
        tags: Optional tags for the check
        display_name: Optional human-readable name
        datasets: Optional list of datasets the check applies to

    Raises:
        DQXError: If a check with the same name already exists
    """
    # Create the check node using root's factory method
    # This will automatically add it to the root and set the parent
    node = context._graph.root.add_check(name=name, tags=tags, datasets=datasets)

    # Call the symbolic check to collect assertions for this check node
    with context.check_context(node):
        _check(provider, context)


def check(
    *,
    name: str,
    tags: list[str] = [],
    datasets: list[str] | None = None,
) -> Callable[[CheckProducer], DecoratedCheck]:
    """
    Decorator for creating data quality check functions.

    Must be used with parentheses and a name:

    @check(name="Important Check", tags=["critical"], datasets=["ds1"])
    def my_labeled_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    Args:
        name: Human-readable name for the check (required)
        tags: Optional tags for categorizing the check
        datasets: Optional list of datasets the check applies to

    Returns:
        Decorated check function

    Raises:
        TypeError: If called without the required 'name' parameter
    """

    def decorator(fn: CheckProducer) -> DecoratedCheck:
        wrapped = functools.wraps(fn)(
            functools.partial(_create_check, _check=fn, name=name, tags=tags, datasets=datasets)
        )
        # No metadata storage needed anymore
        return cast(DecoratedCheck, wrapped)

    return decorator
