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
)
from dqx.evaluator import Evaluator
from dqx.graph.nodes import CheckNode, RootNode
from dqx.graph.traversal import Graph

# import moved to local scope(s) to avoid cyclic dependency
from dqx.plugins import PluginExecutionContext, PluginManager
from dqx.provider import MetricProvider, SymbolicMetric
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

    def is_between(self, lower: float, upper: float, tol: float = functions.EPSILON) -> None:
        """Assert that the expression is between two values (inclusive)."""
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
        """Assert that the expression is positive."""
        validator = SymbolicValidator("> 0", lambda x: functions.is_positive(x, tol))
        self._create_assertion_node(validator)

    def noop(self) -> None:
        """Assert that does nothing - only collects the metric value."""
        validator = SymbolicValidator("", lambda x: True)
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


class Context:
    """
    Execution context for data quality checks containing the dependency graph and assertion utilities.

    Provides methods to create assertions and manage the verification workflow.
    The Context owns the SymbolTable and provides factory methods for creating
    graph nodes that need access to the symbol table.
    """

    def __init__(self, suite: str, db: "MetricDB", execution_id: str) -> None:
        """
        Initialize the context with a root graph node.

        Args:
            suite: Name of the verification suite
            db: Database for storing and retrieving metrics
            execution_id: Unique identifier for this execution
        """
        self._graph = Graph(RootNode(name=suite))
        # MetricProvider now creates its own cache internally
        self._provider = MetricProvider(db, execution_id=execution_id)
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
        >>> db = MetricDB()
        >>> suite = VerificationSuite([my_check], db, "My Suite")
        >>> datasource = DuckRelationDataSource.from_arrow(data, "dataset")
        >>> result = suite.run([datasource], key)
    """

    def __init__(
        self,
        checks: Sequence[CheckProducer | DecoratedCheck],
        db: "MetricDB",
        name: str,
        log_level: int = logging.INFO,
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

        # Create a context with execution_id
        self._context = Context(suite=self._name, db=db, execution_id=self._execution_id)

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
        Access the analysis reports generated by the suite.

        This property provides read-only access to the analysis reports
        generated for each datasource after the suite has been run.

        Returns:
            dict[str, AnalysisReport]: Mapping of datasource names to their analysis reports

        Raises:
            DQXError: If accessed before the suite has been run
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
        Get the cached metrics statistics.

        Returns:
            MetricStats instance containing total and expired metric counts

        Raises:
            DQXError: If accessed before the suite has been run
        """
        self.assert_is_evaluated()
        if self._metrics_stats is None:
            raise DQXError("Metrics stats not available. This should not happen after successful run().")
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
        Return the ResultKey used during the last run() call.

        The ResultKey stores information about the time period and tags used
        during the verification suite execution.

        Returns:
            ResultKey instance used during the last run() call

        Raises:
            DQXError: If called before run() has been executed successfully
        """
        if self._key is None:
            raise DQXError("No ResultKey available. This should not happen after successful run().")
        return self._key

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

    def build_graph(self, context: Context, key: ResultKey) -> None:
        """
        Build the dependency graph by executing all checks without running analysis.

        This method:
        1. Executes all check functions to populate the graph with assertions
        2. Validates the graph structure for errors or warnings
        3. Raises DQXError if validation fails

        Args:
            context: The execution context containing the graph
            key: The result key defining the time period and tags

        Raises:
            DQXError: If validation fails or duplicate checks are found

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> key = ResultKey(date.today(), {"env": "prod"})
            >>> suite.build_graph(suite._context, key)
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
        analyzer = Analyzer(datasources, self.provider, key, execution_id=self._execution_id)
        self._analysis_reports = analyzer.analyze()

    def run(self, datasources: list[SqlDataSource], key: ResultKey, *, enable_plugins: bool = True) -> None:
        """
        Execute the verification suite against the provided data sources.

        Args:
            datasources: List of data sources to analyze
            key: Result key defining the time period and tags
            enable_plugins: Whether to execute plugins after validation (default True)

        Returns:
            Context containing the execution results

        Raises:
            DQXError: If no data sources provided or suite already executed
        """

        # Prevent multiple runs
        if self.is_evaluated:
            raise DQXError("Verification suite has already been executed. Create a new suite instance to run again.")

        # Validate the datasources
        if not datasources:
            raise DQXError("No data sources provided!")

        logger.info(f"Running verification suite '{self._name}' with datasets: {[ds.name for ds in datasources]}")
        logger.info("Execution id: %s", self.execution_id)

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

        # Collect metrics stats and cleanup expired metrics BEFORE analysis
        self._metrics_stats = self.provider._db.get_metrics_stats()
        logger.info(
            f"Metrics stats: {self._metrics_stats.expired_metrics} expired "
            f"out of {self._metrics_stats.total_metrics} total"
        )

        # Cleanup expired metrics before analysis
        if self._metrics_stats.expired_metrics > 0:
            logger.info("Cleaning up expired metrics...")
            self.cleanup_expired_metrics()

        # 2. Analyze by datasources
        with self._analyze_ms:
            self._analyze(datasources, key)

        # 3. Evaluate assertions
        # Use graph in the context to avoid the check if the suite has been evaluated
        evaluator = Evaluator(self.provider, key, self._name)
        self._context._graph.bfs(evaluator)

        # Mark suite as evaluated only after successful completion
        self._is_evaluated = True

        # 4. Process results through plugins if enabled
        if enable_plugins:
            self._process_plugins(datasources)

    def collect_results(self) -> list[AssertionResult]:
        """
        Collect all assertion results after suite execution.

        This method traverses the evaluation graph and extracts results from
        all assertions, converting them into AssertionResult objects suitable
        for persistence or reporting. The ResultKey used during run() is
        automatically applied to all results.

        Results are cached after the first call, so subsequent calls return
        the same object reference for efficiency.

        Returns:
            List of AssertionResult instances, one for each assertion in the suite.
            Results are returned in graph traversal order (breadth-first).

        Raises:
            DQXError: If called before run() has been executed successfully.

        Example:
            >>> suite = VerificationSuite(checks, db, "My Suite")
            >>> datasources = [DuckRelationDataSource.from_arrow(data, "my_data")]
            >>> suite.run(datasources, key)
            >>> results = suite.collect_results()  # No key needed!
            >>> for r in results:
            ...     print(f"{r.check}/{r.assertion}: {r.status}")
            ...     if r.status == "FAILURE":
            ...         failures = r.value.failure()
            ...         for f in failures:
            ...             print(f"  Error: {f.error_message}")
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

            result = AssertionResult(
                yyyy_mm_dd=key.yyyy_mm_dd,
                suite=self._name,
                check=check_node.name,
                assertion=assertion.name,
                severity=assertion.severity,
                status=assertion._result,
                metric=assertion._metric,
                expression=f"{assertion.actual} {assertion.validator.name}",
                tags=key.tags,
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
            if result.severity == "P0" and result.status == "FAILURE":
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
            trace=self.metric_trace(self.provider._db),
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
        self.provider._db.delete_expired_metrics()


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
