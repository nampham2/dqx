from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from typing import Protocol, Self, overload, runtime_checkable, TypedDict, cast

import sympy as sp
from returns.maybe import Some

from dqx import common, functions, graph
from dqx.analyzer import Analyzer
from dqx.common import DQXError, SqlDataSource, ResultKey, ResultKeyProvider, SeverityLevel, SymbolicValidator
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import MetricSpec
from dqx.symbol_table import SymbolTable

CheckProducer = Callable[[MetricProvider, common.Context], None]
CheckCreator = Callable[[CheckProducer], CheckProducer]


class CheckMetadata(TypedDict):
    """Metadata stored on decorated check functions."""

    name: str
    datasets: list[str] | None
    tags: list[str]
    label: str | None


@runtime_checkable
class DecoratedCheck(Protocol):
    """Protocol for check functions with metadata."""

    __name__: str
    _check_metadata: CheckMetadata

    def __call__(self, mp: MetricProvider, ctx: common.Context) -> None: ...


class GraphStates:
    """Constants for graph node states."""

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@runtime_checkable
class AssertListener(Protocol):
    """Protocol for objects that listen to assertion configuration changes."""

    def set_label(self, label: str) -> None: ...
    def set_severity(self, severity: SeverityLevel) -> None: ...
    def set_validator(self, validator: SymbolicValidator) -> None: ...


class SymbolicAssert:
    """
    A symbolic assertion that can be configured with validators and evaluated against data.

    Provides a fluent interface for setting up data quality assertions with various
    comparison operators and tolerance levels.
    """

    def __init__(self, actual: sp.Expr, listeners: list[AssertListener], context: Context | None = None) -> None:
        self._actual = actual
        self._label: str | None = None
        self._severity: SeverityLevel | None = None
        self._validator: SymbolicValidator | None = None
        self.listeners = listeners
        self._context = context

    def on(self, *, label: str | None = None, severity: SeverityLevel | None = None) -> Self:
        """
        Configure the assertion with optional label and severity.

        Args:
            label: Human-readable description of the assertion
            severity: Severity level for assertion failures

        Returns:
            Self for method chaining
        """
        self._label = label
        self._severity = severity

        for listener in self.listeners:
            if severity:
                listener.set_severity(severity)
            if label:
                listener.set_label(label)

        return self

    def _update_validator(self, validator: SymbolicValidator) -> None:
        """Update the validator and notify all listeners."""
        self._validator = validator

        # Update listeners
        for listener in self.listeners:
            listener.set_validator(validator)

    def _create_validator(
        self,
        name: str,
        comparison_fn: Callable[[float, float, float], bool],
        other: float,
        tol: float = functions.EPSILON,
    ) -> None:
        """Create a validator with the given comparison function and parameters."""
        self._update_validator(SymbolicValidator(name=name, fn=lambda a: comparison_fn(a, other, tol)))

    def is_geq(self, other: float, tol: float = functions.EPSILON) -> SymbolicAssert:
        """Assert that the expression is greater than or equal to the given value."""
        self._create_validator(f"\u2265 {other}", functions.is_geq, other, tol)
        return self._create_new_assertion_if_needed()

    def is_gt(self, other: float, tol: float = functions.EPSILON) -> SymbolicAssert:
        """Assert that the expression is greater than the given value."""
        self._create_validator(f"> {other}", functions.is_gt, other, tol)
        return self._create_new_assertion_if_needed()

    def is_leq(self, other: float, tol: float = functions.EPSILON) -> SymbolicAssert:
        """Assert that the expression is less than or equal to the given value."""
        self._create_validator(f"\u2264 {other}", functions.is_leq, other, tol)
        return self._create_new_assertion_if_needed()

    def is_lt(self, other: float, tol: float = functions.EPSILON) -> SymbolicAssert:
        """Assert that the expression is less than the given value."""
        self._create_validator(f"< {other}", functions.is_lt, other, tol)
        return self._create_new_assertion_if_needed()

    def is_eq(self, other: float, tol: float = functions.EPSILON) -> SymbolicAssert:
        """Assert that the expression equals the given value within tolerance."""
        self._create_validator(f"= {other}", functions.is_eq, other, tol)
        return self._create_new_assertion_if_needed()

    def is_negative(self, tol: float = functions.EPSILON) -> SymbolicAssert:
        """Assert that the expression is negative."""
        self._update_validator(SymbolicValidator(name="< 0", fn=functools.partial(functions.is_negative, tol=tol)))
        return self._create_new_assertion_if_needed()

    def is_positive(self, tol: float = functions.EPSILON) -> SymbolicAssert:
        """Assert that the expression is positive."""
        self._update_validator(SymbolicValidator(name="> 0", fn=functools.partial(functions.is_positive, tol=tol)))
        return self._create_new_assertion_if_needed()

    def _create_new_assertion_if_needed(self) -> SymbolicAssert:
        """Create a new assertion node for chaining if the current one already has a validator."""
        # If we don't have a context, we can't create new assertions
        if self._context is None:
            return self

        # For chaining, we need to create a new assertion node and add it to the current check
        if not self._context._graph.children:
            raise DQXError("Cannot create assertion without an active check")

        # Create new assertion node
        node = self._context.create_assertion(actual=self._actual)
        
        # Create new SymbolicAssert
        sa = SymbolicAssert(actual=self._actual, listeners=[node], context=self._context)

        # Attach to the most recent check node (same as assert_that does)
        current_check = self._context._graph.children[-1]
        current_check.add_child(node)

        # Preserve label and severity from current assertion
        if self._label or self._severity:
            sa.on(label=self._label, severity=self._severity)

        return sa


class Context:
    """
    Execution context for data quality checks containing the dependency graph and assertion utilities.

    Provides methods to create assertions and manage the verification workflow.
    The Context owns the SymbolTable and provides factory methods for creating
    graph nodes that need access to the symbol table.
    """

    def __init__(self, suite: str) -> None:
        """
        Initialize the context with a root graph node.

        Args:
            suite: Name of the verification suite
        """
        self._symbol_table = SymbolTable()  # Context owns the symbol table
        self._graph = graph.RootNode(name=suite, context=self)

    @property
    def symbol_table(self) -> SymbolTable:
        """Get the symbol table managed by this context."""
        return self._symbol_table

    @property
    def key(self) -> ResultKeyProvider:
        """Get a result key provider for creating time-based metric keys."""
        return ResultKeyProvider()

    def create_check(
        self,
        name: str,
        tags: list[str] | None = None,
        label: str | None = None,
        datasets: list[str] | None = None,
    ) -> graph.CheckNode:
        """
        Factory method to create a check node.

        Args:
            name: Unique identifier for the check
            tags: Optional tags for categorizing the check
            label: Optional human-readable label
            datasets: Optional list of datasets the check applies to

        Returns:
            CheckNode that can access context through its root node
        """
        return graph.CheckNode(
            name=name,
            tags=tags,
            label=label,
            datasets=datasets,
        )

    def create_assertion(
        self,
        actual: sp.Expr,
        label: str | None = None,
        severity: SeverityLevel | None = None,
        validator: SymbolicValidator | None = None,
    ) -> graph.AssertionNode:
        """
        Factory method to create an assertion node.

        Args:
            actual: Symbolic expression to evaluate
            label: Optional human-readable description
            severity: Optional severity level for failures
            validator: Optional validator function

        Returns:
            AssertionNode that can access context through its root node
        """
        return graph.AssertionNode(
            actual=actual,
            label=label,
            severity=severity,
            validator=validator,
        )

    def assert_that(self, expr: sp.Expr) -> SymbolicAssert:
        """
        Create a symbolic assertion for the given expression.

        Args:
            expr: Symbolic expression to assert on

        Returns:
            SymbolicAssert instance for chaining validation methods

        Raises:
            DQXError: If no active check node exists to attach assertion to
        """
        if not self._graph.children:
            raise DQXError("Cannot create assertion without an active check")

        # Use factory method
        node = self.create_assertion(actual=expr)
        sa = SymbolicAssert(actual=expr, listeners=[node], context=self)

        # Attach to the most recent check node
        current_check = self._graph.children[-1]
        current_check.add_child(node)

        return sa

    def pending_metrics(self, dataset: str | None = None) -> Sequence[MetricSpec]:
        """
        Get pending metrics for the specified dataset or all datasets if none specified.

        Args:
            dataset: Optional dataset name. If None, returns metrics for all datasets.

        Returns:
            Sequence of pending metric specifications
        """
        # Get metrics from symbol table
        return list(self._symbol_table.get_required_metrics(dataset))

    def evaluate(self, key: ResultKey) -> None:
        """Evaluate all ready symbols and assertions in the graph."""
        # Use the symbol table for evaluation
        symbol_table = self._symbol_table

        # Evaluate ready symbols through the symbol table
        symbol_table.evaluate_ready_symbols(key)

        # Evaluate assertions
        for assertion in self._graph.assertions():
            assertion.evaluate()

        # Update check node statuses based on their children
        for check in self._graph.checks():
            check._value = graph.aggregate_children_status(check.children)


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
        self._provider = MetricProvider(db)
        self._name = name.strip()

    def collect(self, key: ResultKey) -> Context:
        """
        Collect all checks and build the dependency graph without executing analysis.

        Args:
            key: The result key defining the time period and tags for analysis

        Returns:
            Context containing the collected checks and dependency graph

        Raises:
            DQXError: If check collection fails or duplicate checks are found
        """
        context = Context(suite=self._name)
        self._execute_checks(context)
        self._build_dependency_graph(context, key)
        return context

    def _execute_checks(self, context: Context) -> None:
        """Execute all checks to collect assertions."""
        for check in self._checks:
            check(self._provider, context)

    def _build_dependency_graph(self, context: Context, key: ResultKey) -> None:
        """Build the dependency graph with symbols and metrics."""
        for assertion in context._graph.assertions():
            self._process_assertion_symbols(assertion, key, context)

    def _process_assertion_symbols(self, assertion: graph.AssertionNode, key: ResultKey, context: Context) -> None:
        """
        Process symbols for a single assertion with improved performance.

        This method has been updated to accept a context parameter instead of
        accessing the symbol table through the graph.

        Args:
            assertion: The assertion node to process
            key: The result key for metric evaluation
            context: The context containing the symbol table to register symbols in
        """
        # Cache symbol lookups to avoid repeated provider calls
        symbol_cache: dict[sp.Symbol, SymbolicMetric] = {}

        # Find the check node that contains this assertion
        check_node = None
        root = assertion.root
        for check in root.checks():
            if assertion in check.children:
                check_node = check
                break

        if not check_node:
            raise RuntimeError("Assertion not found in any check node")

        symbol_table = context.symbol_table

        for sym in sorted(assertion.actual.free_symbols, key=str):
            # Check if symbol already registered
            if symbol_table.get(sym) is not None:
                continue

            if sym in symbol_cache:
                sm = symbol_cache[sym]
            else:
                sm = self._provider.get_symbol(sym)
                symbol_cache[sym] = sm

            # Register symbol in symbol table
            self._register_symbol_in_table(sm, key, symbol_table, check_node)

    def _register_symbol_in_table(
        self, sm: SymbolicMetric, key: ResultKey, symbol_table: SymbolTable, check_node: graph.CheckNode
    ) -> None:
        """Register symbol and its dependencies in the symbol table."""
        # Use the symbol table's register_from_provider method
        symbol = symbol_table.register_from_provider(sm, key)
        
        # If the symbol has multiple datasets or check has datasets, we need to bind it
        # The register_from_provider already handles the first dataset from sm.datasets
        # But we need to ensure consistency with the check's datasets
        entry = symbol_table.get(symbol)
        if entry and entry.dataset is None and check_node.datasets:
            # Bind the symbol to the first dataset from the check
            dataset = check_node.datasets[0]
            entry.dataset = dataset
            symbol_table.bind_symbol_to_dataset(symbol, dataset)

        # Track symbol-check relationship
        symbol_table.register_symbol_for_check(symbol, check_node.name)
        # Note: CheckNode no longer tracks symbols directly - symbols are managed by AssertionNodes

    def run(self, datasources: dict[str, SqlDataSource], key: ResultKey, threading: bool = False) -> Context:
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
        self._validate_datasources(datasources)

        # Create a context
        ctx = self.collect(key)

        # Mark checks without datasets as failed if multiple datasets are provided
        if len(datasources) > 1:
            for check_node in ctx._graph.checks():
                # Check if this check has no datasets specified
                check_fn = None
                for check in self._checks:
                    if check.__name__ == check_node.name:
                        check_fn = check
                        break

                if check_fn and isinstance(check_fn, DecoratedCheck):
                    metadata = check_fn._check_metadata  # type: ignore[attr-defined]
                    if metadata["datasets"] is None:
                        # Fail the check instead of throwing an error
                        failure_msg = (
                            f"Check '{metadata['name']}' does not specify datasets "
                            f"and cannot be run with multiple datasets. "
                            f"Either specify datasets explicitly in the @check decorator or provide only one dataset."
                        )
                        check_node._value = Some(failure_msg)

        ctx._graph.impute_datasets(list(datasources.keys()))

        # Run the checks per dataset
        for ds_name, ds in datasources.items():
            try:
                self._analyze_datasource(ds_name, ds, ctx, key, threading)
            except Exception as e:
                # Log the original error for debugging
                logging.error(f"Analysis failed for dataset {ds_name}: {str(e)}", exc_info=True)
                symbol_table = ctx.symbol_table
                symbol_table.mark_dataset_failed(ds_name, str(e))

        # Update the symbol table to reflect that metrics are now available
        # This marks symbols as ready for evaluation
        symbol_table = ctx.symbol_table
        for ds_name in datasources.keys():
            symbol_table.mark_dataset_ready(ds_name)

        # Run the checks
        ctx.evaluate(key)
        return ctx

    def _validate_datasources(self, datasources: dict[str, SqlDataSource]) -> None:
        """Validate that datasources are provided."""
        if not datasources:
            raise DQXError("No data sources provided!")

    def _analyze_datasource(
        self, ds_name: str, ds: SqlDataSource, ctx: Context, key: ResultKey, threading: bool
    ) -> None:
        """Analyze a single datasource and persist results."""
        analyzer = Analyzer()
        analyzer.analyze(ds, ctx.pending_metrics(ds_name), key, threading=threading)
        analyzer.persist(self._provider._db)
        # Mark pending metrics as success is now handled by symbol table
        pass


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
    tags: list[str] = [],
    label: str | None = None,
    datasets: list[str] | None = None,
) -> None:
    """
    Internal function to create and register a check node in the context graph.

    Args:
        provider: Metric provider for the check
        context: Execution context
        _check: Check function to execute
        tags: Optional tags for the check
        label: Optional human-readable label
        datasets: Optional list of datasets the check applies to

    Raises:
        DQXError: If a check with the same name already exists
    """
    # Use context factory method
    node = context.create_check(name=_check.__name__, tags=tags, label=label, datasets=datasets)

    if context._graph.exists(node):
        raise DQXError(f"Check {node.name} already exists in the graph!")

    context._graph.add_child(node)  # This node should be the last node in the graph

    # Call the symbolic check to collect assertions for this check node
    _check(provider, context)


@overload
def check(_check: CheckProducer) -> DecoratedCheck: ...


@overload
def check(
    *, tags: list[str] = [], label: str | None = None, datasets: list[str] | None = None
) -> Callable[[CheckProducer], DecoratedCheck]: ...


def check(
    _check: CheckProducer | None = None,
    *,
    tags: list[str] = [],
    label: str | None = None,
    datasets: list[str] | None = None,
) -> DecoratedCheck | Callable[[CheckProducer], DecoratedCheck]:
    """
    Decorator for creating data quality check functions.

    Can be used with or without parameters:

    @check
    def my_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    @check(tags=["critical"], label="Important Check", datasets=["ds1"])
    def my_labeled_check(mp: MetricProvider, ctx: Context) -> None:
        # check logic

    Args:
        _check: The check function (when used without parentheses)
        tags: Optional tags for categorizing the check
        label: Optional human-readable label for the check
        datasets: Optional list of datasets the check applies to.
                 If not specified and multiple datasets are provided at runtime,
                 an error will be raised.

    Returns:
        Decorated check function or decorator function
    """
    if _check is not None:
        # Simple @check decorator without parentheses
        wrapped = functools.wraps(_check)(
            functools.partial(_create_check, _check=_check, tags=tags, label=label, datasets=datasets)
        )
        # Store metadata for validation
        wrapped._check_metadata = {  # type: ignore[attr-defined]
            "name": _check.__name__,
            "datasets": datasets,
            "tags": tags,
            "label": label,
        }
        return cast(DecoratedCheck, wrapped)

    def decorator(fn: CheckProducer) -> DecoratedCheck:
        wrapped = functools.wraps(fn)(
            functools.partial(_create_check, _check=fn, tags=tags, label=label, datasets=datasets)
        )
        # Store metadata for validation
        wrapped._check_metadata = {  # type: ignore[attr-defined]
            "name": fn.__name__,
            "datasets": datasets,
            "tags": tags,
            "label": label,
        }
        return cast(DecoratedCheck, wrapped)

    return decorator
