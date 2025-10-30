from __future__ import annotations

import datetime
import logging
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Lock
from typing import TYPE_CHECKING, Callable, overload

import sympy as sp
from returns.result import Failure, Result

from dqx import compute, specs
from dqx.common import DQXError, ExecutionId, ResultKey, RetrievalFn, Tags
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

if TYPE_CHECKING:
    from dqx.graph.traversal import Graph

logger = logging.getLogger(__name__)

SymbolIndex = dict[sp.Symbol, "SymbolicMetric"]


@dataclass
class SymbolInfo:
    """Information about a symbol in an expression.

    Captures metadata about a computed metric symbol, including its value
    and the context in which it was evaluated.

    Attributes:
        name: Symbol identifier (e.g., "x_1", "x_2")
        metric: Human-readable metric description (e.g., "average(price)")
        dataset: Name of the dataset this metric was computed from (optional)
        value: Computation result - Success(float) or Failure(error_message)
        yyyy_mm_dd: Date when the metric was evaluated
        tags: Additional metadata from ResultKey (e.g., {"env": "prod"})
    """

    name: str
    metric: str
    dataset: str | None
    value: Result[float, str]
    yyyy_mm_dd: datetime.date
    tags: Tags = field(default_factory=dict)


@dataclass
class SymbolicMetric:
    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    metric_spec: MetricSpec
    lag: int = 0
    dataset: str | None = None
    required_metrics: list[sp.Symbol] = field(default_factory=list)


def _create_lazy_retrieval_fn(provider: "MetricProvider", metric_spec: MetricSpec, symbol: sp.Symbol) -> RetrievalFn:
    """Create a lazy retrieval function that resolves dataset at evaluation time.

    This factory creates a retrieval function that defers dataset resolution
    until the metric is actually evaluated. This allows metrics to be created
    before their dataset is known (during imputation), while ensuring the
    correct dataset is used during evaluation.

    Args:
        provider: The MetricProvider instance containing the symbol registry.
        metric_spec: The metric specification to evaluate.
        symbol: The symbol representing this metric.

    Returns:
        A retrieval function that looks up the dataset from the SymbolicMetric
        at evaluation time and uses it to fetch the correct metric value.
    """

    def lazy_retrieval_fn(key: ResultKey) -> Result[float, str]:
        # Look up the current dataset from the SymbolicMetric
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        # Call the compute function with the resolved dataset and execution_id
        return compute.simple_metric(provider._db, metric_spec, symbolic_metric.dataset, key, provider.execution_id)

    return lazy_retrieval_fn


def _create_lazy_extended_fn(
    provider: "MetricProvider",
    compute_fn: Callable[[MetricDB, MetricSpec, str, ResultKey, ExecutionId], Result[float, str]],
    metric_spec: MetricSpec,
    symbol: sp.Symbol,
) -> RetrievalFn:
    """Create a lazy retrieval function for extended metrics (DoD, WoW, etc).

    Similar to _create_lazy_retrieval_fn but for extended metrics that
    need to call specialized compute functions.

    Args:
        provider: The MetricProvider instance.
        compute_fn: The compute function to use (e.g., compute.day_over_day).
        metric_spec: The base metric specification.
        symbol: The symbol representing this metric.

    Returns:
        A lazy retrieval function for the extended metric.
    """

    def lazy_extended_fn(key: ResultKey) -> Result[float, str]:
        # Look up the current dataset
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        # Call the compute function with the resolved dataset and execution_id
        return compute_fn(provider._db, metric_spec, symbolic_metric.dataset, key, provider.execution_id)

    return lazy_extended_fn


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: list[SymbolicMetric] = []
        self._symbol_index: SymbolIndex = {}

        self._curr_index: int = 0
        self._mutex = Lock()

    @property
    def symbolic_metrics(self) -> Iterable[SymbolicMetric]:
        return self._metrics

    @property
    def index(self) -> SymbolIndex:
        return self._symbol_index

    @property
    def metrics(self) -> list[SymbolicMetric]:
        return self._metrics

    def _next_symbol(self, prefix: str = "x") -> sp.Symbol:
        """Generate the next symbolic metric.

        Args:
            prefix (str, optional): symbol prefix. Defaults to "x".

        Returns:
            str:
        """
        with self._mutex:
            self._curr_index += 1
            return sp.Symbol(f"{prefix}_{self._curr_index}")

    def get(self, symbol: sp.Symbol | str) -> SymbolicMetric:
        """Find the first symbol data that matches the given symbol.

        Args:
            symbol: Either a sympy Symbol object or a string representation
                   of the symbol (e.g., "x_1")

        Returns:
            SymbolicMetric containing the symbol's metadata

        Raises:
            DQXError: If the symbol is not found
        """
        # Convert string to Symbol if needed
        if isinstance(symbol, str):
            symbol = sp.Symbol(symbol)

        if symbol not in self.index:
            raise DQXError(f"Symbol {symbol} not found.")

        return self.index[symbol]

    def remove(self, symbol: sp.Symbol) -> None:
        """Remove a symbolic metric from the registry.

        Args:
            symbol: The symbolic metric to remove
        """
        # Remove from metrics list
        self._metrics = [sm for sm in self._metrics if sm.symbol != symbol]

        # Remove from index
        if symbol in self.index:
            del self.index[symbol]

    def _exists(self, spec: MetricSpec, lag: int, dataset: str) -> sp.Symbol | None:
        for sm in self._metrics:
            if sm.metric_spec == spec and sm.lag == lag and sm.dataset == dataset:
                return sm.symbol

        return None

    def register(
        self,
        fn: RetrievalFn,
        metric_spec: MetricSpec,
        lag: int = 0,
        dataset: str | None = None,
        required_metrics: list[sp.Symbol] | None = None,
    ) -> sp.Symbol:
        """Register a symbolic metric."""
        sym = self._next_symbol()

        # Check if symbol already exists, returns the existing one
        if dataset and (existing_sym := self._exists(metric_spec, lag, dataset)) is not None:
            return existing_sym

        self._metrics.append(
            sm := SymbolicMetric(
                name=metric_spec.name,
                symbol=sym,
                fn=fn,
                metric_spec=metric_spec,
                lag=lag,
                dataset=dataset,
                required_metrics=required_metrics or [],
            )
        )

        # Update the reversed index
        self.index[sym] = sm

        return sym

    def collect_symbols(self, key: ResultKey) -> list[SymbolInfo]:
        """
        Collect all symbol values with metadata.

        This method retrieves information about all symbols (metrics) that were
        registered, evaluates them, and returns their values along with metadata.
        Symbols are sorted by name for consistent ordering.

        Args:
            key: The ResultKey for evaluation context (date and tags)

        Returns:
            List of SymbolInfo instances, sorted by symbol name in natural numeric
            order (x_1, x_2, ..., x_10, x_11, etc. rather than lexicographic).
            Each contains the symbol name, metric description, dataset,
            computed value, and context information (date, tags).

        Example:
            >>> symbols = provider.collect_symbols(key)
            >>> for s in symbols:
            ...     if s.value.is_success():
            ...         print(f"{s.metric}: {s.value.unwrap()}")
        """
        symbols = []

        # Create all SymbolInfo objects
        for symbolic_metric in self.metrics:
            # Calculate the effective key for this symbol
            effective_key = key.lag(symbolic_metric.lag)

            # Try to evaluate the symbol to get its value
            try:
                value = symbolic_metric.fn(effective_key)
            except Exception:
                # In tests, the symbol might not be evaluable
                from returns.result import Failure

                value = Failure("Not evaluated")

            # Create SymbolInfo with all fields
            symbol_info = SymbolInfo(
                name=str(symbolic_metric.symbol),
                metric=symbolic_metric.name,
                dataset=symbolic_metric.dataset,
                value=value,
                yyyy_mm_dd=effective_key.yyyy_mm_dd,  # Use effective date!
                tags=effective_key.tags,
            )
            symbols.append(symbol_info)

        # Sort by symbol numeric suffix for natural ordering (x_1, x_2, ..., x_10)
        # instead of lexicographic ordering (x_1, x_10, x_2)
        sorted_symbols = sorted(symbols, key=lambda s: int(s.name.split("_")[1]))

        return sorted_symbols


class RegistryMixin:
    @property
    def registry(self) -> MetricRegistry:
        raise NotImplementedError("Subclasses must implement registry property.")

    @property
    def metrics(self) -> list[SymbolicMetric]:
        return self.registry._metrics

    @property
    def index(self) -> SymbolIndex:
        return self.registry._symbol_index

    def symbols(self) -> Iterable[sp.Symbol]:
        return self.registry._symbol_index.keys()

    def get_symbol(self, symbol: sp.Symbol | str) -> SymbolicMetric:
        return self.registry.get(symbol)

    def remove_symbol(self, symbol: sp.Symbol) -> None:
        # Remove dependencies recursively
        sm = self.get_symbol(symbol)
        for dep_symbol in sm.required_metrics:
            self.remove_symbol(dep_symbol)  # Recursive removal only
        self.registry.remove(symbol)

    def collect_symbols(self, key: ResultKey) -> list[SymbolInfo]:
        return self.registry.collect_symbols(key)


class SymbolicMetricBase(ABC, RegistryMixin):
    def __init__(self) -> None:
        self._registry: MetricRegistry = MetricRegistry()

    @property
    def registry(self) -> MetricRegistry:
        return self._registry

    def evaluate(self, symbol: sp.Symbol, key: ResultKey) -> Result[float, str]:
        """
        Evaluate the given symbolic metric.

        This method takes a symbolic metric and an evaluation context (date and tags)
        and returns the computed value of the metric along with its metadata.

        Args:
            symbol: The symbolic metric to evaluate.
            key: The ResultKey for evaluation context (date and tags).

        Returns:
            Result containing the computed value of the metric along with its metadata.
        """
        return self.index[symbol].fn(key)

    def print_symbols(self, key: ResultKey) -> None:
        """
        Collect and display all symbol values in a formatted table.

        This is a convenience method that combines collect_symbols() and
        print_symbols() from the display module into a single call.

        Args:
            key: The ResultKey for evaluation context (date and tags)

        Example:
            >>> # Instead of:
            >>> symbols = provider.collect_symbols(key)
            >>> print_symbols(symbols)
            >>>
            >>> # You can now simply do:
            >>> provider.print_symbols(key)
        """
        from dqx.display import print_symbols

        symbols = self.collect_symbols(key)
        print_symbols(symbols)

    def build_deduplication_map(self, context_key: ResultKey) -> dict[sp.Symbol, sp.Symbol]:
        """Build symbol substitution map for deduplication.

        This method identifies duplicate symbols that represent the same metric
        computed for the same effective date and dataset. It returns a mapping
        from duplicate symbols to their canonical representatives.

        Args:
            context_key: The analysis date context. Used to calculate effective
                        dates for lagged metrics.

        Returns:
            Dict mapping duplicate symbols to canonical symbols. For example:
            {
                sp.Symbol('x_3'): sp.Symbol('x_1'),  # x_3 is duplicate of x_1
                sp.Symbol('x_5'): sp.Symbol('x_2'),  # x_5 is duplicate of x_2
            }

            The canonical symbol is always the one with the lowest index number.
            Empty dict if no duplicates found.

        Example:
            If we have:
            - x_1: average(price) for 2024-01-15
            - x_2: average(price) with lag=1 for 2024-01-16 (effective: 2024-01-15)
            - x_3: average(price) for 2024-01-15 (duplicate of x_1)

            This returns: {x_3: x_1, x_2: x_1}
        """
        groups: dict[tuple[str, str, str | None], list[sp.Symbol]] = {}

        # Group symbols by identity
        for sym_metric in self.metrics:
            # Calculate effective date for this symbol
            effective_date = context_key.yyyy_mm_dd - timedelta(days=sym_metric.lag)

            # Use the human-readable name (e.g., "day_over_day(maximum(tax))")
            # instead of metric_spec.name to properly distinguish extended metrics
            identity = (sym_metric.name, effective_date.isoformat(), sym_metric.dataset)

            if identity not in groups:
                groups[identity] = []
            groups[identity].append(sym_metric.symbol)

        # Build substitution map
        substitutions = {}
        for duplicates in groups.values():
            if len(duplicates) > 1:
                # Keep the lowest numbered symbol as canonical
                duplicates_sorted = sorted(duplicates, key=lambda s: int(s.name.split("_")[1]))
                canonical = duplicates_sorted[0]

                for dup in duplicates_sorted[1:]:
                    substitutions[dup] = canonical

        return substitutions

    def deduplicate_required_metrics(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        """Update required_metrics in all symbolic metrics after deduplication.

        Args:
            substitutions: Map of duplicate symbols to canonical symbols
        """
        for sym_metric in self.metrics:
            if sym_metric.required_metrics:
                # Replace any duplicates in required_metrics
                sym_metric.required_metrics = [substitutions.get(req, req) for req in sym_metric.required_metrics]

    def prune_duplicate_symbols(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        """Remove duplicate symbols from the provider.

        Args:
            substitutions: Map from duplicate symbols to canonical symbols
        """
        if not substitutions:
            return

        to_remove = set(substitutions.keys())

        # Remove duplicate symbols
        self._registry._metrics = [sm for sm in self.metrics if sm.symbol not in to_remove]

        # Remove from index
        removed_symbols = []
        for symbol in to_remove:
            removed_symbols.append(str(symbol))
            del self.index[symbol]

        # Log all removed symbols in one message
        if removed_symbols:
            # Sort symbols by numeric index (x_9 before x_14)
            sorted_symbols = sorted(removed_symbols, key=lambda s: int(s.split("_")[1]))
            logger.info("Pruned %d duplicate symbols: %s", len(sorted_symbols), ", ".join(sorted_symbols))

    def symbol_deduplication(self, graph: "Graph", context_key: ResultKey) -> None:
        """Apply symbol deduplication to graph and provider.

        This method:
        1. Builds a map of duplicate symbols
        2. Applies deduplication to the graph
        3. Updates required_metrics references
        4. Prunes duplicate symbols

        Args:
            graph: The computation graph to apply deduplication to
            context_key: The analysis date context for calculating effective dates
        """
        # Build deduplication map
        substitutions = self.build_deduplication_map(context_key)

        if not substitutions:
            return

        # Apply deduplication to graph
        from dqx.graph.visitors import SymbolDeduplicationVisitor

        dedup_visitor = SymbolDeduplicationVisitor(substitutions)
        graph.dfs(dedup_visitor)  # Use depth-first search to apply visitor

        # Update required_metrics in remaining symbols
        self.deduplicate_required_metrics(substitutions)

        # Prune duplicate symbols
        self.prune_duplicate_symbols(substitutions)


class ExtendedMetricProvider(RegistryMixin):
    """A provider for derivative metrics that builds on top of primitive metrics."""

    def __init__(self, provider: MetricProvider) -> None:
        self._provider = provider

    @property
    def provider(self) -> MetricProvider:
        return self._provider

    @property
    def registry(self) -> MetricRegistry:
        return self._provider.registry

    @property
    def db(self) -> MetricDB:
        return self._provider._db

    @property
    def execution_id(self) -> ExecutionId:
        """The execution ID from the parent provider."""
        return self._provider.execution_id

    def day_over_day(self, metric: sp.Symbol, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        # Get the full SymbolicMetric object
        symbolic_metric = self._provider.get_symbol(metric)
        spec = symbolic_metric.metric_spec

        # Create base metrics with proper lag accumulation
        lag_0 = self.provider.create_metric(spec, lag=lag + 0, dataset=symbolic_metric.dataset)
        lag_1 = self.provider.create_metric(spec, lag=lag + 1, dataset=symbolic_metric.dataset)

        # Generate symbol first
        sym = self.registry._next_symbol()

        # Create lazy function for DoD
        fn = _create_lazy_extended_fn(self._provider, compute.day_over_day, spec, sym)

        # Register with lazy function
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=specs.DayOverDay.from_base_spec(spec).name,
                symbol=sym,
                fn=fn,
                metric_spec=specs.DayOverDay.from_base_spec(spec),
                lag=lag,  # Use the provided lag instead of 0
                dataset=dataset,
                required_metrics=[lag_0, lag_1],
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym

    def week_over_week(self, metric: sp.Symbol, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        # Get the full SymbolicMetric object
        symbolic_metric = self._provider.get_symbol(metric)
        spec = symbolic_metric.metric_spec

        # Create required lag metrics with proper lag accumulation
        lag_0 = self.provider.create_metric(spec, lag=lag + 0, dataset=symbolic_metric.dataset)
        lag_7 = self.provider.create_metric(spec, lag=lag + 7, dataset=symbolic_metric.dataset)

        # Generate symbol first
        sym = self.registry._next_symbol()

        # Create lazy function for WoW
        fn = _create_lazy_extended_fn(self._provider, compute.week_over_week, spec, sym)

        # Register with lazy function
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=specs.WeekOverWeek.from_base_spec(spec).name,
                symbol=sym,
                fn=fn,
                metric_spec=specs.WeekOverWeek.from_base_spec(spec),
                lag=lag,  # Use the provided lag instead of 0
                dataset=dataset,
                required_metrics=[lag_0, lag_7],
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym

    def stddev(self, metric: sp.Symbol, offset: int, n: int, dataset: str | None = None) -> sp.Symbol:
        """Calculate standard deviation of a metric over a window of values.

        This method computes the standard deviation of a metric across n consecutive
        days, starting from the specified offset.

        Args:
            metric: The base metric symbol to calculate standard deviation for.
            offset: The starting position of the window (0 = today, 1 = yesterday, etc.).
                    This determines where the window of values begins.
            n: The number of values to include in the standard deviation calculation.
               The window will span from offset to offset+n-1.
            dataset: Optional dataset name. If not provided, uses the dataset from
                    the base metric.

        Returns:
            A Symbol representing the standard deviation metric.

        Example:
            >>> # Calculate stddev of last 7 days starting from today (offset=0)
            >>> avg_price = provider.average("price")
            >>> stddev_7d = provider.ext.stddev(avg_price, offset=0, n=7)
            >>>
            >>> # Calculate stddev of 5 days starting from 2 days ago (offset=2)
            >>> stddev_5d = provider.ext.stddev(avg_price, offset=2, n=5)
        """
        # Get the full SymbolicMetric object
        symbolic_metric = self._provider.get_symbol(metric)
        spec = symbolic_metric.metric_spec

        # Create required metrics with properly accumulated lag values
        required = []
        for i in range(offset, offset + n):
            # Each required metric needs its own lag value
            required_metric = self.provider.create_metric(spec, lag=i, dataset=symbolic_metric.dataset)
            required.append(required_metric)

        # Create lazy function for stddev using lambda to handle the size parameter
        fn = _create_lazy_extended_fn(
            self._provider,
            lambda db, metric, dataset, key, execution_id: compute.stddev(db, metric, n, dataset, key, execution_id),
            spec,
            sym := self.registry._next_symbol(),
        )

        # Register with lazy function
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=specs.Stddev.from_base_spec(spec, offset, n).name,
                symbol=sym,
                fn=fn,
                metric_spec=specs.Stddev.from_base_spec(spec, offset, n),
                lag=offset,  # stddev itself should have lag=offset (not lag=0)
                dataset=dataset,
                required_metrics=required,
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym


class MetricProvider(SymbolicMetricBase):
    def __init__(self, db: MetricDB, execution_id: ExecutionId) -> None:
        super().__init__()
        self._db = db
        self._execution_id = execution_id

    @property
    def execution_id(self) -> ExecutionId:
        """The execution ID for this provider instance."""
        return self._execution_id

    @property
    def ext(self) -> ExtendedMetricProvider:
        return ExtendedMetricProvider(self)

    def create_metric(
        self,
        metric_spec: MetricSpec,
        lag: int = 0,
        dataset: str | None = None,
    ) -> sp.Symbol:
        """Create a metric symbol handling both simple and extended metrics.

        This method intelligently routes metric creation based on the type:
        - Simple metrics: Uses the standard metric() method
        - Extended metrics: Routes to the appropriate extended metric method

        Args:
            metric_spec: The metric specification to create.
            lag: Number of days to lag the metric evaluation.
            dataset: Optional dataset name. Can be provided now or imputed later.

        Returns:
            A Symbol representing this metric in expressions.

        Raises:
            ValueError: If the metric type is not supported.
        """
        if isinstance(metric_spec, specs.SimpleMetricSpec):
            # Simple metric - use the standard metric method
            return self.metric(metric_spec, lag=lag, dataset=dataset)

        # Extended metric - need to handle specially based on type
        if isinstance(metric_spec, specs.DayOverDay):
            # Don't apply lag to base metric - let DoD handle lag propagation
            base_metric = self.create_metric(metric_spec.base_spec, lag=0, dataset=dataset)
            return self.ext.day_over_day(base_metric, lag=lag, dataset=dataset)
        elif isinstance(metric_spec, specs.WeekOverWeek):
            # Don't apply lag to base metric - let WoW handle lag propagation
            base_metric = self.create_metric(metric_spec.base_spec, lag=0, dataset=dataset)
            return self.ext.week_over_week(base_metric, lag=lag, dataset=dataset)
        elif isinstance(metric_spec, specs.Stddev):
            # Don't apply lag to base metric - stddev will handle lag propagation
            base_metric = self.create_metric(metric_spec.base_spec, lag=0, dataset=dataset)
            # Extract offset and n from the Stddev spec parameters
            params = metric_spec.parameters
            stddev_offset = params["offset"]
            stddev_n = params["n"]
            # Apply the input lag to stddev's offset parameter
            return self.ext.stddev(base_metric, offset=stddev_offset + lag, n=stddev_n, dataset=dataset)
        else:
            raise ValueError(f"Unsupported extended metric type: {metric_spec.metric_type}")

    def metric(
        self,
        metric: specs.SimpleMetricSpec,
        lag: int = 0,
        dataset: str | None = None,
    ) -> sp.Symbol:
        """Register a metric symbol with lazy dataset resolution.

        Creates a symbolic metric that can be evaluated later. If dataset is not
        provided at registration time, it will be resolved during imputation and
        used at evaluation time through lazy retrieval.

        Args:
            metric: The metric specification to register.
            lag: Number of days to lag the metric evaluation.
            dataset: Optional dataset name. Can be provided now or imputed later.

        Returns:
            A Symbol representing this metric in expressions.
        """
        # Generate symbol first
        sym = self.registry._next_symbol()

        # Create lazy retrieval function that will resolve dataset at evaluation time
        fn = _create_lazy_retrieval_fn(self, metric, sym)

        # Register with the lazy function
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=metric.name,
                symbol=sym,
                fn=fn,
                metric_spec=metric.clone(),
                lag=lag,
                dataset=dataset,
                required_metrics=[],
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym

    def num_rows(self, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.NumRows(), lag, dataset)

    def first(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.First(column), lag, dataset)

    def average(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.Average(column), lag, dataset)

    def minimum(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.Minimum(column), lag, dataset)

    def maximum(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.Maximum(column), lag, dataset)

    def sum(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.Sum(column), lag, dataset)

    def null_count(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.NullCount(column), lag, dataset)

    def variance(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.Variance(column), lag, dataset)

    def duplicate_count(self, columns: list[str], lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.DuplicateCount(columns), lag, dataset)

    @overload
    def count_values(self, column: str, values: bool, lag: int = 0, dataset: str | None = ...) -> sp.Symbol: ...

    @overload
    def count_values(self, column: str, values: int, lag: int = 0, dataset: str | None = ...) -> sp.Symbol: ...

    @overload
    def count_values(self, column: str, values: str, lag: int = 0, dataset: str | None = ...) -> sp.Symbol: ...

    @overload
    def count_values(self, column: str, values: list[int], lag: int = 0, dataset: str | None = ...) -> sp.Symbol: ...

    @overload
    def count_values(self, column: str, values: list[str], lag: int = 0, dataset: str | None = ...) -> sp.Symbol: ...

    def count_values(
        self,
        column: str,
        values: int | str | bool | list[int] | list[str],
        lag: int = 0,
        dataset: str | None = None,
    ) -> sp.Symbol:
        """Count occurrences of specific value(s) in a column.

        This operation counts only the specified values, never NULLs.
        Empty strings are counted as values, not as NULLs.

        Args:
            column: Column name to count values in
            values: Value(s) to count - single int/str/bool or list of int/str
            lag: Lag offset in days
            dataset: Optional dataset name

        Returns:
            Symbol representing the count

        Examples:
            >>> from dqx import ValidationSuite
            >>> suite = ValidationSuite("test")

            >>> # Count single value
            >>> suite.count_values("status", "active")

            >>> # Count multiple values efficiently in one query
            >>> suite.count_values("status", ["active", "pending"])

            >>> # Count integer values
            >>> suite.count_values("type_id", [1, 2, 3])

            >>> # Count boolean values
            >>> suite.count_values("is_active", True)
            >>> suite.count_values("is_deleted", False)

        Performance Note:
            Counting multiple values with a list is more efficient than
            making multiple separate count_values calls.
        """
        return self.metric(specs.CountValues(column, values), lag, dataset)

    def unique_count(self, column: str, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        """Count unique (distinct) non-null values in a column.

        This operation counts the number of distinct non-null values in a column.
        NULL values are not included in the count.

        Note: This metric cannot be merged across partitions. The unique count
        from multiple partitions cannot be summed to get the total unique count
        because the same values might appear in multiple partitions.

        Args:
            column: Column name to count unique values in
            lag: Lag offset in days
            dataset: Optional dataset name

        Returns:
            Symbol representing the unique count

        Examples:
            >>> from dqx import ValidationSuite
            >>> suite = ValidationSuite("test")

            >>> # Count unique product IDs
            >>> suite.unique_count("product_id")

            >>> # Count unique users with a 7-day lag
            >>> suite.unique_count("user_id", lag=7)

        See Also:
            - duplicate_count: For counting duplicate values
            - count_values: For counting specific values
            - null_count: For counting NULL values
        """
        return self.metric(specs.UniqueCount(column), lag, dataset)
