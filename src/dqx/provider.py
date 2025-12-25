from __future__ import annotations

import datetime
import logging
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, overload

import sympy as sp
from returns.maybe import Some
from returns.result import Failure, Result

from dqx import compute, specs
from dqx.cache import MetricCache
from dqx.common import DQXError, ExecutionId, MetricKey, ResultKey, RetrievalFn, Tags
from dqx.models import Metric
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

if TYPE_CHECKING:
    from dqx.cache import MetricCache
    from dqx.common import SqlDataSource
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
        data_av_ratio: Data availability ratio (0.0 to 1.0)
    """

    name: str
    metric: str
    dataset: str | None
    value: Result[float, str]
    yyyy_mm_dd: datetime.date
    tags: Tags = field(default_factory=dict)
    data_av_ratio: float = 1.0


@dataclass
class SymbolicMetric:
    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    metric_spec: MetricSpec
    lag: int = 0
    dataset: str | None = None
    required_metrics: list[sp.Symbol] = field(default_factory=list)
    data_av_ratio: float = 1.0


def _create_lazy_retrieval_fn(provider: "MetricProvider", metric_spec: MetricSpec, symbol: sp.Symbol) -> RetrievalFn:
    """Create retrieval function with deferred dataset resolution."""

    def lazy_simple_fn(key: ResultKey) -> Result[float, str]:
        # Look up the current dataset from the SymbolicMetric
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        # Check data availability before computing
        if symbolic_metric.data_av_ratio < provider._data_av_threshold:
            return Failure(f"Insufficient data availability: {symbolic_metric.data_av_ratio:.2f}")

        # Call the compute function with the resolved dataset and execution_id
        return compute.simple_metric(metric_spec, symbolic_metric.dataset, key, provider.execution_id, provider._cache)

    return lazy_simple_fn


def _create_lazy_extended_fn(
    provider: "MetricProvider",
    compute_fn: Callable[..., Result[float, str]],
    metric_spec: MetricSpec,
    symbol: sp.Symbol,
) -> RetrievalFn:
    """Create retrieval function for extended metrics with deferred dataset resolution."""

    def lazy_extended_fn(key: ResultKey) -> Result[float, str]:
        # Look up the current dataset
        try:
            symbolic_metric = provider.get_symbol(symbol)
        except DQXError as e:
            return Failure(f"Failed to resolve symbol {symbol}: {str(e)}")

        if symbolic_metric.dataset is None:
            return Failure(f"Dataset not imputed for metric {symbolic_metric.name}")

        # Check data availability before computing
        if symbolic_metric.data_av_ratio < provider._data_av_threshold:
            return Failure(f"Insufficient data availability: {symbolic_metric.data_av_ratio:.2f}")

        # Call the compute function with the resolved dataset and execution_id
        # Note: compute_fn may be a lambda that already includes additional parameters
        return compute_fn(metric_spec, symbolic_metric.dataset, key, provider.execution_id, provider._cache)

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
        """Generate next unique symbol name (x_1, x_2, etc.)."""
        with self._mutex:
            self._curr_index += 1
            return sp.Symbol(f"{prefix}_{self._curr_index}")

    def get(self, symbol: sp.Symbol | str) -> SymbolicMetric:
        """Retrieve symbolic metric by symbol.

        Args:
            symbol: Symbol object or string (e.g., "x_1")

        Returns:
            SymbolicMetric containing the symbol's metadata

        Raises:
            DQXError: If symbol not found
        """
        # Convert string to Symbol if needed
        if isinstance(symbol, str):
            symbol = sp.Symbol(symbol)

        if symbol not in self.index:
            raise DQXError(f"Symbol {symbol} not found.")

        return self.index[symbol]

    def remove(self, symbol: sp.Symbol) -> None:
        """Remove symbolic metric from registry."""
        # Remove from metrics list
        self._metrics = [sm for sm in self._metrics if sm.symbol != symbol]

        # Remove from index
        if symbol in self.index:
            del self.index[symbol]

    def _exists(self, spec: MetricSpec, lag: int, dataset: str) -> sp.Symbol | None:
        """Check if metric already exists in registry."""
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
        """Register new symbolic metric or return existing one."""
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
        """Evaluate all symbols and return values with metadata.

        Args:
            key: ResultKey for evaluation context (date and tags)

        Returns:
            List of SymbolInfo sorted by natural numeric order (x_1, x_2, ..., x_10).
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
                data_av_ratio=symbolic_metric.data_av_ratio,  # Propagate data availability ratio!
            )
            symbols.append(symbol_info)

        # Sort by symbol numeric suffix for natural ordering (x_1, x_2, ..., x_10)
        # instead of lexicographic ordering (x_1, x_10, x_2)
        sorted_symbols = sorted(symbols, key=lambda s: int(s.name.split("_")[1]))

        return sorted_symbols

    def topological_sort(self) -> None:
        """Sort metrics by dependencies for correct evaluation order.

        Raises:
            DQXError: If circular dependency detected.
        """
        from collections import deque

        n = len(self._metrics)
        if n == 0:
            return

        # Build dependency graph
        in_degree: dict[sp.Symbol, int] = {}
        adjacency: dict[sp.Symbol, list[sp.Symbol]] = {}

        # Initialize structures
        for sm in self._metrics:
            # Only count dependencies that are actually in the registry
            internal_deps = [req for req in sm.required_metrics if req in self._symbol_index]
            in_degree[sm.symbol] = len(internal_deps)
            adjacency[sm.symbol] = []

        # Build reverse adjacency (who depends on me?)
        for sm in self._metrics:
            for req_symbol in sm.required_metrics:
                if req_symbol in adjacency:  # Only if required metric is in current list
                    adjacency[req_symbol].append(sm.symbol)

        # Initialize queue with metrics having no dependencies
        queue: deque[SymbolicMetric] = deque(sm for sm in self._metrics if in_degree[sm.symbol] == 0)
        result: list[SymbolicMetric] = []

        # Process metrics in topological order
        while queue:
            current = queue.popleft()
            result.append(current)

            # Reduce in-degree for dependent metrics
            for dependent_symbol in adjacency[current.symbol]:
                in_degree[dependent_symbol] -= 1
                if in_degree[dependent_symbol] == 0:
                    # Use self.get() to retrieve the metric
                    queue.append(self.get(dependent_symbol))

        # Check for cycles
        if len(result) != n:
            # Find metrics involved in cycle
            remaining = [sm for sm in self._metrics if sm not in result]
            cycle_info = self._find_cycle_details(remaining)
            raise DQXError(f"Circular dependency detected:\n{cycle_info}")

        # Replace _metrics with sorted order
        self._metrics = result

    def calculate_data_av_ratios(self, datasources: dict[str, "SqlDataSource"], key: ResultKey) -> None:
        """Calculate data availability ratios for all metrics.

        Updates the data_av_ratio field of each SymbolicMetric based on
        whether its effective dates are in the dataset's skip_dates.

        For simple metrics: 0.0 if date is excluded, 1.0 otherwise
        For extended metrics: average of child metric ratios

        Args:
            datasources: Dictionary mapping dataset names to SqlDataSource instances
            key: ResultKey providing context date for lag calculations
        """
        # Import here to avoid circular dependency

        # Ensure metrics are sorted by dependencies
        self.topological_sort()

        # Calculate ratios in dependency order
        for sm in self._metrics:
            if not sm.required_metrics:
                # Simple metric - check if its effective date is excluded
                effective_date = key.yyyy_mm_dd - timedelta(days=sm.lag)

                # Get skip_dates from the datasource for this metric's dataset
                if sm.dataset and sm.dataset in datasources:
                    skip_dates = datasources[sm.dataset].skip_dates
                    sm.data_av_ratio = 0.0 if effective_date in skip_dates else 1.0
                else:
                    # No dataset or datasource found - assume full availability
                    sm.data_av_ratio = 1.0
            else:
                # Extended metric - average child ratios
                child_ratios: list[float] = []
                for req_symbol in sm.required_metrics:
                    req_metric = self.get(req_symbol)
                    child_ratios.append(req_metric.data_av_ratio)
                sm.data_av_ratio = sum(child_ratios) / len(child_ratios)
            if sm.data_av_ratio < 1.0:
                logger.warning(f"Low data availability ratio for {sm.symbol}: {sm.data_av_ratio:.2f}")

    def _find_cycle_details(self, remaining_metrics: list[SymbolicMetric]) -> str:
        """Generate helpful error message about circular dependencies."""
        cycle_symbols = {sm.symbol for sm in remaining_metrics}
        details = []

        for sm in remaining_metrics:
            deps_in_cycle = [str(dep) for dep in sm.required_metrics if dep in cycle_symbols]
            if deps_in_cycle:
                details.append(f"  {sm.symbol} ({sm.name}) depends on: {', '.join(deps_in_cycle)}")

        if not details:
            # No internal cycle dependencies found, might be external
            details.append("  Metrics depend on symbols not in the registry")

        return "\n".join(details)

    def symbol_lookup_table(self, key: ResultKey) -> dict[MetricKey, sp.Symbol]:
        """Create mapping from metric keys to symbol names."""
        symbol_lookup: dict[MetricKey, sp.Symbol] = {}
        for sym_metric in self.metrics:
            if sym_metric.dataset is not None:
                # Calculate effective key based on lag
                effective_key = key.lag(sym_metric.lag)
                metric_key = (sym_metric.metric_spec, effective_key, sym_metric.dataset)
                symbol_lookup[metric_key] = sym_metric.symbol
        return symbol_lookup


class RegistryMixin:
    @property
    def registry(self) -> MetricRegistry:
        """Access to metric registry."""
        raise NotImplementedError("Subclasses must implement registry property.")

    @property
    def metrics(self) -> list[SymbolicMetric]:
        """Access to all registered metrics."""
        return self.registry._metrics

    @property
    def index(self) -> SymbolIndex:
        """Access to symbol index mapping."""
        return self.registry._symbol_index

    def symbols(self) -> Iterable[sp.Symbol]:
        """Get all registered symbols."""
        return self.registry._symbol_index.keys()

    def get_symbol(self, symbol: sp.Symbol | str) -> SymbolicMetric:
        """Retrieve symbolic metric by symbol."""
        return self.registry.get(symbol)

    def remove_symbol(self, symbol: sp.Symbol) -> None:
        """Remove symbol and its dependencies recursively."""
        sm = self.get_symbol(symbol)
        for dep_symbol in sm.required_metrics:
            self.remove_symbol(dep_symbol)  # Recursive removal only
        self.registry.remove(symbol)

    def collect_symbols(self, key: ResultKey) -> list[SymbolInfo]:
        """Evaluate all symbols and return values with metadata."""
        return self.registry.collect_symbols(key)


class SymbolicMetricBase(ABC, RegistryMixin):
    def __init__(self) -> None:
        self._registry: MetricRegistry = MetricRegistry()

    @property
    def registry(self) -> MetricRegistry:
        return self._registry

    def evaluate(self, symbol: sp.Symbol, key: ResultKey) -> Result[float, str]:
        """Evaluate symbol to compute its numeric value.

        Args:
            symbol: The symbolic metric to evaluate.
            key: The ResultKey for evaluation context (date and tags).

        Returns:
            Result containing the computed value or error message.
        """
        return self.index[symbol].fn(key)

    def print_symbols(self, key: ResultKey) -> None:
        """Display all symbols in formatted table.

        Args:
            key: The ResultKey for evaluation context (date and tags)
        """
        from dqx.display import print_symbols

        symbols = self.collect_symbols(key)
        print_symbols(symbols)

    def build_deduplication_map(self, context_key: ResultKey) -> dict[sp.Symbol, sp.Symbol]:
        """Find duplicate symbols with same metric and effective date.

        Args:
            context_key: Analysis date context for calculating effective dates.

        Returns:
            Dict mapping duplicate symbols to canonical symbols.
            Canonical symbol is always the one with lowest index number.
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
        """Update symbol dependencies after deduplication.

        Args:
            substitutions: Map of duplicate symbols to canonical symbols
        """
        for sym_metric in self.metrics:
            if sym_metric.required_metrics:
                # Replace any duplicates in required_metrics
                sym_metric.required_metrics = [substitutions.get(req, req) for req in sym_metric.required_metrics]

    def prune_duplicate_symbols(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        """Remove duplicate symbols from registry.

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
        """Apply full deduplication workflow to graph and registry.

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
        """Access to parent provider."""
        return self._provider

    @property
    def registry(self) -> MetricRegistry:
        """Access to metric registry from parent provider."""
        return self._provider.registry

    @property
    def db(self) -> MetricDB:
        """Access to metric database from parent provider."""
        return self._provider._db

    @property
    def execution_id(self) -> ExecutionId:
        """The execution ID from the parent provider."""
        return self._provider.execution_id

    def day_over_day(self, metric: sp.Symbol, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        """Create day-over-day ratio metric (today/yesterday)."""
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
        cloned_spec = specs.DayOverDay.from_base_spec(spec)
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=cloned_spec.name,
                symbol=sym,
                fn=fn,
                metric_spec=cloned_spec,
                lag=lag,  # Use the provided lag instead of 0
                dataset=dataset,
                required_metrics=[lag_0, lag_1],
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym

    def week_over_week(self, metric: sp.Symbol, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        """Create week-over-week ratio metric (today/week_ago)."""
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
        cloned_spec = specs.WeekOverWeek.from_base_spec(spec)
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=cloned_spec.name,
                symbol=sym,
                fn=fn,
                metric_spec=cloned_spec,
                lag=lag,  # Use the provided lag instead of 0
                dataset=dataset,
                required_metrics=[lag_0, lag_7],
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym

    def stddev(self, metric: sp.Symbol, offset: int, n: int, dataset: str | None = None) -> sp.Symbol:
        """Create standard deviation metric over time window.

        Args:
            metric: Base metric symbol to calculate standard deviation for.
            offset: Starting position (0=today, 1=yesterday, etc.).
            n: Number of values in window (offset to offset+n-1).
            dataset: Optional dataset name.

        Returns:
            Symbol representing the standard deviation metric.
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

        # Generate symbol first
        sym = self.registry._next_symbol()

        # Create lazy function for stddev using lambda to handle the size parameter
        fn = _create_lazy_extended_fn(
            self._provider,
            lambda metric, dataset, key, execution_id, cache: compute.stddev(
                metric, dataset, key, execution_id, n, cache
            ),
            spec,
            sym,
        )

        # Register with lazy function
        cloned_spec = specs.Stddev.from_base_spec(spec, offset, n)
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=cloned_spec.name,
                symbol=sym,
                fn=fn,
                metric_spec=cloned_spec,
                lag=offset,  # stddev itself should have lag=offset (not lag=0)
                dataset=dataset,
                required_metrics=required,
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym


class MetricProvider(SymbolicMetricBase):
    def __init__(self, db: MetricDB, execution_id: ExecutionId, data_av_threshold: float) -> None:
        super().__init__()
        self._db = db
        self._execution_id = execution_id
        self._data_av_threshold = data_av_threshold

        # Create cache in provider
        from dqx.cache import MetricCache

        self._cache = MetricCache(db)

    @property
    def execution_id(self) -> ExecutionId:
        """The execution ID for this provider instance."""
        return self._execution_id

    @property
    def cache(self) -> MetricCache:
        """Access to the metric cache."""
        return self._cache

    @property
    def ext(self) -> ExtendedMetricProvider:
        """Access to extended metric provider for DoD, WoW, stddev."""
        return ExtendedMetricProvider(self)

    def clear_cache(self) -> None:
        """Clear the metric cache."""
        self._cache.clear()

    def flush_cache(self) -> int:
        """Flush dirty cache entries to DB.

        Returns:
            Number of metrics flushed
        """
        return self._cache.write_back()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {"total_cached": len(self._cache._cache), "dirty_count": self._cache.get_dirty_count()}

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
        cloned_spec = metric.clone()
        fn = _create_lazy_retrieval_fn(self, cloned_spec, sym)

        # Register with the lazy function
        self.registry._metrics.append(
            sm := SymbolicMetric(
                name=cloned_spec.name,
                symbol=sym,
                fn=fn,
                metric_spec=cloned_spec,
                lag=lag,
                dataset=dataset,
                required_metrics=[],
            )
        )

        # Update index
        self.registry.index[sym] = sm

        return sym

    def num_rows(self, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None) -> sp.Symbol:
        """Create metric counting number of rows."""
        return self.metric(specs.NumRows(parameters=parameters), lag, dataset)

    def first(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric returning first value in column."""
        return self.metric(specs.First(column, parameters=parameters), lag, dataset)

    def average(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric calculating average of column."""
        return self.metric(specs.Average(column, parameters=parameters), lag, dataset)

    def minimum(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric finding minimum value in column."""
        return self.metric(specs.Minimum(column, parameters=parameters), lag, dataset)

    def maximum(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric finding maximum value in column."""
        return self.metric(specs.Maximum(column, parameters=parameters), lag, dataset)

    def sum(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric summing values in column."""
        return self.metric(specs.Sum(column, parameters=parameters), lag, dataset)

    def null_count(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric counting null values in column."""
        return self.metric(specs.NullCount(column, parameters=parameters), lag, dataset)

    def variance(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric calculating variance of column."""
        return self.metric(specs.Variance(column, parameters=parameters), lag, dataset)

    def duplicate_count(
        self, columns: list[str], lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric counting duplicate rows by columns."""
        return self.metric(specs.DuplicateCount(columns, parameters=parameters), lag, dataset)

    @overload
    def count_values(
        self,
        column: str,
        values: bool,
        lag: int = 0,
        dataset: str | None = ...,
        parameters: dict[str, Any] | None = ...,
    ) -> sp.Symbol: ...

    @overload
    def count_values(
        self, column: str, values: int, lag: int = 0, dataset: str | None = ..., parameters: dict[str, Any] | None = ...
    ) -> sp.Symbol: ...

    @overload
    def count_values(
        self, column: str, values: str, lag: int = 0, dataset: str | None = ..., parameters: dict[str, Any] | None = ...
    ) -> sp.Symbol: ...

    @overload
    def count_values(
        self,
        column: str,
        values: list[int],
        lag: int = 0,
        dataset: str | None = ...,
        parameters: dict[str, Any] | None = ...,
    ) -> sp.Symbol: ...

    @overload
    def count_values(
        self,
        column: str,
        values: list[str],
        lag: int = 0,
        dataset: str | None = ...,
        parameters: dict[str, Any] | None = ...,
    ) -> sp.Symbol: ...

    def count_values(
        self,
        column: str,
        values: int | str | bool | list[int] | list[str],
        lag: int = 0,
        dataset: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> sp.Symbol:
        """Count occurrences of specific values in column.

        Args:
            column: Column name to count values in
            values: Value(s) to count - single value or list
            lag: Lag offset in days
            dataset: Optional dataset name
            parameters: Additional parameters to pass to the metric

        Returns:
            Symbol representing the count
        """
        return self.metric(specs.CountValues(column, values, parameters=parameters), lag, dataset)

    def unique_count(
        self, column: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Count distinct non-null values in column.

        Note: Cannot be merged across partitions.

        Args:
            column: Column name to count unique values in
            lag: Lag offset in days
            dataset: Optional dataset name
            parameters: Additional parameters to pass to the metric

        Returns:
            Symbol representing the unique count
        """
        return self.metric(specs.UniqueCount(column, parameters=parameters), lag, dataset)

    def custom_sql(
        self, sql_expression: str, lag: int = 0, dataset: str | None = None, parameters: dict[str, Any] | None = None
    ) -> sp.Symbol:
        """Create metric using custom SQL expression.

        Args:
            sql_expression: SQL expression to evaluate
            lag: Lag offset in days
            dataset: Optional dataset name
            parameters: Additional parameters to pass to the metric

        Returns:
            Symbol representing the custom SQL metric
        """
        return self.metric(specs.CustomSQL(sql_expression, parameters=parameters), lag, dataset)

    def get_metric(
        self, metric_spec: MetricSpec, result_key: ResultKey, dataset: str, execution_id: ExecutionId
    ) -> Result[Metric, str]:
        """Get metric from cache or database."""
        # Try cache first
        cache_result = self._cache.get((metric_spec, result_key, dataset, execution_id))
        match cache_result:
            case Some(metric):
                # Wrap in Result for return type compatibility
                from returns.result import Success

                return Success(metric)
            case _:
                # Cache miss - continue to DB lookup
                pass

        # Cache miss - get from DB
        metrics = self._db.get_by_execution_id(execution_id)
        for metric in metrics:
            if metric.spec == metric_spec and metric.key == result_key and metric.dataset == dataset:
                # Populate cache before returning
                self._cache.put(metric)
                # Wrap in Result for return type compatibility
                from returns.result import Success

                return Success(metric)

        # Return failure instead of Nothing
        return Failure(f"Metric not found: {metric_spec.name} for {dataset}")

    def persist(self, metrics: list[Metric]) -> None:
        """Save metrics to database and update cache."""
        # Persist to DB
        self._db.persist(metrics)

        # Update cache
        self._cache.put(metrics)

    def get_metrics_by_execution_id(self, execution_id: ExecutionId) -> list[Metric]:
        """Get all metrics for execution ID using cache."""
        # First check if we can get all metrics from cache
        # This is a simple implementation - in production you might want
        # to track which execution_ids are fully cached
        metrics_from_db = list(self._db.get_by_execution_id(execution_id))

        # Try to get each metric from cache first
        result_metrics = []
        for metric in metrics_from_db:
            cache_key = (metric.spec, metric.key, metric.dataset, execution_id)
            cache_result = self._cache.get(cache_key)
            match cache_result:
                case Some(cached_metric):
                    result_metrics.append(cached_metric)
                case _:
                    # Not in cache, add it
                    self._cache.put(metric)
                    result_metrics.append(metric)

        return result_metrics
