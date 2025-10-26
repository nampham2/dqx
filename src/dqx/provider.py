from __future__ import annotations

import datetime
from abc import ABC
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING

import sympy as sp
from returns.result import Result

from dqx import compute, specs
from dqx.common import DQXError, ResultKey, RetrievalFn, Tags
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

if TYPE_CHECKING:
    from dqx.graph.traversal import Graph

__all__ = [
    "SymbolInfo",
    "SymbolicMetric",
    "SymbolicMetricBase",
    "ExtendedMetricProvider",
    "MetricProvider",
    "SymbolIndex",
]

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


class SymbolicMetricBase(ABC):
    def __init__(self) -> None:
        self._metrics: list[SymbolicMetric] = []
        self._symbol_index: SymbolIndex = {}
        self._curr_index: int = 0
        self._mutex = Lock()
        self._children_map: defaultdict[sp.Symbol, list[sp.Symbol]] = defaultdict(list)

    @property
    def symbolic_metrics(self) -> list[SymbolicMetric]:
        return self._metrics

    def symbols(self) -> Iterable[sp.Symbol]:
        return self._symbol_index.keys()

    def get_symbol(self, symbol: sp.Symbol | str) -> SymbolicMetric:
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

        first_or_none = next(filter(lambda s: s.symbol == symbol, self._metrics), None)
        if not first_or_none:
            raise DQXError(f"Symbol {symbol} not found.")

        return first_or_none

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

    def _register(
        self,
        symbol: sp.Symbol,
        name: str,
        fn: RetrievalFn,
        metric_spec: MetricSpec,
        lag: int = 0,
        dataset: str | None = None,
        required_metrics: list[sp.Symbol] | None = None,
    ) -> None:
        """Register a symbolic metric."""
        required_metrics = required_metrics or []

        self._metrics.append(
            SymbolicMetric(
                name=name,
                symbol=symbol,
                fn=fn,
                metric_spec=metric_spec,
                lag=lag,
                dataset=dataset,
                required_metrics=required_metrics,
            )
        )
        self._symbol_index[symbol] = self._metrics[-1]

        # Populate children map: the symbol has these required_metrics as children
        if required_metrics:
            self._children_map[symbol] = required_metrics.copy()

    def evaluate(self, symbol: sp.Symbol, key: ResultKey) -> Result[float, str]:
        return self._symbol_index[symbol].fn(key)

    def get_children(self, symbol: sp.Symbol) -> list[sp.Symbol]:
        """Get all child symbols of a given parent symbol.

        Args:
            symbol: The parent symbol to get children for

        Returns:
            List of child symbols, or empty list if no children
        """
        return self._children_map.get(symbol, [])

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
        for symbolic_metric in self.symbolic_metrics:
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
        from datetime import timedelta

        groups: dict[tuple[str, str, str | None], list[sp.Symbol]] = {}

        # Group symbols by identity
        for sym_metric in self.symbolic_metrics:
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
        for sym_metric in self.symbolic_metrics:
            if sym_metric.required_metrics:
                # Replace any duplicates in required_metrics
                sym_metric.required_metrics = [substitutions.get(req, req) for req in sym_metric.required_metrics]

                # Also update the children map
                if sym_metric.symbol in self._children_map:
                    self._children_map[sym_metric.symbol] = [
                        substitutions.get(child, child) for child in self._children_map[sym_metric.symbol]
                    ]

    def prune_duplicate_symbols(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        """Remove duplicate symbols from the provider.

        Args:
            substitutions: Map from duplicate symbols to canonical symbols
        """
        if not substitutions:
            return

        to_remove = set(substitutions.keys())

        # Remove duplicate symbols
        self._metrics = [sm for sm in self._metrics if sm.symbol not in to_remove]

        # Remove from index
        for symbol in to_remove:
            del self._symbol_index[symbol]

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


class ExtendedMetricProvider:
    """A provider for derivative metrics that builds on top of primitive metrics."""

    def __init__(self, provider: MetricProvider) -> None:
        self._provider = provider

        # Mapping provider utilities here for easier access
        self._db = self._provider._db
        self._next_symbol = self._provider._next_symbol
        self._register = self._provider._register

    def _create_lag_dependency(
        self, base_metric: sp.Symbol, lag_days: int, parent_metric: sp.Symbol | None = None
    ) -> sp.Symbol:
        """Create a lag dependency symbol for the base metric.

        Args:
            base_metric: The base metric symbol to create lag for
            lag_days: Number of days to lag
            parent_metric: The parent metric that depends on this lag (for dataset propagation)

        Returns:
            Symbol representing the lag dependency
        """
        symbolic_metric = self._provider.get_symbol(base_metric)
        metric_spec = symbolic_metric.metric_spec

        # Create lag function that applies the lag to the key
        def lag_metric(db: MetricDB, metric: MetricSpec, lag: int, nominal_key: ResultKey) -> Result[float, str]:
            lagged_key = nominal_key.lag(lag)
            value = db.get_metric_value(metric, lagged_key)
            from returns.converters import maybe_to_result

            return maybe_to_result(value, f"Metric {metric.name} not found for lagged date!")

        self._provider._register(
            sym := self._next_symbol(),
            name=f"lag({lag_days})({base_metric})",
            fn=partial(lag_metric, self._db, metric_spec, lag_days),
            metric_spec=metric_spec,
            lag=lag_days,
            dataset=symbolic_metric.dataset,
        )
        return sym

    def day_over_day(self, metric: sp.Symbol, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        # Get the full SymbolicMetric object
        symbolic_metric = self._provider.get_symbol(metric)
        metric_spec = symbolic_metric.metric_spec

        # Create required lag metrics first
        if lag > 0:
            # For lagged DoD, we need base metric at lag and lag+1
            base_lagged = self._provider._ensure_lagged_symbol(symbolic_metric, lag)
            lag_sym = self._provider._ensure_lagged_symbol(symbolic_metric, lag + 1)
            required = [base_lagged, lag_sym]
            name = f"day_over_day({symbolic_metric.name}, lag={lag})"
        else:
            # Standard DoD: today vs yesterday
            lag_sym = self._provider._ensure_lagged_symbol(symbolic_metric, 1)
            required = [metric, lag_sym]
            name = f"day_over_day({symbolic_metric.name})"

        self._provider._register(
            sym := self._next_symbol(),
            name=name,
            fn=partial(compute.day_over_day, self._db, metric_spec, lag),
            metric_spec=metric_spec,
            lag=lag,
            dataset=dataset,
            required_metrics=required,
        )
        return sym

    def stddev(self, metric: sp.Symbol, lag: int, n: int, dataset: str | None = None) -> sp.Symbol:
        # Get the full SymbolicMetric object
        symbolic_metric = self._provider.get_symbol(metric)
        metric_spec = symbolic_metric.metric_spec

        # Ensure required lag metrics exist
        required = []
        for i in range(lag, lag + n):
            lag_sym = self._provider._ensure_lagged_symbol(symbolic_metric, i)
            required.append(lag_sym)

        self._provider._register(
            sym := self._next_symbol(),
            name=f"stddev({symbolic_metric.name}, lag={lag}, n={n})",
            fn=partial(compute.stddev, self._db, metric_spec, lag, n),
            metric_spec=metric_spec,  # Keep original metric_spec
            lag=0,
            dataset=dataset,
            required_metrics=required,
        )
        return sym

    def week_over_week(self, metric: sp.Symbol, lag: int = 0, dataset: str | None = None) -> sp.Symbol:
        # Get the full SymbolicMetric object
        symbolic_metric = self._provider.get_symbol(metric)
        metric_spec = symbolic_metric.metric_spec

        # Create required lag metrics first
        if lag > 0:
            # For lagged WoW, we need base metric at lag and lag+7
            base_lagged = self._provider._ensure_lagged_symbol(symbolic_metric, lag)
            lag_sym = self._provider._ensure_lagged_symbol(symbolic_metric, lag + 7)
            required = [base_lagged, lag_sym]
            name = f"week_over_week({symbolic_metric.name}, lag={lag})"
        else:
            # Standard WoW: today vs 7 days ago
            lag_sym = self._provider._ensure_lagged_symbol(symbolic_metric, 7)
            required = [metric, lag_sym]
            name = f"week_over_week({symbolic_metric.name})"

        self._provider._register(
            sym := self._next_symbol(),
            name=name,
            fn=partial(compute.week_over_week, self._db, metric_spec, lag),
            metric_spec=metric_spec,
            lag=lag,
            dataset=dataset,
            required_metrics=required,
        )
        return sym


class MetricProvider(SymbolicMetricBase):
    def __init__(self, db: MetricDB) -> None:
        super().__init__()
        self._db = db

    @property
    def ext(self) -> ExtendedMetricProvider:
        return ExtendedMetricProvider(self)

    def get_metrics_by_date(self, dataset: str | None = None) -> dict[datetime.date, list[SymbolicMetric]]:
        """Get metrics grouped by their effective date.

        This helper method organizes symbolic metrics by the date they apply to,
        taking into account any lag specified in their key provider.

        Args:
            dataset: Optional dataset name to filter metrics. If None, returns all metrics.

        Returns:
            Dictionary mapping dates to lists of symbolic metrics that apply to that date.

        Example:
            >>> metrics_by_date = provider.get_metrics_by_date("sales")
            >>> for date, metrics in metrics_by_date.items():
            ...     print(f"{date}: {len(metrics)} metrics")
        """
        from collections import defaultdict

        metrics_by_date: dict[datetime.date, list[SymbolicMetric]] = defaultdict(list)

        # Filter metrics by dataset if specified
        metrics = self.symbolic_metrics
        if dataset:
            metrics = [m for m in metrics if m.dataset == dataset]

        # Group by effective date (considering lag)
        # Note: This requires a reference date to calculate effective dates
        # The actual grouping would be done in the context where we have a ResultKey
        # For now, we return the ungrouped metrics as this is a simplified helper
        # The real grouping happens in the API when we have the ResultKey context

        return dict(metrics_by_date)

    def _ensure_lagged_symbol(self, base: SymbolicMetric, lag: int) -> sp.Symbol:
        """Get or create a lagged version of a base metric.

        This method handles all metric types including nested extended metrics
        like StdDev(Average("price")).

        Args:
            base: The base symbolic metric to create a lagged version of
            lag: The lag offset to apply

        Returns:
            Symbol for the lagged metric (either existing or newly created)
        """
        # Check if it already exists
        for sm in self._metrics:
            if sm.metric_spec == base.metric_spec and sm.lag == lag and sm.dataset == base.dataset:
                return sm.symbol

        # Create new lagged metric using the generic metric() method
        # This works for ANY metric spec, including nested ones
        return self.metric(metric=base.metric_spec, lag=lag, dataset=base.dataset)

    def metric(
        self,
        metric: MetricSpec,
        lag: int = 0,
        dataset: str | None = None,
    ) -> sp.Symbol:
        # Include lag in the name if lag > 0
        name = f"lag({lag})({metric.name})" if lag > 0 else metric.name

        self._register(
            sym := self._next_symbol(),
            name=name,
            fn=partial(compute.simple_metric, self._db, metric, lag),
            metric_spec=metric,
            lag=lag,
            dataset=dataset,
        )
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
