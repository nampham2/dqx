from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from threading import Lock

import sympy as sp
from returns.result import Result

from dqx import compute, specs
from dqx.common import DQXError, ResultKey, ResultKeyProvider, RetrievalFn
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

SymbolIndex = dict[sp.Symbol, "SymbolicMetric"]


@dataclass
class SymbolicMetric:
    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    key_provider: ResultKeyProvider
    metric_spec: MetricSpec
    dataset: str | None = None
    parent_symbol: sp.Symbol | None = None


class SymbolicMetricBase(ABC):
    def __init__(self) -> None:
        self._metrics: list[SymbolicMetric] = []
        self._symbol_index: SymbolIndex = {}
        self._curr_index: int = 0
        self._mutex = Lock()
        self._children_map: dict[sp.Symbol, list[sp.Symbol]] = {}

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
        key: ResultKeyProvider,
        metric_spec: MetricSpec,
        dataset: str | None = None,
        parent: sp.Symbol | None = None,
    ) -> None:
        """Register a symbolic metric and track parent-child relationships."""
        metric = SymbolicMetric(
            name=name,
            symbol=symbol,
            fn=fn,
            key_provider=key,
            metric_spec=metric_spec,
            dataset=dataset,
            parent_symbol=parent,
        )
        self._metrics.append(metric)
        self._symbol_index[symbol] = metric

        # Update children map
        if parent is not None:
            if parent not in self._children_map:
                self._children_map[parent] = []
            self._children_map[parent].append(symbol)

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


class ExtendedMetricProvider:
    """A provider for derivative metrics that builds on top of primitive metrics."""

    def __init__(self, provider: MetricProvider) -> None:
        self._provider = provider

        # Mapping provider utilities here for easier access
        self._db = self._provider._db
        self._next_symbol = self._provider._next_symbol
        self._register = self._provider._register

    def _resolve_metric_spec(self, metric: sp.Symbol) -> MetricSpec:
        """Resolve a symbol to its underlying MetricSpec."""
        symbolic_metric = self._provider.get_symbol(metric)
        return symbolic_metric.metric_spec

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
        metric_spec = self._resolve_metric_spec(base_metric)
        base_metric_info = self._provider.get_symbol(base_metric)

        # Create lag function that applies the lag to the key
        def lag_metric(
            db: MetricDB, metric: MetricSpec, lag: int, key_provider: ResultKeyProvider, nominal_key: ResultKey
        ) -> Result[float, str]:
            lagged_key = nominal_key.lag(lag)
            key = key_provider.create(lagged_key)
            value = db.get_metric_value(metric, key)
            from returns.converters import maybe_to_result

            return maybe_to_result(value, f"Metric {metric.name} not found for lagged date!")

        self._provider._register(
            sym := self._next_symbol(),
            name=f"lag({lag_days})({base_metric})",
            fn=partial(lag_metric, self._db, metric_spec, lag_days, base_metric_info.key_provider),
            key=base_metric_info.key_provider,
            metric_spec=metric_spec,
            dataset=base_metric_info.dataset,
            parent=parent_metric,  # Now the parent is the derived metric
        )
        return sym

    def day_over_day(
        self, metric: sp.Symbol, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        metric_spec = self._resolve_metric_spec(metric)

        # First register the day_over_day metric
        self._provider._register(
            sym := self._next_symbol(),
            name=f"day_over_day({metric_spec.name})",
            fn=partial(compute.day_over_day, self._db, metric_spec, key),
            key=key,
            metric_spec=metric_spec,
            dataset=dataset,
            parent=None,  # day_over_day is now the parent
        )

        # Create lag(1) dependency with day_over_day as parent
        self._create_lag_dependency(metric, 1, parent_metric=sym)

        # Register base metric as child of day_over_day
        if sym not in self._provider._children_map:
            self._provider._children_map[sym] = []
        if metric not in self._provider._children_map[sym]:
            self._provider._children_map[sym].append(metric)

        return sym

    def stddev(
        self,
        metric: sp.Symbol,
        lag: int,
        n: int,
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None,
    ) -> sp.Symbol:
        metric_spec = self._resolve_metric_spec(metric)

        # First register the stddev metric
        self._provider._register(
            sym := self._next_symbol(),
            name=f"stddev({metric_spec.name})",
            fn=partial(compute.stddev, self._db, metric_spec, lag, n, key),
            key=key,
            metric_spec=metric_spec,
            dataset=dataset,
            parent=None,  # stddev is now the parent
        )

        # Create lag dependencies for the window with stddev as parent
        # stddev needs lag values from lag to lag+n-1
        for i in range(lag, lag + n):
            self._create_lag_dependency(metric, i, parent_metric=sym)

        # Register base metric as child of stddev
        if sym not in self._provider._children_map:
            self._provider._children_map[sym] = []
        if metric not in self._provider._children_map[sym]:
            self._provider._children_map[sym].append(metric)

        return sym

    def week_over_week(
        self, metric: sp.Symbol, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        metric_spec = self._resolve_metric_spec(metric)

        # First register the week_over_week metric
        self._provider._register(
            sym := self._next_symbol(),
            name=f"week_over_week({metric_spec.name})",
            fn=partial(compute.week_over_week, self._db, metric_spec, key),
            key=key,
            metric_spec=metric_spec,
            dataset=dataset,
            parent=None,  # week_over_week is now the parent
        )

        # Create lag(7) dependency with week_over_week as parent
        self._create_lag_dependency(metric, 7, parent_metric=sym)

        # Register base metric as child of week_over_week
        if sym not in self._provider._children_map:
            self._provider._children_map[sym] = []
        if metric not in self._provider._children_map[sym]:
            self._provider._children_map[sym].append(metric)

        return sym


class MetricProvider(SymbolicMetricBase):
    def __init__(self, db: MetricDB) -> None:
        super().__init__()
        self._db = db

    @property
    def ext(self) -> ExtendedMetricProvider:
        return ExtendedMetricProvider(self)

    def metric(
        self,
        metric: MetricSpec,
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None,
        parent: sp.Symbol | None = None,
    ) -> sp.Symbol:
        self._register(
            sym := self._next_symbol(),
            name=metric.name,
            fn=partial(compute.simple_metric, self._db, metric, key),
            key=key,
            metric_spec=metric,
            dataset=dataset,
            parent=parent,
        )
        return sym

    def num_rows(self, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.NumRows(), key, dataset)

    def first(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.First(column), key, dataset)

    def average(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Average(column), key, dataset)

    def minimum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Minimum(column), key, dataset)

    def maximum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Maximum(column), key, dataset)

    def sum(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None) -> sp.Symbol:
        return self.metric(specs.Sum(column), key, dataset)

    def null_count(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        return self.metric(specs.NullCount(column), key, dataset)

    def variance(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Variance(column), key, dataset)

    def duplicate_count(
        self, columns: list[str], key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        return self.metric(specs.DuplicateCount(columns), key, dataset)

    def count_values(
        self,
        column: str,
        values: int | str | bool | list[int] | list[str],
        key: ResultKeyProvider = ResultKeyProvider(),
        dataset: str | None = None,
    ) -> sp.Symbol:
        """Count occurrences of specific value(s) in a column.

        This operation counts only the specified values, never NULLs.
        Empty strings are counted as values, not as NULLs.

        Args:
            column: Column name to count values in
            values: Value(s) to count - single int/str/bool or list of int/str
            key: Result key provider
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
        return self.metric(specs.CountValues(column, values), key, dataset)
