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


class SymbolicMetricBase(ABC):
    def __init__(self) -> None:
        self._metrics: list[SymbolicMetric] = []
        self._symbol_index: SymbolIndex = {}
        self._curr_index: int = 0
        self._mutex = Lock()

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
    ) -> None:
        self._metrics.append(sm := SymbolicMetric(name, symbol, fn, key, metric_spec, dataset))
        self._symbol_index[symbol] = sm

    def evaluate(self, symbol: sp.Symbol, key: ResultKey) -> Result[float, str]:
        return self._symbol_index[symbol].fn(key)


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

    def day_over_day(
        self, metric: sp.Symbol, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        metric_spec = self._resolve_metric_spec(metric)
        self._provider._register(
            sym := self._next_symbol(),
            name=f"day_over_day({metric_spec.name})",
            fn=partial(compute.day_over_day, self._db, metric_spec, key),
            key=key,
            metric_spec=metric_spec,
            dataset=dataset,
        )
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
        self._provider._register(
            sym := self._next_symbol(),
            name=f"stddev({metric_spec.name})",
            fn=partial(compute.stddev, self._db, metric_spec, lag, n, key),
            key=key,
            metric_spec=metric_spec,
            dataset=dataset,
        )
        return sym

    def week_over_week(
        self, metric: sp.Symbol, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        metric_spec = self._resolve_metric_spec(metric)
        self._provider._register(
            sym := self._next_symbol(),
            name=f"week_over_week({metric_spec.name})",
            fn=partial(compute.week_over_week, self._db, metric_spec, key),
            key=key,
            metric_spec=metric_spec,
            dataset=dataset,
        )
        return sym


class MetricProvider(SymbolicMetricBase):
    def __init__(self, db: MetricDB) -> None:
        super().__init__()
        self._db = db

    @property
    def ext(self) -> ExtendedMetricProvider:
        return ExtendedMetricProvider(self)

    def metric(
        self, metric: MetricSpec, key: ResultKeyProvider = ResultKeyProvider(), dataset: str | None = None
    ) -> sp.Symbol:
        self._register(
            sym := self._next_symbol(),
            name=metric.name,
            fn=partial(compute.simple_metric, self._db, metric, key),
            key=key,
            metric_spec=metric,
            dataset=dataset,
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
