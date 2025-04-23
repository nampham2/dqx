from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from threading import Lock

import sympy as sp
from returns.result import Result

from dqx import compute, specs
from dqx.common import DQXError, ResultKey, ResultKeyProvider, RetrievalFn
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

# Metric retrieval function
Dependency = tuple[MetricSpec, ResultKeyProvider]
SymbolIndex = dict[sp.Symbol, "SymbolicMetric"]


@dataclass
class SymbolicMetric:
    name: str
    symbol: sp.Symbol
    fn: RetrievalFn
    key_provider: ResultKeyProvider
    dependencies: list[Dependency]
    datasets: list[str] = field(default_factory=list)


class SymbolicMetricBase(ABC):
    def __init__(self) -> None:
        self._symbols: list[SymbolicMetric] = []
        self._symbol_index: SymbolIndex = {}
        self._curr_index: int = 0
        self._mutex = Lock()

    def symbols(self) -> Iterable[sp.Symbol]:
        return self._symbol_index.keys()

    def get_symbol(self, symbol: sp.Symbol) -> SymbolicMetric:
        """Find the first symbol data that matches the given symbol."""
        first_or_none = next(filter(lambda s: s.symbol == symbol, self._symbols), None)
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
        dependencies: list[Dependency],
        datasets: list[str],
    ) -> None:
        """Register the map between a symbol and a metric

        Args:
            symbol (sp.Symbol): symbol
            metric (Metric): metric
        """

        self._symbols.append(sm := SymbolicMetric(name, symbol, fn, key, dependencies, datasets))
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

    def day_over_day(
        self, metric: MetricSpec, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        self._provider._register(
            sym := self._next_symbol(),
            name=f"day_over_day({metric.name})",
            fn=partial(compute.day_over_day, self._db, metric, key),
            key=key,
            dependencies=[(metric, key)],
            datasets=datasets or [],
        )
        return sym

    def stddev(
        self,
        metric: MetricSpec,
        lag: int,
        n: int,
        key: ResultKeyProvider = ResultKeyProvider(),
        datasets: list[str] | None = None,
    ) -> sp.Symbol:
        self._provider._register(
            sym := self._next_symbol(),
            name=f"stddev({metric.name})",
            fn=partial(compute.stddev, self._db, metric, lag, n, key),
            key=key,
            dependencies=[(metric, key)],
            datasets=datasets or [],
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
        self, metric: MetricSpec, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        self._register(
            sym := self._next_symbol(),
            name=metric.name,
            fn=partial(compute.simple_metric, self._db, metric, key),
            key=key,
            dependencies=[(metric, key)],
            datasets=datasets or [],
        )
        return sym

    def num_rows(self, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol:
        return self.metric(specs.NumRows(), key, datasets)

    def first(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.First(column), key, datasets)

    def average(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Average(column), key, datasets)

    def minimum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Minimum(column), key, datasets)

    def maximum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Maximum(column), key, datasets)

    def sum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Sum(column), key, datasets)

    def null_count(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.NullCount(column), key, datasets)

    def variance(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.Variance(column), key, datasets)

    def approx_cardinality(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol:
        return self.metric(specs.ApproxCardinality(column), key, datasets)
