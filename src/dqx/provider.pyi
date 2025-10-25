from __future__ import annotations

import datetime
from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import Lock
from typing import overload

import sympy as sp
from returns.result import Result

from dqx.common import ResultKey, ResultKeyProvider, RetrievalFn, Tags
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

SymbolIndex = dict[sp.Symbol, SymbolicMetric]

@dataclass
class SymbolInfo:
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
    key_provider: ResultKeyProvider
    metric_spec: MetricSpec
    dataset: str | None = None
    parent_symbol: sp.Symbol | None = None

class SymbolicMetricBase(ABC):
    _metrics: list[SymbolicMetric]
    _symbol_index: SymbolIndex
    _curr_index: int
    _mutex: Lock
    _children_map: defaultdict[sp.Symbol, list[sp.Symbol]]

    def __init__(self) -> None: ...
    @property
    def symbolic_metrics(self) -> list[SymbolicMetric]: ...
    def symbols(self) -> Iterable[sp.Symbol]: ...
    def get_symbol(self, symbol: sp.Symbol | str) -> SymbolicMetric: ...
    def _next_symbol(self, prefix: str = "x") -> sp.Symbol: ...
    def _register(
        self,
        symbol: sp.Symbol,
        name: str,
        fn: RetrievalFn,
        key: ResultKeyProvider,
        metric_spec: MetricSpec,
        dataset: str | None = None,
        parent: sp.Symbol | None = None,
    ) -> None: ...
    def evaluate(self, symbol: sp.Symbol, key: ResultKey) -> Result[float, str]: ...
    def get_children(self, symbol: sp.Symbol) -> list[sp.Symbol]: ...
    def collect_symbols(self, key: ResultKey) -> list[SymbolInfo]: ...
    def print_symbols(self, key: ResultKey) -> None: ...

class ExtendedMetricProvider:
    _provider: MetricProvider
    _db: MetricDB
    _next_symbol: Callable[[str], sp.Symbol]
    _register: Callable[..., None]

    def __init__(self, provider: MetricProvider) -> None: ...
    def _resolve_metric_spec(self, metric: sp.Symbol) -> MetricSpec: ...
    def day_over_day(
        self, metric: sp.Symbol, key: ResultKeyProvider = ..., dataset: str | None = None
    ) -> sp.Symbol: ...
    def stddev(
        self,
        metric: sp.Symbol,
        lag: int,
        n: int,
        key: ResultKeyProvider = ...,
        dataset: str | None = None,
    ) -> sp.Symbol: ...
    def week_over_week(
        self, metric: sp.Symbol, key: ResultKeyProvider = ..., dataset: str | None = None
    ) -> sp.Symbol: ...

class MetricProvider(SymbolicMetricBase):
    _db: MetricDB

    def __init__(self, db: MetricDB) -> None: ...
    @property
    def ext(self) -> ExtendedMetricProvider: ...
    def metric(self, metric: MetricSpec, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def num_rows(self, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def first(self, column: str, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def average(self, column: str, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def minimum(self, column: str, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def maximum(self, column: str, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def sum(self, column: str, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def null_count(self, column: str, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def variance(self, column: str, key: ResultKeyProvider = ..., dataset: str | None = None) -> sp.Symbol: ...
    def duplicate_count(
        self, columns: list[str], key: ResultKeyProvider = ..., dataset: str | None = None
    ) -> sp.Symbol: ...
    @overload
    def count_values(
        self, column: str, values: bool, key: ResultKeyProvider = ..., dataset: str | None = ...
    ) -> sp.Symbol: ...
    @overload
    def count_values(
        self, column: str, values: int, key: ResultKeyProvider = ..., dataset: str | None = ...
    ) -> sp.Symbol: ...
    @overload
    def count_values(
        self, column: str, values: str, key: ResultKeyProvider = ..., dataset: str | None = ...
    ) -> sp.Symbol: ...
    @overload
    def count_values(
        self, column: str, values: list[int], key: ResultKeyProvider = ..., dataset: str | None = ...
    ) -> sp.Symbol: ...
    @overload
    def count_values(
        self, column: str, values: list[str], key: ResultKeyProvider = ..., dataset: str | None = ...
    ) -> sp.Symbol: ...
