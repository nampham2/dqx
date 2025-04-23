from __future__ import annotations

import datetime
import datetime as dt
import random
import string
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, Type, runtime_checkable

import duckdb
import sympy as sp
from returns.result import Result

if TYPE_CHECKING:
    from dqx.api import SymbolicAssert
    from dqx.specs import MetricSpec


class DQXError(Exception): ...


TimeSeries = Mapping[dt.date, float]
Tags = dict[str, Any]

SeverityLevel = Literal["P0", "P1", "P2", "P3"]
Parameters = dict[str, Any]


class ValidationCode(StrEnum):
    OK = "ok"
    FAILED = "failed"
    ERROR = "error"


@dataclass(frozen=True)
class ResultKey:
    yyyy_mm_dd: datetime.date
    tags: Tags

    def lag(self, n: int) -> ResultKey:
        return ResultKey(
            yyyy_mm_dd=self.yyyy_mm_dd - datetime.timedelta(days=n),
            tags=self.tags,
        )

    def range(self, lag: int, window: int) -> tuple[dt.date, dt.date]:
        """
        Calculate a date range (inclusive) based on the given lag and window.

        Args:
            lag (int): The number of days to lag.
            window (int): The size of the window in days.

        Returns:
            tuple[dt.date, dt.date]: A tuple containing the start and end dates (inclusive) of the range.
        """
        return (
            self.yyyy_mm_dd - datetime.timedelta(days=lag + window - 1),
            self.yyyy_mm_dd - datetime.timedelta(days=lag),
        )

    def __hash__(self) -> int:
        return hash((self.yyyy_mm_dd, tuple(self.tags)))

    def __repr__(self) -> str:
        return f"ResultKey({self.yyyy_mm_dd.isoformat()}, {self.tags})"

    def __str__(self) -> str:
        return self.__repr__()


class ResultKeyProvider:
    def __init__(self) -> None:
        self._lag: int = 0

    def lag(self, n: int) -> Self:
        self._lag = n
        return self

    def create(self, nominal_key: ResultKey) -> ResultKey:
        return nominal_key.lag(self._lag)


@runtime_checkable
class DuckDataSource(Protocol):
    """Adapt a data source to a duckdb table name"""

    name: str
    analyzer_class: Type

    @property
    def cte(self) -> str: ...

    def query(self, query: str) -> duckdb.DuckDBPyRelation: ...


@runtime_checkable
class DuckBatchDataSource(Protocol):
    def arrow_ds(self) -> Iterable[DuckDataSource]: ...


def random_prefix(k: int = 6) -> str:
    """
    Generate a random table name consisting of lowercase ASCII letters.

    Args:
        k (int): The length of the random string to generate. Default is 6.

    Returns:
        str: A string starting with an underscore followed by a random sequence of lowercase ASCII letters.
    """
    return "_" + "".join(random.choices(string.ascii_lowercase, k=k))


Validator = Callable[[Any], bool]


@dataclass
class SymbolicValidator:
    name: str
    fn: Validator | None = None


RetrievalFn = Callable[[ResultKey], Result[float, str]]


@runtime_checkable
class ExtendedMetricProvider(Protocol):
    def day_over_day(self, metric: MetricSpec, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def stddev(
        self, metric: MetricSpec, lag: int, n: int, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...


@runtime_checkable
class MetricProvider(Protocol):
    @property
    def ext(self) -> ExtendedMetricProvider: ...

    def num_rows(self, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def first(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def average(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def minimum(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def maximum(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def sum(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def null_count(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def variance(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...

    def approx_cardinality(self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None) -> sp.Symbol: ...


@runtime_checkable
class Context(Protocol):
    def assert_that(self, expr: sp.Expr) -> SymbolicAssert: ...

    @property
    def key(self) -> ResultKeyProvider: ...
