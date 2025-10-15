from __future__ import annotations

import datetime
import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

import duckdb
import sympy as sp
from returns.result import Result

if TYPE_CHECKING:
    from dqx.analyzer import AnalysisReport
    from dqx.api import AssertionDraft
    from dqx.specs import MetricSpec


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
        suite: Name of the verification suite that evaluated this symbol
        tags: Additional metadata from ResultKey (e.g., {"env": "prod"})
    """

    name: str
    metric: str
    dataset: str | None
    value: Result[float, str]
    yyyy_mm_dd: datetime.date
    suite: str
    tags: Tags = field(default_factory=dict)


@dataclass
class EvaluationFailure:
    """Failure information for evaluation errors"""

    error_message: str  # Overall error message
    expression: str  # The symbolic expression
    symbols: list[SymbolInfo]  # List of symbol information


@dataclass
class AssertionResult:
    """Result of a single assertion evaluation.

    This dataclass captures the complete state of an assertion after evaluation,
    including its location in the hierarchy (suite/check/assertion), the actual
    result value, and any error information if the assertion failed.

    Attributes:
        yyyy_mm_dd: Date from the ResultKey used during evaluation
        suite: Name of the verification suite
        check: Name of the parent check
        assertion: Name of the assertion (always present, names are mandatory)
        severity: Priority level (P0, P1, P2, P3)
        status: Validation result ("OK" or "FAILURE")
        metric: The metric computation result (Success with value or Failure with errors)
        expression: Full validation expression (e.g., "average(price) > 0")
        tags: Tags from the ResultKey (e.g., {"env": "prod"})
    """

    yyyy_mm_dd: datetime.date
    suite: str
    check: str
    assertion: str
    severity: SeverityLevel
    status: AssertionStatus  # Uses the type we defined in Task 1
    metric: Result[float, list[EvaluationFailure]]  # The metric computation result
    expression: str | None = None
    tags: Tags = field(default_factory=dict)


class DQXError(Exception): ...


TimeSeries = Mapping[dt.date, float]
Tags = dict[str, Any]

SeverityLevel = Literal["P0", "P1", "P2", "P3"]
RecomputeStrategy = Literal["ALWAYS", "MISSING", "NEVER"]
Parameters = dict[str, Any]
AssertionStatus = Literal["OK", "FAILURE"]


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
class SqlDataSource(Protocol):
    """
    Protocol for SQL data sources that can be analyzed by DQX.

    This protocol defines the interface for adapting various data sources
    (e.g., Arrow tables, BigQuery, DuckDB tables) to work with the DQX
    analysis framework.

    Attributes:
        name: A unique identifier for this data source instance
        dialect: The SQL dialect name used for query generation (e.g., 'duckdb', 'postgresql')
    """

    name: str
    dialect: str

    @property
    def cte(self) -> str:
        """
        Return a Common Table Expression (CTE) SQL fragment for this data source.

        This should return valid SQL that can be used in a WITH clause,
        typically selecting from the underlying data source.

        Returns:
            str: SQL CTE fragment (without the 'AS' keyword or parentheses)

        Example:
            "SELECT * FROM my_table WHERE date = '2023-01-01'"
        """
        ...

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """
        Execute a SQL query against this data source.

        Args:
            query: SQL query string to execute. The query should reference
                  the data source using the CTE format.

        Returns:
            DuckDBPyRelation: Query result that can be further processed

        Raises:
            DQXError: If the query fails or the data source is unavailable

        Example:
            >>> result = datasource.query("SELECT COUNT(*) FROM source")
            >>> count = result.fetchone()[0]
        """
        ...


Validator = Callable[[Any], bool]
RetrievalFn = Callable[[ResultKey], Result[float, str]]


@dataclass
class SymbolicValidator:
    name: str
    fn: Validator


@runtime_checkable
class Analyzer(Protocol):
    """
    Protocol for data analysis engines that process SQL data sources.

    This protocol defines the minimal interface for analyzers that can process
    individual SqlDataSource instances, generating analysis reports based on
    specified metrics.
    """

    def analyze(
        self,
        ds: SqlDataSource,
        metrics: Sequence[MetricSpec],
        key: ResultKey,
    ) -> AnalysisReport:
        """
        Analyze a data source using the specified metrics.

        This is the main entry point for analysis operations. It processes
        a single data source and generates an analysis report.

        Args:
            ds: The data source to analyze
            metrics: Sequence of metric specifications to compute
            key: Result key for organizing and retrieving analysis results

        Returns:
            AnalysisReport: Report containing computed metrics and their values

        Raises:
            DQXError: If the data source type is unsupported or analysis fails

        Example:
            >>> analyzer = MyAnalyzer()
            >>> report = analyzer.analyze(data_source, [metric1, metric2], result_key)
            >>> print(f"Found {len(report)} computed metrics")
        """
        ...


@runtime_checkable
class Context(Protocol):
    def assert_that(self, expr: sp.Expr) -> AssertionDraft: ...

    @property
    def key(self) -> ResultKeyProvider: ...
