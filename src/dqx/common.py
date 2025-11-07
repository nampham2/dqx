from __future__ import annotations

import datetime
import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import duckdb
import sympy as sp
from returns.result import Result

if TYPE_CHECKING:
    from dqx.analyzer import AnalysisReport
    from dqx.api import AssertionDraft
    from dqx.models import Metric
    from dqx.provider import SymbolInfo
    from dqx.specs import MetricSpec

# Type aliases
DatasetName = str
ExecutionId = str
TimeSeries = dict[dt.date, "Metric"]
Tags = dict[str, str]
Parameters = dict[str, Any]
SeverityLevel = Literal["P0", "P1", "P2", "P3"]
RecomputeStrategy = Literal["ALWAYS", "MISSING", "NEVER"]
AssertionStatus = Literal["PASSED", "FAILED", "SKIPPED", "ERROR"]
Validator = Callable[[float], bool]
RetrievalFn = Callable[["ResultKey"], Result[float, str]]
MetricKey = tuple["MetricSpec", "ResultKey", DatasetName]


class DQXError(Exception): ...


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


@dataclass
class Metadata:
    """Metadata for metric lifecycle and execution context."""

    execution_id: str | None = None
    ttl_hours: int = 168  # 7 days default


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
        status: Validation result ("PASSED" or "FAILED")
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


@dataclass
class SymbolicValidator:
    name: str
    fn: Validator


@dataclass(frozen=True)
class PluginMetadata:
    """Immutable metadata that plugins must provide."""

    name: str
    version: str
    author: str
    description: str
    capabilities: set[str] = field(default_factory=set)


@runtime_checkable
class SqlDataSource(Protocol):
    """
    Protocol for SQL data sources that can be analyzed by DQX.

    This protocol defines the interface for adapting various data sources
    (e.g., Arrow tables, BigQuery, DuckDB tables) to work with the DQX
    analysis framework.

    Attributes:
        name: A unique identifier for this data source instance
        dialect: The SQL dialect name used for query generation
    """

    dialect: str

    @property
    def name(self) -> str:
        """Get the name of this data source."""
        ...

    @property
    def skip_dates(self) -> set[datetime.date]:
        """Dates to exclude from metric calculations for this dataset."""
        return set()

    def cte(self, nominal_date: datetime.date, parameters: Parameters | None = None) -> str:
        """
        Get the Common Table Expression (CTE) for this data source.

        Args:
            nominal_date: The date for which the CTE should filter data.
                         Implementations may ignore this parameter if date
                         filtering is not needed.
            parameters: Optional parameters for filtering or transforming the CTE.
                       Implementations can use these for custom SQL operations.

        Returns:
            The CTE SQL string
        """
        ...

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """
        Execute a query against this data source.

        Args:
            query: The SQL query to execute

        Returns:
            Query results as a DuckDB relation
        """
        ...


@runtime_checkable
class Analyzer(Protocol):
    """
    Protocol for data analysis engines that process SQL data sources.

    This protocol defines the minimal interface for analyzers that can process
    SqlDataSource instances, generating analysis reports based on specified
    metrics across multiple dates/keys.
    """

    def analyze(
        self,
        ds: SqlDataSource,
        metrics: dict[ResultKey, Sequence[MetricSpec]],
    ) -> AnalysisReport:
        """
        Analyze a data source using the specified metrics across multiple dates.

        This is the main entry point for analysis operations. It processes
        a data source and generates an analysis report by executing metrics
        in an optimized batch query.

        Args:
            ds: The data source to analyze
            metrics: Dictionary mapping ResultKeys to their metrics

        Returns:
            AnalysisReport: Report containing computed metrics and their values

        Raises:
            DQXError: If the metrics dict is empty or analysis fails

        Example:
            >>> analyzer = MyAnalyzer()
            >>> metrics_by_key = {
            ...     ResultKey(date(2024, 1, 1), {}): [Sum("revenue"), Average("price")],
            ...     ResultKey(date(2024, 1, 2), {}): [Sum("revenue"), Average("price")]
            ... }
            >>> report = analyzer.analyze(data_source, metrics_by_key)
            >>> print(f"Found {len(report)} computed metrics")
        """
        ...


@runtime_checkable
class Context(Protocol):
    def assert_that(self, expr: sp.Expr) -> AssertionDraft: ...
