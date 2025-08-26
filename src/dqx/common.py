from __future__ import annotations

import datetime
import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

import duckdb
import sympy as sp
from returns.result import Result


if TYPE_CHECKING:
    from dqx.api import SymbolicAssert
    from dqx.specs import MetricSpec
    from dqx.analyzer import AnalysisReport


class DQXError(Exception): ...


TimeSeries = Mapping[dt.date, float]
Tags = dict[str, Any]

SeverityLevel = Literal["P0", "P1", "P2", "P3"]
Parameters = dict[str, Any]


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
        analyzer_class: The analyzer class used to process this data source
        dialect: The SQL dialect used for query generation
    """

    name: str
    analyzer_class: type[Analyzer]
    dialect: Any  # Will be Dialect protocol when imported

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


@runtime_checkable
class BatchSqlDataSource(Protocol):
    """
    Protocol for batch SQL data sources that provide data in multiple batches.
    
    This protocol extends the concept of SqlDataSource to handle scenarios where
    data processing needs to be done in batches (e.g., for memory efficiency,
    parallel processing, or when dealing with partitioned datasets).
    
    The protocol defines a consistent interface for iterating over batches of
    SqlDataSource objects, enabling efficient processing of large datasets.
    """
    
    def batches(self) -> Iterable[SqlDataSource]:
        """
        Return an iterable of SqlDataSource instances representing data batches.
        
        Each batch should be a complete SqlDataSource that can be independently
        processed. This method enables batch-wise processing for memory efficiency
        and parallel execution.
        
        Returns:
            Iterable[SqlDataSource]: An iterable of SqlDataSource instances,
                                   where each instance represents a batch of data
                                   
        Example:
            >>> batch_source = MyBatchSqlDataSource()
            >>> for batch in batch_source.batches():
            ...     result = analyzer.analyze_single(batch, metrics, key)
            
        Note:
            - Each batch should be self-contained and independently queryable
            - The iteration order should be deterministic when possible
            - Batches should not overlap in terms of data coverage
        """
        ...


Validator = Callable[[Any], bool]


@dataclass
class SymbolicValidator:
    name: str
    fn: Validator | None = None


RetrievalFn = Callable[[ResultKey], Result[float, str]]


@runtime_checkable
class ExtendedMetricProvider(Protocol):
    def day_over_day(
        self, metric: MetricSpec, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def stddev(
        self,
        metric: MetricSpec,
        lag: int,
        n: int,
        key: ResultKeyProvider = ResultKeyProvider(),
        datasets: list[str] | None = None,
    ) -> sp.Symbol: ...


@runtime_checkable
class MetricProvider(Protocol):
    @property
    def ext(self) -> ExtendedMetricProvider: ...

    def num_rows(
        self, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def first(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def average(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def minimum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def maximum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def sum(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def null_count(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def variance(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...

    def approx_cardinality(
        self, column: str, key: ResultKeyProvider = ResultKeyProvider(), datasets: list[str] | None = None
    ) -> sp.Symbol: ...


@runtime_checkable
class Analyzer(Protocol):
    """
    Protocol for data analysis engines that process SQL data sources.
    
    This protocol defines the minimal interface for analyzers that can process
    both individual SqlDataSource instances and BatchSqlDataSource collections,
    generating analysis reports based on specified metrics.
    
    The protocol supports both synchronous and asynchronous (threaded) processing
    modes for optimal performance with different data source types.
    """

    def analyze(
        self,
        ds: SqlDataSource | BatchSqlDataSource,
        metrics: Sequence[MetricSpec],
        key: ResultKey,
        threading: bool = False,
    ) -> AnalysisReport:
        """
        Analyze a data source using the specified metrics.
        
        This is the main entry point for analysis operations. It handles both
        single data sources and batch data sources, automatically choosing the
        appropriate processing strategy.
        
        Args:
            ds: The data source to analyze (single or batch)
            metrics: Sequence of metric specifications to compute
            key: Result key for organizing and retrieving analysis results
            threading: Whether to use multi-threaded processing for batch sources
            
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

    def analyze_single(
        self, ds: SqlDataSource, metrics: Sequence[MetricSpec], key: ResultKey
    ) -> AnalysisReport:
        """
        Analyze a single SQL data source.
        
        This method processes a single SqlDataSource instance, computing all
        specified metrics and returning the results in an analysis report.
        
        Args:
            ds: The single SQL data source to analyze
            metrics: Sequence of metric specifications to compute
            key: Result key for organizing the analysis results
            
        Returns:
            AnalysisReport: Report containing the computed metrics
            
        Raises:
            DQXError: If no metrics are provided or analysis fails
            
        Note:
            This method is typically called internally by `analyze()` for
            single data sources or for each batch in batch processing.
        """
        ...

    def persist(self, db: Any, overwrite: bool = True) -> None:
        """
        Persist the analysis results to the database.
        
        Args:
            db: The database instance to persist results to
            overwrite: Whether to overwrite existing results or merge them
        """
        ...


@runtime_checkable
class Context(Protocol):
    def assert_that(self, expr: sp.Expr) -> SymbolicAssert: ...

    @property
    def key(self) -> ResultKeyProvider: ...
