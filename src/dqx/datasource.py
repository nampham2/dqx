"""Data source implementations for DQX framework.

This module provides data source adapters that implement the SqlDataSource protocol,
enabling various data formats to be analyzed within the DQX data quality framework.

The primary implementation is DuckRelationDataSource, which wraps DuckDB relations
and provides the necessary interface for the DQX analyzer to execute SQL queries.
"""

from __future__ import annotations

import datetime
from typing import Self

import duckdb
import pyarrow as pa

from dqx.common import Parameters, SqlDataSource
from dqx.utils import random_prefix


class DuckRelationDataSource(SqlDataSource):
    """Adapter for DuckDB relations to work as DQX data sources.

    This class wraps a DuckDB relation and implements the SqlDataSource protocol,
    enabling it to be used with the DQX analyzer. It provides a temporary table
    name for SQL queries and methods to execute queries against the wrapped relation.

    The adapter is particularly useful when you have complex DuckDB queries or
    transformations that you want to analyze for data quality metrics without
    materializing the results to disk.

    Attributes:
        name: Name of this specific dataset (e.g., "orders", "users")
        dialect: SQL dialect used for query generation, always "duckdb"

    Example:
        >>> # From an existing DuckDB query
        >>> relation = duckdb.sql("SELECT * FROM sales WHERE year = 2023")
        >>> ds = DuckRelationDataSource(relation, "sales_2023")
        >>>
        >>> # Use with analyzer
        >>> report = analyzer.analyze_single(ds, metrics, key)
    """

    dialect: str = "duckdb"

    def __init__(
        self, relation: duckdb.DuckDBPyRelation, name: str, skip_dates: set[datetime.date] | None = None
    ) -> None:
        """Initialize the DuckDB relation data source.

        Creates a wrapper around a DuckDB relation with a randomly generated
        internal table name for use in SQL queries. The table name is prefixed
        with an underscore and followed by 6 random characters to avoid collisions.

        Args:
            relation: A DuckDB relation object to wrap. This can be the result
                     of any DuckDB query or transformation.
            name: The name of this dataset (e.g., "orders", "users")
            skip_dates: Optional set of dates to exclude from calculations

        Example:
            >>> conn = duckdb.connect()
            >>> rel = conn.sql("SELECT * FROM 'data.csv'")
            >>> ds = DuckRelationDataSource(rel, "my_data")
        """
        self._name = name
        self._relation = relation
        self._table_name = random_prefix(k=6)
        self._skip_dates = skip_dates or set()

        # Initialize DuckDB settings
        self._setup_duckdb()

    def _setup_duckdb(self) -> None:
        duckdb.execute("SET enable_progress_bar = false")

    def cte(self, nominal_date: datetime.date, parameters: Parameters | None = None) -> str:
        """Get the CTE for this data source.

        Args:
            nominal_date: The date for filtering (currently ignored)
            parameters: Optional parameters for filtering (currently ignored)

        Returns:
            The CTE SQL string
        """
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute a query against the DuckDB relation.

        Args:
            query: The SQL query to execute

        Returns:
            Query results as a DuckDB relation
        """
        return self._relation.query(self._table_name, query)

    @property
    def name(self) -> str:
        """Get the name of this data source (read-only)."""
        return self._name

    @property
    def skip_dates(self) -> set[datetime.date]:
        """Get the skip_dates for this data source (read-only)."""
        return self._skip_dates

    @classmethod
    def from_arrow(
        cls, table: pa.RecordBatch | pa.Table, name: str, skip_dates: set[datetime.date] | None = None
    ) -> Self:
        """Create a DuckRelationDataSource from PyArrow data structures.

        This factory method provides a convenient way to create a DuckDB data source
        directly from PyArrow Tables or RecordBatches. It leverages DuckDB's native
        Arrow integration to create a relation from the Arrow data, then wraps it
        in a DuckRelationDataSource for use with the DQX analyzer.

        This method is particularly useful when you have data in Arrow format
        (e.g., from Parquet files, Arrow IPC, or computational results) and want
        to analyze it using DQX without intermediate conversions.

        Args:
            table: A PyArrow Table or RecordBatch containing the data to analyze.
                   Both types are supported and will be converted to a DuckDB
                   relation automatically.
            name: The name of this dataset (e.g., "orders", "users")
            skip_dates: Optional set of dates to exclude from calculations

        Returns:
            A new DuckRelationDataSource instance wrapping the Arrow data.

        Example:
            >>> import pyarrow as pa
            >>> from dqx.datasource import DuckRelationDataSource
            >>>
            >>> # From a PyArrow Table
            >>> arrow_table = pa.table({
            ...     'id': [1, 2, 3, 4],
            ...     'value': [10.5, 20.3, 30.1, 40.7]
            ... })
            >>> ds = DuckRelationDataSource.from_arrow(arrow_table, "sales_data")
            >>>
            >>> # From a RecordBatch with skip_dates
            >>> batch = pa.record_batch([
            ...     pa.array([1, 2, 3]),
            ...     pa.array(['a', 'b', 'c'])
            ... ], names=['id', 'category'])
            >>> import datetime
            >>> skip_dates = {datetime.date(2024, 1, 1)}
            >>> ds = DuckRelationDataSource.from_arrow(batch, "category_data", skip_dates)
            >>>
            >>> # Use with analyzer
            >>> analyzer = Analyzer()
            >>> metrics = [MetricSpec.num_rows(), MetricSpec.cardinality('category')]
            >>> report = analyzer.analyze_single(ds, metrics, key)
        """
        relation: duckdb.DuckDBPyRelation = duckdb.arrow(table)
        return cls(relation, name, skip_dates)
