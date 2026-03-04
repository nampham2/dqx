"""Data source implementations for DQX framework.

This module provides data source adapters that implement the SqlDataSource protocol,
enabling various data formats to be analyzed within the DQX data quality framework.

Available implementations:
- DuckRelationDataSource: Wraps DuckDB relations for complex query pipelines
- ArrowDataSource: Direct adapter for PyArrow Tables and RecordBatches
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

        # Cache schema to avoid repeated Arrow conversions
        self._schema = self._relation.arrow().schema

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

    @property
    def schema(self) -> pa.Schema:
        """Get the PyArrow schema of the underlying relation.

        Returns the schema of the raw data before any CTE filtering,
        allowing consumers to understand the data structure.

        Returns:
            pa.Schema: The PyArrow schema of the dataset.
        """
        return self._schema

    def query(self, query: str) -> pa.Table:
        """Execute a query against the DuckDB relation.

        Args:
            query: The SQL query to execute

        Returns:
            Query results as a PyArrow Table
        """
        result = self._relation.query(self._table_name, query)
        arrow_result = result.arrow()
        # If it's a RecordBatchReader, read all batches into a Table
        if isinstance(arrow_result, pa.RecordBatchReader):
            return arrow_result.read_all()
        # If it's already a Table, return it (defensive, currently DuckDB always returns RecordBatchReader)
        return arrow_result  # pragma: no cover

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


class ArrowDataSource(SqlDataSource):
    """Adapter for PyArrow Tables to work as DQX data sources.

    This class provides a direct, lightweight wrapper around PyArrow Tables and
    RecordBatches, implementing the SqlDataSource protocol for use with the DQX
    analyzer. Unlike DuckRelationDataSource, this adapter stores the Arrow data
    directly without creating a persistent DuckDB relation, making it ideal for
    simple in-memory tables.

    The adapter creates temporary DuckDB relations on-demand for SQL query execution,
    providing the full power of SQL analytics while maintaining a minimal API surface.

    Attributes:
        name: Name of this specific dataset (e.g., "orders", "users")
        dialect: SQL dialect used for query generation, always "duckdb"

    Example:
        >>> import pyarrow as pa
        >>> from dqx.datasource import ArrowDataSource
        >>>
        >>> # From a PyArrow Table
        >>> arrow_table = pa.table({
        ...     'id': [1, 2, 3, 4],
        ...     'value': [10.5, 20.3, 30.1, 40.7]
        ... })
        >>> ds = ArrowDataSource(arrow_table, "sales_data")
        >>>
        >>> # Use with analyzer
        >>> from dqx.analyzer import Analyzer
        >>> analyzer = Analyzer()
        >>> report = analyzer.analyze_single(ds, metrics, key)
    """

    dialect: str = "duckdb"

    def __init__(
        self,
        table: pa.Table | pa.RecordBatch,
        name: str,
        skip_dates: set[datetime.date] | None = None,
    ) -> None:
        """Initialize the Arrow data source.

        Creates a wrapper around a PyArrow Table or RecordBatch with a randomly
        generated internal table name for use in SQL queries. The table name is
        prefixed with an underscore and followed by 6 random characters to avoid
        collisions when multiple instances exist.

        Args:
            table: A PyArrow Table or RecordBatch containing the data to analyze.
                   Both types are supported and handled transparently.
            name: The name of this dataset (e.g., "orders", "users")
            skip_dates: Optional set of dates to exclude from calculations

        Example:
            >>> import pyarrow as pa
            >>> table = pa.table({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
            >>> ds = ArrowDataSource(table, "my_data")
            >>>
            >>> # With skip_dates
            >>> import datetime
            >>> skip_dates = {datetime.date(2024, 1, 1)}
            >>> ds = ArrowDataSource(table, "my_data", skip_dates=skip_dates)
        """
        self._table = table
        self._name = name
        self._skip_dates = skip_dates or set()
        self._table_name = random_prefix(k=6)

        # Cache schema to avoid repeated access
        self._schema = table.schema

    def cte(self, nominal_date: datetime.date, parameters: Parameters | None = None) -> str:
        """Get the CTE for this data source.

        Returns a simple SELECT * statement since ArrowDataSource does not
        support date-based or parameter-based filtering. The CTE selects all
        rows from the internal table name.

        Args:
            nominal_date: The date for filtering (ignored by ArrowDataSource)
            parameters: Optional parameters for filtering (ignored by ArrowDataSource)

        Returns:
            The CTE SQL string selecting all rows from the table
        """
        return f"SELECT * FROM {self._table_name}"

    @property
    def schema(self) -> pa.Schema:
        """Get the PyArrow schema of the underlying table.

        Returns the schema of the raw data, allowing consumers to understand
        the data structure without executing queries.

        Returns:
            pa.Schema: The PyArrow schema of the dataset.
        """
        return self._schema

    def query(self, query: str) -> pa.Table:
        """Execute a query against the Arrow table.

        Creates a temporary DuckDB relation from the Arrow table and executes
        the provided SQL query. The query should reference the internal table
        name accessible via the cte() method.

        Args:
            query: The SQL query to execute, referencing the table by its
                   internal name (accessible via self._table_name or cte())

        Returns:
            Query results as a PyArrow Table

        Example:
            >>> ds = ArrowDataSource(table, "data")
            >>> result = ds.query(f"SELECT COUNT(*) FROM {ds._table_name}")
        """
        # Create temporary relation for query execution
        relation = duckdb.arrow(self._table)
        result = relation.query(self._table_name, query)
        arrow_result = result.arrow()

        # If it's a RecordBatchReader, read all batches into a Table
        if isinstance(arrow_result, pa.RecordBatchReader):
            return arrow_result.read_all()
        # If it's already a Table, return it
        return arrow_result  # pragma: no cover

    @property
    def name(self) -> str:
        """Get the name of this data source (read-only)."""
        return self._name

    @property
    def skip_dates(self) -> set[datetime.date]:
        """Get the skip_dates for this data source (read-only)."""
        return self._skip_dates
