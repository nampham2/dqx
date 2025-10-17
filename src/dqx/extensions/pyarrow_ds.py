"""PyArrow data source adapter for DQX framework.

This module provides an adapter that enables PyArrow data structures (Tables and RecordBatches)
to be used as data sources within the DQX data quality analysis framework.

ArrowDataSource: For PyArrow Tables or RecordBatches

The adapter uses DuckDB as the SQL query engine, leveraging its efficient Arrow integration
to execute SQL queries directly on Arrow data without copying.

Note: For time-series analysis, data sources can use the nominal_date parameter
to filter data appropriately. The current implementation ignores this parameter,
but custom data sources can leverage it:

Example:
    class DateFilteredArrowSource(ArrowDataSource):
        def __init__(self, table: pa.Table, date_column: str):
            super().__init__(table)
            self.date_column = date_column

        def cte(self, nominal_date: date) -> str:
            # Filter data for specific date in the CTE
            return f'''
                SELECT * FROM {self._table_name}
                WHERE DATE({self.date_column}) = '{nominal_date.isoformat()}'
            '''

Example:
    >>> import pyarrow as pa
    >>> from dqx.extensions.pyarrow_ds import ArrowDataSource
    >>> from dqx.analyzer import Analyzer
    >>> from dqx.specs import MetricSpec
    >>>
    >>> # Single table analysis
    >>> table = pa.table({'id': [1, 2, 3], 'value': [10, 20, 30]})
    >>> ds = ArrowDataSource(table)
    >>>
    >>> # Analyze with DQX
    >>> analyzer = Analyzer()
    >>> metrics = [MetricSpec.num_rows(), MetricSpec.average("value")]
    >>> report = analyzer.analyze(ds, metrics, key)
"""

from __future__ import annotations

import datetime

import duckdb
import pyarrow as pa

from dqx.utils import random_prefix


class ArrowDataSource:
    """Adapter for PyArrow Tables and RecordBatches to work as DQX data sources.

    This class wraps PyArrow data structures and implements the SqlDataSource protocol,
    enabling them to be analyzed using DQX. It leverages DuckDB's native Arrow support
    to execute SQL queries directly on Arrow data with minimal overhead.

    The adapter is ideal for analyzing data that's already in Arrow format, such as
    data loaded from Parquet files, received from Arrow Flight, or processed using
    other Arrow-compatible tools.

    Attributes:
        name: Identifier for this data source type, always "pyarrow"
        dialect: SQL dialect used for query generation, always "duckdb"

    Example:
        >>> # From a PyArrow Table
        >>> table = pa.table({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> ds = ArrowDataSource(table)
        >>>
        >>> # From a RecordBatch
        >>> batch = pa.record_batch([pa.array([1, 2, 3])], names=['values'])
        >>> ds = ArrowDataSource(batch)
        >>>
        >>> # Analyze with DQX
        >>> report = analyzer.analyze_single(ds, metrics, key)
    """

    name: str = "pyarrow"
    dialect: str = "duckdb"

    def __init__(self, table: pa.RecordBatch | pa.Table) -> None:
        """Initialize the Arrow data source.

        Creates a wrapper around a PyArrow Table or RecordBatch with a randomly
        generated internal table name for use in SQL queries. The table name is
        prefixed with an underscore and followed by 6 random characters.

        Args:
            table: PyArrow Table or RecordBatch to wrap. Both types are supported
                  and will be handled transparently by DuckDB.

        Example:
            >>> import pyarrow.parquet as pq
            >>> table = pq.read_table('data.parquet')
            >>> ds = ArrowDataSource(table)
        """
        self._table = table
        self._table_name = random_prefix(k=6)

    def cte(self, nominal_date: datetime.date) -> str:
        """Get the CTE for this data source.

        Args:
            nominal_date: The date for filtering (currently ignored)

        Returns:
            The CTE SQL string
        """
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str, nominal_date: datetime.date) -> duckdb.DuckDBPyRelation:
        """Execute a query against the Arrow data.

        Args:
            query: The SQL query to execute
            nominal_date: The date for filtering (currently ignored)

        Returns:
            Query results as a DuckDB relation
        """
        return duckdb.arrow(self._table).query(
            self._table_name,
            query,
        )
