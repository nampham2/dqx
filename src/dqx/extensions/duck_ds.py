"""DuckDB data source adapter for DQX framework.

This module provides an adapter that enables DuckDB relations to be used as data sources
within the DQX data quality analysis framework. It implements the SqlDataSource protocol,
allowing DuckDB relations to be analyzed for various data quality metrics.

The primary use case is when you already have data loaded in DuckDB and want to analyze
it using DQX without converting to other formats. This adapter wraps DuckDB relations
and provides the necessary interface for the DQX analyzer to execute SQL queries.

Example:
    >>> import duckdb
    >>> from dqx.extensions.duck_ds import DuckRelationDataSource
    >>> from dqx.analyzer import Analyzer
    >>> from dqx.specs import MetricSpec
    >>>
    >>> # Create a DuckDB relation
    >>> conn = duckdb.connect()
    >>> relation = conn.sql("SELECT * FROM 'data.parquet'")
    >>>
    >>> # Wrap it as a DQX data source
    >>> ds = DuckRelationDataSource(relation)
    >>>
    >>> # Analyze with DQX
    >>> analyzer = Analyzer()
    >>> metrics = [MetricSpec.num_rows(), MetricSpec.average("price")]
    >>> report = analyzer.analyze_single(ds, metrics, key)
"""

from __future__ import annotations

import duckdb
import pyarrow as pa

from dqx.common import SqlDataSource
from dqx.utils import random_prefix


class DuckRelationDataSource:
    """Adapter for DuckDB relations to work as DQX data sources.

    This class wraps a DuckDB relation and implements the SqlDataSource protocol,
    enabling it to be used with the DQX analyzer. It provides a temporary table
    name for SQL queries and methods to execute queries against the wrapped relation.

    The adapter is particularly useful when you have complex DuckDB queries or
    transformations that you want to analyze for data quality metrics without
    materializing the results to disk.

    Attributes:
        name: Identifier for this data source type, always "duckdb"
        dialect: SQL dialect used for query generation, always "duckdb"

    Example:
        >>> # From an existing DuckDB query
        >>> relation = duckdb.sql("SELECT * FROM sales WHERE year = 2023")
        >>> ds = DuckRelationDataSource(relation)
        >>>
        >>> # Use with analyzer
        >>> report = analyzer.analyze_single(ds, metrics, key)
    """

    name: str = "duckdb"
    dialect: str = "duckdb"

    def __init__(self, relation: duckdb.DuckDBPyRelation) -> None:
        """Initialize the DuckDB relation data source.

        Creates a wrapper around a DuckDB relation with a randomly generated
        internal table name for use in SQL queries. The table name is prefixed
        with an underscore and followed by 6 random characters to avoid collisions.

        Args:
            relation: A DuckDB relation object to wrap. This can be the result
                     of any DuckDB query or transformation.

        Example:
            >>> conn = duckdb.connect()
            >>> rel = conn.sql("SELECT * FROM 'data.csv'")
            >>> ds = DuckRelationDataSource(rel)
        """
        self._relation = relation
        self._table_name = random_prefix(k=6)

    @property
    def cte(self) -> str:
        """Return a Common Table Expression (CTE) SQL fragment.

        This property provides the SQL needed to reference this data source
        in a WITH clause. The DQX analyzer uses this when constructing queries
        to analyze the data.

        Returns:
            SQL SELECT statement that references the internal table name.

        Example:
            >>> ds = DuckRelationDataSource(relation)
            >>> print(ds.cte)
            'SELECT * FROM _abc123'
        """
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query against the wrapped DuckDB relation.

        This method allows the DQX analyzer to run SQL queries against the
        data source. The query should reference the data using the internal
        table name available via the cte property.

        Args:
            query: SQL query string to execute. The query should reference
                  the data source using the table name from self._table_name.

        Returns:
            DuckDBPyRelation containing the query results.

        Example:
            >>> ds = DuckRelationDataSource(relation)
            >>> result = ds.query(f"SELECT COUNT(*) FROM {ds._table_name}")
            >>> count = result.fetchone()[0]

        Note:
            The query parameter typically comes from the DQX analyzer and
            already includes proper references to the internal table name.
        """
        return self._relation.query(self._table_name, query)

    @classmethod
    def from_arrow(cls, table: pa.RecordBatch | pa.Table) -> SqlDataSource:
        """Create a data source from PyArrow data.

        This factory method delegates to ArrowDataSource, which is better
        suited for handling PyArrow data structures. This ensures that
        PyArrow data is processed using the most appropriate adapter.

        Args:
            table: PyArrow Table or RecordBatch to create a data source from.

        Returns:
            ArrowDataSource instance wrapping the provided PyArrow data.

        Example:
            >>> import pyarrow as pa
            >>> table = pa.table({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
            >>> ds = DuckRelationDataSource.from_arrow(table)
            >>> # ds is actually an ArrowDataSource instance

        Note:
            This method exists for API consistency but always returns an
            ArrowDataSource rather than a DuckRelationDataSource.
        """
        from dqx.extensions.pyarrow_ds import ArrowDataSource

        return ArrowDataSource(table)
