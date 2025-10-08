"""PyArrow data source adapters for DQX framework.

This module provides adapters that enable PyArrow data structures (Tables and RecordBatches)
to be used as data sources within the DQX data quality analysis framework. It includes two
main adapters:

1. ArrowDataSource: For single PyArrow Tables or RecordBatches
2. ArrowBatchDataSource: For processing multiple batches, particularly useful for large
   datasets that need to be processed in chunks for memory efficiency

Both adapters use DuckDB as the SQL query engine, leveraging its efficient Arrow integration
to execute SQL queries directly on Arrow data without copying.

Example:
    >>> import pyarrow as pa
    >>> from dqx.extensions.pyarrow_ds import ArrowDataSource, ArrowBatchDataSource
    >>> from dqx.analyzer import Analyzer
    >>> from dqx.specs import MetricSpec
    >>>
    >>> # Single table analysis
    >>> table = pa.table({'id': [1, 2, 3], 'value': [10, 20, 30]})
    >>> ds = ArrowDataSource(table)
    >>>
    >>> # Batch processing for large datasets
    >>> batch_ds = ArrowBatchDataSource.from_parquets(['file1.parquet', 'file2.parquet'])
    >>>
    >>> # Analyze with DQX
    >>> analyzer = Analyzer()
    >>> metrics = [MetricSpec.num_rows(), MetricSpec.average("value")]
    >>> report = analyzer.analyze(ds, metrics, key)
"""

from __future__ import annotations

from typing import Any, Iterable

import duckdb
import pyarrow as pa
from pyarrow.dataset import dataset

from dqx.utils import random_prefix

MAX_ARROW_BATCH_SIZE: int = 10_000_000


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

    @property
    def cte(self) -> str:
        """Return a Common Table Expression (CTE) SQL fragment.

        This property provides the SQL needed to reference this data source
        in a WITH clause. The DQX analyzer uses this when constructing queries
        to analyze the data.

        Returns:
            SQL SELECT statement that references the internal table name.

        Example:
            >>> ds = ArrowDataSource(table)
            >>> print(ds.cte)
            'SELECT * FROM _xyz789'
        """
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query against the wrapped Arrow data.

        This method uses DuckDB's Arrow integration to run SQL queries directly
        on the Arrow data. The query should reference the data using the internal
        table name available via the cte property.

        Args:
            query: SQL query string to execute. The query should reference
                  the data source using the table name from self._table_name.

        Returns:
            DuckDBPyRelation containing the query results.

        Example:
            >>> ds = ArrowDataSource(table)
            >>> result = ds.query(f"SELECT COUNT(*) FROM {ds._table_name}")
            >>> count = result.fetchone()[0]

        Note:
            DuckDB creates a zero-copy view of the Arrow data when possible,
            making this operation very efficient.
        """
        return duckdb.arrow(self._table).query(
            self._table_name,
            query,
        )


class ArrowBatchDataSource:
    """Batch processor for multiple PyArrow Tables or RecordBatches.

    This class handles collections of Arrow data structures, enabling efficient
    batch-wise processing of large datasets. It's particularly useful when dealing
    with data that's too large to fit in memory at once, or when processing
    partitioned datasets like those stored in multiple Parquet files.

    The class provides a convenient factory method for loading data from Parquet
    files with automatic batching support.

    Attributes:
        name: Identifier for this data source type, always "pyarrow_batch"
        dialect: SQL dialect used for query generation, always "duckdb"

    Example:
        >>> # From a list of tables
        >>> tables = [table1, table2, table3]
        >>> batch_ds = ArrowBatchDataSource(tables)
        >>>
        >>> # From Parquet files with automatic batching
        >>> batch_ds = ArrowBatchDataSource.from_parquets([
        ...     'data/2023-01.parquet',
        ...     'data/2023-02.parquet',
        ...     'data/2023-03.parquet'
        ... ], batch_size=5_000_000)
        >>>
        >>> # Process with analyzer
        >>> report = analyzer.analyze(batch_ds, metrics, key, threading=True)
    """

    name: str = "pyarrow_batch"
    dialect: str = "duckdb"

    def __init__(self, batches: Iterable[pa.RecordBatch | pa.Table]) -> None:
        """Initialize the batch data source.

        Creates a batch processor that can iterate over multiple Arrow data
        structures. Each batch will be processed independently, allowing for
        memory-efficient processing of large datasets.

        Args:
            batches: Iterable of PyArrow Tables or RecordBatches. Can be a
                    list, generator, or any iterable that yields Arrow data.

        Example:
            >>> # From record batches with limited memory
            >>> def generate_batches():
            ...     for file in large_files:
            ...         yield pq.read_table(file, memory_map=True)
            >>>
            >>> batch_ds = ArrowBatchDataSource(generate_batches())
        """
        self._batches = batches

    def arrow_ds(self) -> Iterable[ArrowDataSource]:
        """Generate ArrowDataSource instances for each batch.

        This method provides an iterator over ArrowDataSource instances, with
        one instance per batch in the collection. This enables the DQX analyzer
        to process each batch independently, which is useful for parallel
        processing or memory-constrained environments.

        Yields:
            ArrowDataSource instances wrapping each batch.

        Example:
            >>> batch_ds = ArrowBatchDataSource([table1, table2])
            >>> for ds in batch_ds.arrow_ds():
            ...     # Each ds is an ArrowDataSource that can be analyzed
            ...     result = analyzer.analyze_single(ds, metrics, key)

        Note:
            This method is typically called internally by the analyzer when
            processing batch data sources.
        """
        for batch in self._batches:
            yield ArrowDataSource(batch)

    @classmethod
    def from_parquets(
        cls,
        parquets: Iterable[str],
        batch_size: int = MAX_ARROW_BATCH_SIZE,
        filesystem: Any | None = None,
    ) -> ArrowBatchDataSource:
        """Create a batch data source from Parquet files.

        This factory method provides a convenient way to load multiple Parquet
        files as a batch data source. It uses PyArrow's dataset API to efficiently
        read the files and automatically splits them into manageable batches.

        Args:
            parquets: Iterable of Parquet file paths. Can include glob patterns
                     or directory paths containing Parquet files.
            batch_size: Maximum number of rows per batch. Defaults to 10 million.
                       Smaller batches use less memory but may be slower to process.
            filesystem: Optional filesystem to use for reading files. Can be any
                       PyArrow filesystem (S3, GCS, HDFS, etc.). If None, uses
                       the local filesystem.

        Returns:
            ArrowBatchDataSource configured to read the specified Parquet files.

        Example:
            >>> # Load all Parquet files from a directory
            >>> batch_ds = ArrowBatchDataSource.from_parquets(
            ...     ['data/2023/*.parquet'],
            ...     batch_size=1_000_000
            ... )
            >>>
            >>> # Load from S3 with custom filesystem
            >>> import pyarrow.fs
            >>> s3_fs = pyarrow.fs.S3FileSystem(region='us-east-1')
            >>> batch_ds = ArrowBatchDataSource.from_parquets(
            ...     ['s3://bucket/data/*.parquet'],
            ...     filesystem=s3_fs,
            ...     batch_size=5_000_000
            ... )

        Note:
            The dataset API provides efficient reading with predicate pushdown
            and column pruning when supported by the file format.
        """
        return cls(dataset(parquets, format="parquet", filesystem=filesystem).to_batches(batch_size=batch_size))
