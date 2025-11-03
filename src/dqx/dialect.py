"""SQL dialect abstraction for DQX framework.

This module provides a protocol-based abstraction for SQL dialects,
allowing DQX to support different SQL databases beyond DuckDB.

## Overview

The dialect system enables DQX to generate SQL compatible with different
database systems. Each dialect handles:
- Translation of SqlOp operations to dialect-specific SQL
- Query formatting and structure
- Database-specific function mappings

## Usage

### Using the default DuckDB dialect:

    >>> from dqx.dialect import DuckDBDialect
    >>> from dqx.ops import Average, Sum
    >>>
    >>> dialect = DuckDBDialect()
    >>> avg_op = Average("price")
    >>> sql = dialect.translate_sql_op(avg_op)
    >>> print(sql)  # CAST(AVG(price) AS DOUBLE) AS 'prefix_average(price)'

### Building formatted queries:

    >>> expressions = [
    ...     "COUNT(*) AS 'total_count'",
    ...     "AVG(price) AS 'avg_price'",
    ...     "SUM(quantity) AS 'total_quantity'"
    ... ]
    >>> query = dialect.build_cte_query(
    ...     "SELECT * FROM orders",
    ...     expressions
    ... )
    >>> print(query)
    WITH source AS (
        SELECT * FROM orders
    )
    SELECT
        COUNT(*)      AS 'total_count'
      , AVG(price)    AS 'avg_price'
      , SUM(quantity) AS 'total_quantity'
    FROM source

## Extending with new dialects

To add support for a new database (e.g., PostgreSQL):

    from dqx.dialect import build_cte_query, auto_register

    @auto_register  # Automatically registers the dialect on import
    class PostgreSQLDialect:
        name = "postgresql"

        def translate_sql_op(self, op: ops.SqlOp) -> str:
            match op:
                case ops.NumRows():
                    return f"COUNT(*)::FLOAT8 AS {op.sql_col}"

                case ops.NullCount(column=col):
                    # PostgreSQL doesn't have COUNT_IF
                    return f"COUNT(CASE WHEN {col} IS NULL THEN 1 END) AS {op.sql_col}"

                # ... handle other operations

        def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
            return build_cte_query(cte_sql, select_expressions)

## Integration with DQX

When integrated with the analyzer, dialects will be used like:

    analyzer = Analyzer(dialect=PostgreSQLDialect())
    # or
    datasource = PostgreSQLDataSource(..., dialect=PostgreSQLDialect())

This allows the same DQX code to work across different databases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, Type, runtime_checkable

from dqx import ops
from dqx.common import DQXError, ResultKey

if TYPE_CHECKING:
    from dqx.ops import SqlOp


@dataclass
class BatchCTEData:
    """Data for building a batch CTE query."""

    key: ResultKey
    cte_sql: str
    ops: Sequence[SqlOp]


# Dialect Registry
_DIALECT_REGISTRY: dict[str, Type[Dialect]] = {}


def build_cte_query(cte_sql: str, select_expressions: list[str]) -> str:
    """Build CTE query without formatting.

    Args:
        cte_sql: The CTE SQL expression (e.g., "SELECT * FROM table")
        select_expressions: List of SQL expressions to select

    Returns:
        SQL query string

    Raises:
        ValueError: If no SELECT expressions are provided
    """
    if not select_expressions:
        raise ValueError("No SELECT expressions provided")

    select_clause = ", ".join(select_expressions)
    return f"WITH source AS ({cte_sql}) SELECT {select_clause} FROM source"


def _build_cte_parts(
    dialect: "Dialect", cte_data: list["BatchCTEData"]
) -> tuple[list[str], list[tuple[str, list[ops.SqlOp]]]]:
    """Build CTE parts for batch query - shared between dialects.

    Args:
        dialect: The dialect instance to use for SQL translation
        cte_data: List of BatchCTEData objects

    Returns:
        Tuple of (cte_parts, metrics_info)
        where metrics_info contains (metrics_cte_name, ops) for each CTE with ops

    Raises:
        ValueError: If no CTE data provided
    """
    if not cte_data:
        raise ValueError("No CTE data provided")

    cte_parts = []
    metrics_info: list[tuple[str, list[ops.SqlOp]]] = []

    for i, data in enumerate(cte_data):
        # Format date for CTE names (yyyy_mm_dd)
        # Include index to ensure unique names even for same date with different tags
        date_suffix = data.key.yyyy_mm_dd.strftime("%Y_%m_%d")
        source_cte = f"source_{date_suffix}_{i}"
        metrics_cte = f"metrics_{date_suffix}_{i}"

        # Add source CTE
        cte_parts.append(f"{source_cte} AS ({data.cte_sql})")

        # Build metrics CTE with all expressions if ops exist
        if data.ops:
            # Translate ops to expressions
            expressions = [dialect.translate_sql_op(op) for op in data.ops]
            metrics_select = ", ".join(expressions)
            cte_parts.append(f"{metrics_cte} AS (SELECT {metrics_select} FROM {source_cte})")

            # Store metrics info for later use
            metrics_info.append((metrics_cte, list(data.ops)))

    return cte_parts, metrics_info


def _build_batch_query_with_values(
    dialect: "Dialect", cte_data: list["BatchCTEData"], value_formatter: Callable[[list[ops.SqlOp]], str]
) -> str:
    """Build batch query with custom value formatting.

    Args:
        dialect: The dialect instance to use for SQL translation
        cte_data: List of BatchCTEData objects
        value_formatter: Function that formats ops into a value expression (MAP, STRUCT, etc.)

    Returns:
        Complete SQL query with CTEs and formatted values

    Raises:
        ValueError: If no CTE data provided or no metrics to compute
    """
    cte_parts, metrics_info = _build_cte_parts(dialect, cte_data)

    # Simple validation inline
    if not metrics_info:
        raise ValueError("No metrics to compute")

    value_selects = []
    for data, (metrics_cte, data_ops) in zip(cte_data, metrics_info):
        date_str = data.key.yyyy_mm_dd.isoformat()
        values_expr = value_formatter(data_ops)
        value_selects.append(f"SELECT '{date_str}' as date, {values_expr} as values FROM {metrics_cte}")

    cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
    union_clause = "\n".join(f"{'UNION ALL' if i > 0 else ''}\n{select}" for i, select in enumerate(value_selects))

    return f"{cte_clause}\n{union_clause}"


@runtime_checkable
class Dialect(Protocol):
    """Protocol for SQL dialect implementations.

    Dialects handle the translation of SqlOp operations to
    dialect-specific SQL expressions and query formatting.
    """

    @property
    def name(self) -> str:
        """Name of the SQL dialect (e.g., 'duckdb', 'postgresql')."""
        ...

    def translate_sql_op(self, op: ops.SqlOp) -> str:
        """Translate a SqlOp to dialect-specific SQL expression.

        Args:
            op: The SqlOp operation to translate

        Returns:
            SQL expression string including column alias

        Raises:
            ValueError: If the SqlOp type is not supported
        """
        ...

    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build a complete CTE query.

        Args:
            cte_sql: The CTE SQL expression (e.g., "SELECT * FROM table")
            select_expressions: List of SQL expressions to select

        Returns:
            SQL query string
        """
        ...

    def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
        """Build a batch CTE query for multiple dates.

        Args:
            cte_data: List of BatchCTEData objects containing:
                - key: ResultKey with the date
                - cte_sql: CTE SQL for this date
                - ops: List of SqlOp objects to translate

        Returns:
            Complete SQL query with CTEs and UNION ALL

        Example output:
            WITH
              source_2024_01_01 AS (...),
              metrics_2024_01_01 AS (SELECT ... FROM source_2024_01_01)
            SELECT '2024-01-01' as date, 'x_1' as symbol, x_1 as value FROM metrics_2024_01_01
            UNION ALL
            SELECT '2024-01-01' as date, 'x_2' as symbol, x_2 as value FROM metrics_2024_01_01
        """
        ...


def register_dialect(name: str, dialect_class: Type[Dialect]) -> None:
    """Register a dialect in the global registry.

    Args:
        name: The name to register the dialect under
        dialect_class: The dialect class to register

    Raises:
        ValueError: If a dialect with this name is already registered
    """
    if name in _DIALECT_REGISTRY:
        raise ValueError(f"Dialect '{name}' is already registered")
    _DIALECT_REGISTRY[name] = dialect_class


def get_dialect(name: str) -> Dialect:
    """Get a dialect instance by name from the registry.

    Args:
        name: The name of the dialect to retrieve

    Returns:
        An instance of the requested dialect

    Raises:
        DQXError: If the dialect is not found in the registry
    """
    if name not in _DIALECT_REGISTRY:
        available = ", ".join(sorted(_DIALECT_REGISTRY.keys()))
        raise DQXError(f"Dialect '{name}' not found in registry. Available dialects: {available}")

    dialect_class = _DIALECT_REGISTRY[name]
    return dialect_class()


def auto_register(cls: Type[Dialect]) -> Type[Dialect]:
    """Decorator to automatically register a dialect class.

    Usage:
        @auto_register
        class MyDialect:
            name = "mydialect"
            ...

    Args:
        cls: The dialect class to register

    Returns:
        The same class (unchanged)
    """
    # Create instance to get the dialect name
    instance = cls()
    register_dialect(instance.name, cls)
    return cls


@auto_register
class DuckDBDialect:
    """DuckDB SQL dialect implementation.

    This dialect generates SQL compatible with DuckDB's syntax,
    including its specific functions like COUNT_IF and FIRST.
    """

    name = "duckdb"

    def translate_sql_op(self, op: ops.SqlOp) -> str:
        """Translate SqlOp to DuckDB SQL syntax."""

        # Pattern matching for different SqlOp types
        match op:
            case ops.NumRows():
                return f"CAST(COUNT(*) AS DOUBLE) AS '{op.sql_col}'"

            case ops.Average(column=col):
                return f"CAST(AVG({col}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.Minimum(column=col):
                return f"CAST(MIN({col}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.Maximum(column=col):
                return f"CAST(MAX({col}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.Sum(column=col):
                return f"CAST(SUM({col}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.Variance(column=col):
                return f"CAST(VARIANCE({col}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.First(column=col):
                return f"CAST(FIRST({col}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.NullCount(column=col):
                return f"CAST(COUNT_IF({col} IS NULL) AS DOUBLE) AS '{op.sql_col}'"

            case ops.NegativeCount(column=col):
                return f"CAST(COUNT_IF({col} < 0.0) AS DOUBLE) AS '{op.sql_col}'"

            case ops.DuplicateCount(columns=cols):
                # For duplicate count: COUNT(*) - COUNT(DISTINCT (col1, col2, ...))
                # Columns are already sorted in the op
                if len(cols) == 1:
                    distinct_expr = cols[0]
                else:
                    distinct_expr = f"({', '.join(cols)})"
                return f"CAST(COUNT(*) - COUNT(DISTINCT {distinct_expr}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.CountValues(column=col, _values=vals, _is_single=is_single):
                # Build the condition for COUNT_IF
                if is_single:
                    val = vals[0]
                    if isinstance(val, str):
                        # Escape single quotes and backslashes
                        escaped_val = val.replace("\\", "\\\\").replace("'", "''")
                        condition = f"{col} = '{escaped_val}'"
                    elif isinstance(val, bool):
                        # Handle boolean values - DuckDB uses TRUE/FALSE
                        condition = f"{col} = {'TRUE' if val else 'FALSE'}"
                    else:
                        condition = f"{col} = {val}"
                else:
                    # Multiple values - use IN operator
                    if isinstance(vals[0], str):
                        # String values - escape and quote each
                        # Type narrowing for mypy: if first element is str, all are str
                        escaped_vals = [str(v).replace("\\", "\\\\").replace("'", "''") for v in vals]
                        values_list = ", ".join(f"'{v}'" for v in escaped_vals)
                    else:
                        # Integer values
                        values_list = ", ".join(str(v) for v in vals)
                    condition = f"{col} IN ({values_list})"
                return f"CAST(COUNT_IF({condition}) AS DOUBLE) AS '{op.sql_col}'"

            case ops.UniqueCount(column=col):
                return f"CAST(COUNT(DISTINCT {col}) AS DOUBLE) AS '{op.sql_col}'"

            case _:
                raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")

    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build CTE query.

        Delegates to the standalone build_cte_query function which can be
        used by other dialects as well.
        """
        return build_cte_query(cte_sql, select_expressions)

    def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
        """Build batch CTE query using array format for DuckDB.

        This method uses an array of key-value pairs to return metrics,
        providing a uniform structure across all dialects.

        Args:
            cte_data: List of BatchCTEData objects containing:
                - key: ResultKey with the date
                - cte_sql: CTE SQL for this date
                - ops: List of SqlOp objects to translate

        Returns:
            Complete SQL query with CTEs and array-based results

        Example output:
            WITH
              source_2024_01_01_0 AS (...),
              metrics_2024_01_01_0 AS (SELECT ... FROM source_2024_01_01_0)
            SELECT '2024-01-01' as date,
                   [{'key': 'x_1', 'value': "x_1"}, {'key': 'x_2', 'value': "x_2"}] as values
            FROM metrics_2024_01_01_0
        """

        def format_array_values(ops: list[ops.SqlOp]) -> str:
            """Format ops as DuckDB array of key-value pairs.

            Args:
                ops: List of SqlOp objects

            Returns:
                SQL array expression
            """
            array_entries = []
            for op in ops:
                # DuckDB syntax: {'key': 'name', 'value': column}
                array_entries.append(f"{{'key': '{op.sql_col}', 'value': \"{op.sql_col}\"}}")
            return "[" + ", ".join(array_entries) + "]"

        return _build_batch_query_with_values(self, cte_data, format_array_values)


@auto_register
class BigQueryDialect:
    """BigQuery SQL dialect implementation.

    This dialect generates SQL compatible with BigQuery's syntax,
    including COUNTIF, VAR_SAMP, and STRUCT-based batch queries.
    """

    name = "bigquery"

    def translate_sql_op(self, op: ops.SqlOp) -> str:
        """Translate SqlOp to BigQuery SQL syntax."""
        match op:
            case ops.NumRows():
                return f"CAST(COUNT(*) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Average(column=col):
                return f"CAST(AVG({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Minimum(column=col):
                return f"CAST(MIN({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Maximum(column=col):
                return f"CAST(MAX({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Sum(column=col):
                return f"CAST(SUM({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.Variance(column=col):
                return f"CAST(VAR_SAMP({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.First(column=col):
                # Using MIN for deterministic "first" value
                return f"CAST(MIN({col}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.NullCount(column=col):
                return f"CAST(COUNTIF({col} IS NULL) AS FLOAT64) AS `{op.sql_col}`"

            case ops.NegativeCount(column=col):
                return f"CAST(COUNTIF({col} < 0) AS FLOAT64) AS `{op.sql_col}`"

            case ops.DuplicateCount(columns=cols):
                # For duplicate count: COUNT(*) - COUNT(DISTINCT (col1, col2, ...))
                # Columns are already sorted in the op
                if len(cols) == 1:
                    distinct_expr = cols[0]
                else:
                    distinct_expr = f"({', '.join(cols)})"
                return f"CAST(COUNT(*) - COUNT(DISTINCT {distinct_expr}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.CountValues(column=col, _values=vals, _is_single=is_single):
                # Build the condition for COUNTIF
                if is_single:
                    val = vals[0]
                    if isinstance(val, str):
                        # Escape single quotes and backslashes
                        escaped_val = val.replace("\\", "\\\\").replace("'", "''")
                        condition = f"{col} = '{escaped_val}'"
                    elif isinstance(val, bool):
                        # Handle boolean values - BigQuery uses TRUE/FALSE
                        condition = f"{col} = {'TRUE' if val else 'FALSE'}"
                    else:
                        condition = f"{col} = {val}"
                else:
                    # Multiple values - use IN operator
                    if isinstance(vals[0], str):
                        # String values - escape and quote each
                        # Type narrowing for mypy: if first element is str, all are str
                        escaped_vals = [str(v).replace("\\", "\\\\").replace("'", "''") for v in vals]
                        values_list = ", ".join(f"'{v}'" for v in escaped_vals)
                    else:
                        # Integer values
                        values_list = ", ".join(str(v) for v in vals)
                    condition = f"{col} IN ({values_list})"
                return f"CAST(COUNTIF({condition}) AS FLOAT64) AS `{op.sql_col}`"

            case ops.UniqueCount(column=col):
                return f"CAST(COUNT(DISTINCT {col}) AS FLOAT64) AS `{op.sql_col}`"

            case _:
                raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")

    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build CTE query using the common helper."""
        return build_cte_query(cte_sql, select_expressions)

    def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
        """Build batch CTE query using array format for BigQuery.

        This method generates a query that returns results as:
        - date: The date string
        - values: An array of STRUCTs with key and value fields

        This uniform array approach allows UNION ALL across different dates
        with different metrics, solving the STRUCT schema mismatch issue.

        Args:
            cte_data: List of BatchCTEData objects

        Returns:
            Complete SQL query with CTEs and array-based results

        Example output:
            WITH
              source_2024_01_01_0 AS (...),
              metrics_2024_01_01_0 AS (SELECT ... FROM source_2024_01_01_0)
            SELECT '2024-01-01' as date,
                   [STRUCT('x_1' AS key, `x_1` AS value),
                    STRUCT('x_2' AS key, `x_2` AS value)] as values
            FROM metrics_2024_01_01_0
        """

        def format_array_values(ops: list[ops.SqlOp]) -> str:
            """Format ops as BigQuery array of key-value STRUCTs.

            Args:
                ops: List of SqlOp objects

            Returns:
                SQL array expression
            """
            array_entries = []
            for op in ops:
                # BigQuery syntax: STRUCT('name' AS key, column AS value)
                array_entries.append(f"STRUCT('{op.sql_col}' AS key, `{op.sql_col}` AS value)")
            return "[" + ", ".join(array_entries) + "]"

        return _build_batch_query_with_values(self, cte_data, format_array_values)
