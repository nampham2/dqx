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

    from dqx.dialect import build_cte_query

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

from typing import TYPE_CHECKING, Protocol, Type, runtime_checkable

from dqx import ops
from dqx.common import DQXError

if TYPE_CHECKING:
    from dqx.models import BatchCTEData


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


class DuckDBDialect:
    """DuckDB SQL dialect implementation.

    This dialect generates SQL compatible with DuckDB's syntax,
    including its specific functions like COUNT_IF and FIRST.
    """

    name = "duckdb"

    def _build_cte_parts(self, cte_data: list["BatchCTEData"]) -> tuple[list[str], list[tuple[str, list[ops.SqlOp]]]]:
        """Build CTE parts for batch query.

        Args:
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
                expressions = [self.translate_sql_op(op) for op in data.ops]
                metrics_select = ", ".join(expressions)
                cte_parts.append(f"{metrics_cte} AS (SELECT {metrics_select} FROM {source_cte})")

                # Store metrics info for later use
                metrics_info.append((metrics_cte, list(data.ops)))

        return cte_parts, metrics_info

    def _validate_metrics(self, metrics_info: list[tuple[str, list[ops.SqlOp]]]) -> None:
        """Validate that metrics exist to compute.

        Args:
            metrics_info: List of (metrics_cte_name, ops) tuples

        Raises:
            ValueError: If no metrics to compute
        """
        if not metrics_info:
            raise ValueError("No metrics to compute")

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

            case _:
                raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")

    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build CTE query.

        Delegates to the standalone build_cte_query function which can be
        used by other dialects as well.
        """
        return build_cte_query(cte_sql, select_expressions)

    def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
        """Build batch CTE query using MAP for DuckDB.

        This method uses DuckDB's MAP feature to return all metrics as a single
        MAP per date, reducing the result from N*M rows to just N rows.

        Args:
            cte_data: List of BatchCTEData objects containing:
                - key: ResultKey with the date
                - cte_sql: CTE SQL for this date
                - ops: List of SqlOp objects to translate

        Returns:
            Complete SQL query with CTEs and MAP-based results

        Example output:
            WITH
              source_2024_01_01_0 AS (...),
              metrics_2024_01_01_0 AS (SELECT ... FROM source_2024_01_01_0)
            SELECT '2024-01-01' as date, MAP {'x_1': "x_1", 'x_2': "x_2"} as values
            FROM metrics_2024_01_01_0
        """
        # Use helper to build CTE parts
        cte_parts, metrics_info = self._build_cte_parts(cte_data)

        # Validate metrics
        self._validate_metrics(metrics_info)

        # Build MAP-based SELECT statements
        map_selects = []
        for i, (data, (metrics_cte, data_ops)) in enumerate(zip(cte_data, metrics_info)):
            date_str = data.key.yyyy_mm_dd.isoformat()

            # Build MAP entries
            map_entries = [f"'{op.sql_col}': \"{op.sql_col}\"" for op in data_ops]
            map_expr = "MAP {" + ", ".join(map_entries) + "}"

            map_selects.append(f"SELECT '{date_str}' as date, {map_expr} as values FROM {metrics_cte}")

        # Build final query
        cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
        union_clause = "\n".join(f"{'UNION ALL' if i > 0 else ''}\n{select}" for i, select in enumerate(map_selects))

        return f"{cte_clause}\n{union_clause}"


# Dialect Registry
_DIALECT_REGISTRY: dict[str, Type[Dialect]] = {}


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


# Register built-in dialects
register_dialect("duckdb", DuckDBDialect)
