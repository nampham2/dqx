from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, Type, runtime_checkable

from dqx import ops
from dqx.common import DQXError, ResultKey
from dqx.utils import freeze_for_hashing

if TYPE_CHECKING:
    from dqx.common import SqlDataSource
    from dqx.ops import SqlOp


@dataclass
class BatchCTEData:
    """Data for building a CTE query."""

    key: ResultKey
    cte_sql: str
    ops: Sequence[SqlOp]

    def group_ops_by_parameters(self) -> dict[tuple, list[SqlOp]]:
        """Group operations by their parameters for efficient CTE generation.

        Operations with identical parameters can share the same source CTE,
        reducing query complexity and improving performance.

        Uses self.ops: list of SQL operations on the instance.

        Returns:
            Dictionary mapping parameter tuples to operation lists
        """
        groups: dict[tuple, list[SqlOp]] = {}

        for op in self.ops:
            # Create hashable key from parameters using freeze_for_hashing
            # This handles complex parameter values like lists, dicts, sets
            params_key = tuple(sorted((k, freeze_for_hashing(v)) for k, v in op.parameters.items()))

            if params_key not in groups:
                groups[params_key] = []
            groups[params_key].append(op)

        return groups


# Dialect Registry
_DIALECT_REGISTRY: dict[str, Type[Dialect]] = {}


def _build_cte_parts(
    dialect: "Dialect", cte_data: list["BatchCTEData"], data_source: "SqlDataSource | None" = None
) -> tuple[list[str], list[tuple[str, list[ops.SqlOp]]]]:
    """Build CTE parts for batch query - shared between dialects.

    Args:
        dialect: The dialect instance to use for SQL translation
        cte_data: List of BatchCTEData objects
        data_source: Optional data source for parameter-aware CTE generation

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

        ops_by_params = data.group_ops_by_parameters()

        # Create CTE for each parameter group
        for j, (params_key, grouped_ops) in enumerate(ops_by_params.items()):
            # Source CTE with parameters
            source_cte = f"source_{date_suffix}_{i}_{j}"

            # Reuse the original parameter mapping so types stay intact
            params_dict = dict(grouped_ops[0].parameters) if grouped_ops else {}

            # Generate parameter-aware CTE SQL if data source provided
            if data_source and params_dict:
                # Pass parameters to data source for optimized CTE generation
                parameterized_cte_sql = data_source.cte(data.key.yyyy_mm_dd, params_dict)
            else:
                # Fall back to provided CTE SQL
                parameterized_cte_sql = data.cte_sql

            cte_parts.append(f"{source_cte} AS ({parameterized_cte_sql})")

            # Metrics CTE for this parameter group
            if grouped_ops:
                metrics_cte = f"metrics_{date_suffix}_{i}_{j}"
                expressions = [dialect.translate_sql_op(op) for op in grouped_ops]
                metrics_select = ", ".join(expressions)
                cte_parts.append(f"{metrics_cte} AS (SELECT {metrics_select} FROM {source_cte})")

                # Store metrics info
                metrics_info.append((metrics_cte, list(grouped_ops)))

    return cte_parts, metrics_info


def _build_query_with_values(
    dialect: "Dialect",
    cte_data: list["BatchCTEData"],
    value_formatter: Callable[[list[ops.SqlOp]], str],
    data_source: "SqlDataSource | None" = None,
) -> str:
    """Build batch query with custom value formatting.

    Args:
        dialect: The dialect instance to use for SQL translation
        cte_data: List of BatchCTEData objects
        value_formatter: Function that formats ops into a value expression (MAP, STRUCT, etc.)
        data_source: Optional data source for parameter-aware CTE generation

    Returns:
        Complete SQL query with CTEs and formatted values

    Raises:
        ValueError: If no CTE data provided or no metrics to compute
    """
    cte_parts, metrics_info = _build_cte_parts(dialect, cte_data, data_source)

    # Simple validation inline
    if not metrics_info:
        raise ValueError("No metrics to compute")

    # Build value selects from metrics_info
    # Map from date to list of selects for that date
    date_to_selects: dict[str, list[str]] = {}

    # Build date mapping from original cte_data
    date_map = {i: data.key.yyyy_mm_dd for i, data in enumerate(cte_data)}

    for metrics_cte, data_ops in metrics_info:
        # Extract date from metrics CTE name (format: metrics_YYYY_MM_DD_i_j)
        parts = metrics_cte.split("_")
        if len(parts) >= 5:  # metrics_YYYY_MM_DD_i_j
            year, month, day = parts[1], parts[2], parts[3]
            date_str = f"{year}-{month}-{day}"
        else:
            # Fallback: get from original data
            # Extract index from CTE name
            idx = int(parts[-2]) if len(parts) >= 2 else 0
            date_str = date_map.get(idx, datetime.date.today()).isoformat()

        values_expr = value_formatter(data_ops)
        select_stmt = f"SELECT '{date_str}' as date, {values_expr} as values FROM {metrics_cte}"

        if date_str not in date_to_selects:
            date_to_selects[date_str] = []
        date_to_selects[date_str].append(select_stmt)

    # Flatten all selects
    value_selects = []
    for date_str in sorted(date_to_selects.keys()):
        value_selects.extend(date_to_selects[date_str])

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

    def build_cte_query(self, cte_data: list["BatchCTEData"], data_source: "SqlDataSource") -> str:
        """Build a batch CTE query for multiple dates.

        Args:
            cte_data: List of BatchCTEData objects containing:
                - key: ResultKey with the date
                - cte_sql: CTE SQL for this date
                - ops: List of SqlOp objects to translate
            data_source: The data source for parameter-aware CTE generation

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

            case ops.CustomSQL():
                # Use the SQL expression directly - no substitution
                return f"CAST(({op.sql_expression}) AS DOUBLE) AS '{op.sql_col}'"

            case _:
                raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")

    def build_cte_query(self, cte_data: list["BatchCTEData"], data_source: "SqlDataSource") -> str:
        """Build batch CTE query using array format for DuckDB.

        This method uses an array of key-value pairs to return metrics,
        providing a uniform structure across all dialects.

        Args:
            cte_data: List of BatchCTEData objects containing:
                - key: ResultKey with the date
                - cte_sql: CTE SQL for this date
                - ops: List of SqlOp objects to translate
            data_source: The data source for parameter-aware CTE generation

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

        return _build_query_with_values(self, cte_data, format_array_values, data_source)


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

            case ops.CustomSQL():
                # Use the SQL expression directly - no substitution
                return f"CAST(({op.sql_expression}) AS FLOAT64) AS `{op.sql_col}`"

            case _:
                raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")

    def build_cte_query(self, cte_data: list["BatchCTEData"], data_source: "SqlDataSource") -> str:
        """Build batch CTE query using array format for BigQuery.

        This method generates a query that returns results as:
        - date: The date string
        - values: An array of STRUCTs with key and value fields

        This uniform array approach allows UNION ALL across different dates
        with different metrics, solving the STRUCT schema mismatch issue.

        Args:
            cte_data: List of BatchCTEData objects
            data_source: The data source for parameter-aware CTE generation

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

        return _build_query_with_values(self, cte_data, format_array_values, data_source)
