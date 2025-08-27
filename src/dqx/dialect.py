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

from typing import Protocol, runtime_checkable, Type

from dqx import ops
from dqx.common import DQXError


def build_cte_query(cte_sql: str, select_expressions: list[str]) -> str:
    """Build beautifully formatted CTE query with aligned columns.
    
    This function generates a CTE (Common Table Expression) query with
    properly formatted and aligned SELECT expressions. It can be used
    by any SQL dialect that supports standard CTE syntax.
    
    Args:
        cte_sql: The CTE SQL expression (e.g., "SELECT * FROM table")
        select_expressions: List of SQL expressions to select
        
    Returns:
        Formatted SQL query string with proper indentation and alignment
        
    Raises:
        ValueError: If no SELECT expressions are provided
        
    Example:
        >>> expressions = [
        ...     "COUNT(*) AS 'total_count'",
        ...     "AVG(price) AS 'avg_price'",
        ...     "SUM(quantity) AS 'total_quantity'"
        ... ]
        >>> query = build_cte_query("SELECT * FROM orders", expressions)
        >>> print(query)
        WITH source AS (
            SELECT * FROM orders
        )
        SELECT
            COUNT(*)      AS 'total_count'
          , AVG(price)    AS 'avg_price'
          , SUM(quantity) AS 'total_quantity'
        FROM source
    """
    if not select_expressions:
        raise ValueError("No SELECT expressions provided")
    
    # Split expressions to find alignment point
    expression_parts = []
    for expr in select_expressions:
        parts = expr.split(" AS ", 1)
        if len(parts) == 2:
            expression_parts.append((parts[0].strip(), parts[1].strip()))
        else:
            # Handle expressions without AS clause
            expression_parts.append((expr.strip(), ""))
    
    # Find the longest expression for alignment
    max_expr_length = max(len(expr) for expr, _ in expression_parts)
    
    # Format expressions with alignment
    formatted_expressions = []
    for i, (expr, alias) in enumerate(expression_parts):
        if alias:
            formatted = f"{expr.ljust(max_expr_length)} AS {alias}"
        else:
            formatted = expr
        
        if i == 0:
            formatted_expressions.append(f"    {formatted}")
        else:
            formatted_expressions.append(f"  , {formatted}")
    
    # Build the formatted query
    return (
        f"WITH source AS (\n"
        f"    {cte_sql}\n"
        f")\n"
        f"SELECT\n"
        f"{chr(10).join(formatted_expressions)}\n"
        f"FROM source"
    )


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
        """Build a complete CTE query with beautiful formatting.
        
        Args:
            cte_sql: The CTE SQL expression (e.g., "SELECT * FROM table")
            select_expressions: List of SQL expressions to select
            
        Returns:
            Formatted SQL query string with proper indentation and alignment
        """
        ...


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
            
            case _:
                raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")
    
    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        """Build beautifully formatted CTE query with aligned columns.
        
        Delegates to the standalone build_cte_query function which can be
        used by other dialects as well.
        """
        return build_cte_query(cte_sql, select_expressions)


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
