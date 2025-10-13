# Dialect Implementation Summary

## Overview

This document summarizes the implementation of SQL dialect support in DQX, which allows the framework to support different SQL databases beyond DuckDB.

## Architecture Decision: Hybrid Approach

After analyzing the pros and cons of adding dialect support in either the Analyzer or DataSource, we implemented a **hybrid solution** that leverages the strengths of both approaches:

### Key Design Decisions

1. **Dialect added to DataSource Protocol**
   - Each DataSource now has a `dialect` attribute
   - DataSources are responsible for providing the appropriate dialect
   - Default is DuckDBDialect for backward compatibility

2. **Analyzer made dialect-aware**
   - Analyzer uses the DataSource's dialect for SQL generation
   - Falls back to legacy SQL generation for DataSources without dialect
   - Maintains backward compatibility

## Implementation Details

### 1. Updated SqlDataSource Protocol
```python
@runtime_checkable
class SqlDataSource(Protocol):
    name: str
    dialect: str  # The SQL dialect name used for query generation
    # ... rest of the protocol
```

### 2. Modified Analyzer
```python
def analyze_sql_ops(ds: T, ops: Sequence[SqlOp]) -> None:
    # ...
    if hasattr(ds, "dialect"):
        # Use dialect for SQL generation
        expressions = [ds.dialect.translate_sql_op(op) for op in distinct_ops]
        sql = ds.dialect.build_cte_query(ds.cte, expressions)
    else:
        # Legacy behavior for backward compatibility
        sql = textwrap.dedent(
            f"""
            WITH source AS ( {ds.cte} )
            SELECT {", ".join(op.sql for op in distinct_ops)} FROM source
            """
        )
```

### 3. Updated DataSource Implementations

All existing DataSource implementations now accept an optional dialect parameter:

```python
class ArrowDataSource:
    def __init__(
        self, table: pa.RecordBatch | pa.Table, dialect: Dialect | None = None
    ):
        self._table = table
        self._table_name = random_prefix(k=6)
        self.dialect = dialect or DuckDBDialect()
```

## Benefits of This Approach

1. **Clean Separation of Concerns**
   - DataSources handle data access and provide dialect information
   - Analyzer focuses on analysis logic
   - Dialect handles SQL translation

2. **Flexibility**
   - Easy to add new database support by implementing new dialects
   - DataSources can be configured with any dialect
   - Analyzer can apply optimizations based on dialect

3. **Backward Compatibility**
   - Existing code continues to work without changes
   - DataSources without dialect attribute use legacy SQL generation
   - Default dialect is DuckDB for all DataSource implementations

4. **Reusability**
   - Dialect implementations can be shared across different DataSources
   - The `build_cte_query` function provides beautiful formatting for any dialect
   - SQL translation logic is centralized in dialect classes

## Example: Adding PostgreSQL Support

```python
class PostgreSQLDialect:
    name = "postgresql"

    def translate_sql_op(self, op: ops.SqlOp) -> str:
        match op:
            case ops.NumRows():
                return f"COUNT(*)::FLOAT8 AS {op.sql_col}"
            case ops.NullCount(column=col):
                # PostgreSQL doesn't have COUNT_IF
                return (
                    f"COUNT(CASE WHEN {col} IS NULL THEN 1 END)::FLOAT8 AS {op.sql_col}"
                )
            # ... handle other operations

    def build_cte_query(self, cte_sql: str, select_expressions: list[str]) -> str:
        from dqx.dialect import build_cte_query

        return build_cte_query(cte_sql, select_expressions)


# Usage
ds = ArrowDataSource(table, dialect=PostgreSQLDialect())
analyzer = Analyzer()
report = analyzer.analyze(ds, metrics, key)
```

## Testing

The implementation includes comprehensive tests that verify:
- Default DuckDB dialect behavior
- Custom dialect support
- Dialect-aware SQL generation in Analyzer
- Beautiful query formatting
- Backward compatibility for DataSources without dialect

All tests pass and type checking is successful.

## Future Enhancements

1. **Add more dialect implementations** (PostgreSQL, MySQL, SQLite, etc.)
2. **Dialect-specific optimizations** in the Analyzer
3. **Dialect feature detection** (e.g., support for window functions)
4. **Query execution abstraction** to handle different result formats

## Conclusion

The hybrid approach successfully balances flexibility, maintainability, and backward compatibility. It provides a solid foundation for supporting multiple SQL databases in DQX while keeping the codebase clean and extensible.
