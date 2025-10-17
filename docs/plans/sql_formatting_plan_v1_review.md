# SQL Formatting Plan v1 - Review

## Executive Summary
The SQL formatting plan has good intentions but needs significant revision based on the clarified requirements. The original plan is over-engineered and would introduce regression in formatting quality.

## Key Issues with Original Plan

### 1. Formatting Quality Regression
The current `build_cte_query` function provides sophisticated column alignment:
```sql
SELECT
    COUNT(*)      AS 'total_count'
  , AVG(price)    AS 'avg_price'
  , SUM(quantity) AS 'total_quantity'
```

The proposed sqlparse implementation cannot achieve this vertical alignment, producing instead:
```sql
SELECT
    COUNT(*) AS 'total_count',
    AVG(price) AS 'avg_price',
    SUM(quantity) AS 'total_quantity'
```

### 2. Architectural Complexity
- Multiple formatting points instead of centralized control
- Violates DRY by calling `format_sql()` from multiple locations
- Adds unnecessary dependency when existing solution works well

### 3. KISS/YAGNI Violation
- Introduces external library for something already solved
- More complex than the current implementation
- No clear benefit over existing approach

## Clarified Requirements

Nam clarified that:
1. The complex manual alignment is not needed - it's too much manual work
2. Format SQL at **only one point**: in `analyzer.analyze_sql_ops`
3. Remove formatting logic everywhere else

## Revised Implementation Plan

### Step 1: Simplify build_cte_query
**File**: `src/dqx/dialect.py`

Remove alignment logic and make it a simple concatenation:
```python
def build_cte_query(cte_sql: str, select_expressions: list[str]) -> str:
    """Build CTE query without formatting."""
    if not select_expressions:
        raise ValueError("No SELECT expressions provided")

    select_clause = ", ".join(select_expressions)
    return f"WITH source AS ({cte_sql}) SELECT {select_clause} FROM source"
```

### Step 2: Add sqlparse dependency
**File**: `pyproject.toml`
- Add `"sqlparse>=0.5.0",` to dependencies
- Run `uv sync`
- Commit dependency changes

### Step 3: Format in analyzer only
**File**: `src/dqx/analyzer.py`

Add formatting at the single control point:
```python
import sqlparse  # Add with other imports

# In analyze_sql_ops, after line 163:
sql = dialect_instance.build_cte_query(ds.cte(nominal_date), expressions)

# Add formatting here:
sql = sqlparse.format(
    sql,
    reindent=True,
    keyword_case='upper',
    identifier_case='lower',
    indent_width=2
)
```

### Step 4: Update tests
- Update tests expecting aligned format
- Ensure all tests pass with new formatting

## Benefits of Revised Approach

1. **Single Point of Control**: Formatting happens in exactly one place
2. **Removes Complexity**: No more manual alignment calculations
3. **Follows KISS**: Simple solution that meets requirements
4. **Easy to Modify**: Change formatting rules in one location
5. **Clear Architecture**: Clean separation of concerns

## Recommendation

Proceed with the **simplified implementation** that:
- Removes complex manual formatting from `build_cte_query`
- Adds sqlparse formatting at a single point in `analyze_sql_ops`
- Maintains clean architecture and follows KISS principles

This approach achieves the goal of consistent SQL formatting while actually simplifying the codebase rather than adding complexity.
