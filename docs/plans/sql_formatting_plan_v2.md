# SQL Formatting Implementation Plan v2

## Overview
This plan implements SQL formatting at a single point in DQX using the `sqlparse` library. Based on review feedback, this approach simplifies the codebase by removing complex manual formatting and centralizing all formatting logic.

## Key Changes from v1
- Format SQL at **only one location**: `analyzer.analyze_sql_ops`
- Remove complex manual alignment from `build_cte_query`
- No separate formatter module needed
- Follows KISS principle by removing complexity

## Goals
1. Simplify `build_cte_query` to remove manual formatting
2. Add sqlparse dependency
3. Apply formatting at single control point in analyzer
4. Maintain all tests and 100% coverage

## Implementation Tasks

### Task Group 1: Add sqlparse Dependency
**Goal**: Add sqlparse library to project dependencies

1. **Update dependencies**
   - File: `pyproject.toml`
   - Add `"sqlparse>=0.5.0",` to the dependencies list

2. **Install the dependency**
   ```bash
   uv sync
   ```

3. **Commit the dependency**
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "build: add sqlparse dependency for SQL formatting"
   ```

### Task Group 2: Simplify build_cte_query
**Goal**: Remove complex manual formatting logic

1. **Update build_cte_query function**
   - File: `src/dqx/dialect.py`
   - Remove all alignment logic and simplify:
   ```python
   def build_cte_query(cte_sql: str, select_expressions: list[str]) -> str:
       """Build CTE query without formatting."""
       if not select_expressions:
           raise ValueError("No SELECT expressions provided")

       select_clause = ", ".join(select_expressions)
       return f"WITH source AS ({cte_sql}) SELECT {select_clause} FROM source"
   ```

2. **Update tests expecting aligned format**
   - File: `tests/test_dialect.py`
   - Update tests to expect simple concatenated format (no alignment)
   - The SQL will be unformatted at this stage

3. **Run tests and commit**
   ```bash
   uv run pytest tests/test_dialect.py -v
   git add src/dqx/dialect.py tests/test_dialect.py
   git commit -m "refactor: simplify build_cte_query to remove manual formatting"
   ```

### Task Group 3: Add Formatting to Analyzer
**Goal**: Apply SQL formatting at the single control point

1. **Update analyzer.py**
   - File: `src/dqx/analyzer.py`
   - Add import at the top with other imports:
   ```python
   import sqlparse
   ```

2. **Add formatting in analyze_sql_ops**
   - Locate the line where SQL is built (around line 163):
   ```python
   sql = dialect_instance.build_cte_query(ds.cte(nominal_date), expressions)
   ```
   - Add formatting immediately after:
   ```python
   # Format SQL for consistent output
   sql = sqlparse.format(
       sql,
       reindent=True,
       keyword_case='upper',
       identifier_case='lower',
       indent_width=2,
       wrap_after=120,
       comma_first=False
   )
   ```

3. **Update analyzer tests**
   - File: `tests/test_analyzer.py`
   - Update any tests that check SQL output to expect formatted SQL
   - The SQL should now have proper formatting

4. **Run tests and commit**
   ```bash
   uv run pytest tests/test_analyzer.py -v
   git add src/dqx/analyzer.py tests/test_analyzer.py
   git commit -m "feat: add SQL formatting to analyzer at single control point"
   ```

### Task Group 4: Update Remaining Tests
**Goal**: Fix all tests expecting old format

1. **Identify affected tests**
   - Run full test suite to find failures:
   ```bash
   uv run pytest tests/ -v
   ```

2. **Update test expectations**
   - Files likely affected:
     - `tests/test_api.py` - May check SQL output
     - `tests/e2e/` - End-to-end tests
     - Any tests checking log output containing SQL

3. **Fix each test file**
   - Update SQL string expectations to match sqlparse output
   - Focus on tests that fail due to formatting changes only

4. **Run tests and commit**
   ```bash
   uv run pytest tests/ -v
   git add tests/
   git commit -m "test: update tests for new SQL formatting"
   ```

### Task Group 5: Final Validation
**Goal**: Ensure quality standards are met

1. **Verify all tests pass**
   ```bash
   uv run pytest tests/ -v --cov=dqx --cov-report=term-missing
   ```
   - Ensure 100% coverage maintained

2. **Run type checking**
   ```bash
   uv run mypy src/
   ```

3. **Run pre-commit hooks**
   ```bash
   ./bin/run-hooks.sh
   ```

4. **Create example showing formatted output**
   - Create `examples/sql_formatting_demo.py`:
   ```python
   """Demonstrate SQL formatting in DQX."""

   import datetime
   from dqx import DataQualityAnalyzer
   from dqx.models import Check, Column, ColumnMetric

   # Example showing formatted SQL output in logs
   analyzer = DataQualityAnalyzer()

   # Create sample check
   check = Check(
       name="row_count_check",
       columns=[Column(name="*", metrics=[ColumnMetric(name="row_count")])]
   )

   # This will show formatted SQL in the output
   # ... (complete example)
   ```

5. **Final commit**
   ```bash
   git add examples/sql_formatting_demo.py
   git commit -m "docs: add SQL formatting demo"
   ```

## Benefits of This Approach

1. **Simplicity**: Removes complex code instead of adding it
2. **Single Control Point**: All formatting happens in one place
3. **Maintainability**: Easy to modify formatting rules
4. **Clean Architecture**: Clear separation of concerns
5. **Follows KISS/YAGNI**: Solves the problem with minimal complexity

## Testing Strategy

- Unit tests verify `build_cte_query` produces simple concatenated SQL
- Integration tests verify analyzer produces formatted SQL
- End-to-end tests ensure the complete flow works correctly

## Rollback Plan

If issues arise:
1. Keep sqlparse dependency (no harm)
2. Remove formatting call from analyzer
3. Existing functionality continues to work

## Success Criteria

1. Complex manual formatting removed from `build_cte_query`
2. SQL formatting applied at single point in analyzer
3. All tests pass with 100% coverage
4. Cleaner, simpler codebase

## Implementation Notes

- This is a simplification, not an enhancement
- Focus on removing code, not adding it
- Each commit should maintain a working state
- The goal is better maintainability through simplicity
