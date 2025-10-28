# Active Context

## Current Work Focus
- Successfully implemented UniqueCount metric for counting distinct values in columns
- Leveraged existing NonMergeable state infrastructure to reduce code duplication

## Recent Changes (2025-10-28)
### Just Completed:
- Added `UniqueCount` operation to `ops.py`:
  - Inherits from `SqlOp` for SQL generation
  - Generates `COUNT(DISTINCT column)` SQL expressions
  - Properly handles column name escaping for both DuckDB and BigQuery

- Enhanced `NonMergeable` state class in `states.py`:
  - Added `metric_type` parameter for better identification
  - Now shared between UniqueCount and DuplicateCount metrics
  - Reduces code duplication and maintains consistency

- Added `UniqueCount` spec to `specs.py`:
  - Uses `NonMergeable` state with `metric_type="UniqueCount"`
  - Follows established patterns for metric specifications
  - Properly integrated into the registry

- Enhanced `MetricProvider` in `provider.py`:
  - Added `unique_count()` method for easy API access
  - Returns proper SymbolicMetric instances

- SQL Dialect Support in `dialect.py`:
  - Added support for both DuckDB and BigQuery
  - Handles column quoting and escaping properly
  - Comprehensive test coverage

### Also Completed:
- Updated DuplicateCount to use NonMergeable state
- Modified related tests to accommodate the state changes
- Created comprehensive API integration tests
- All 243 tests passing

## Key Implementation Details
- Decision to use existing `NonMergeable` state instead of creating new state class
- COUNT(DISTINCT) properly excludes NULL values as expected
- Comprehensive test coverage including edge cases (empty data, all nulls, etc.)
- Type hints and docstrings throughout

## Command Usage
- Through MetricProvider: `mp.unique_count("customer_id")`
- Direct spec usage: `specs.UniqueCount("customer_id")`
- Works with all data types supported by the framework

## Previous Work (2025-01-27)
- Successfully implemented `uv run hooks` command to replace shell script
- Python commands in DQX are implemented in `scripts/commands.py`
- Commands are exposed via `[project.scripts]` in `pyproject.toml`

## Important Patterns and Preferences
- Prefer reusing existing infrastructure over creating duplicate functionality
- NonMergeable states are appropriate for metrics that cannot be incrementally computed
- Always include comprehensive test coverage including edge cases
- Follow established patterns for consistency
- Use type hints and docstrings throughout
- No backward compatibility needed unless explicitly requested

## Learnings and Project Insights
- The project has good infrastructure for adding new metrics
- State classes can be shared between metrics with similar characteristics
- The registry pattern makes it easy to add new metric types
- SQL dialect abstraction allows supporting multiple databases cleanly
- Test coverage is critical - the project maintains high standards
