# Progress

## What Works
- Core DQX functionality (data quality checks, assertions, metrics)
- Plugin system with audit plugin showing execution summaries
- Pre-commit hooks integrated with the project
- BigQuery dialect support
- Batch SQL optimization for performance
- Dataset validation with clear error messages
- Comprehensive test coverage (100%)
- **New Python-based hooks command (`uv run hooks`) replacing shell script**
- **UniqueCount metric for counting distinct values in columns**
- **Enhanced NonMergeable state supporting multiple metric types**
- **Comprehensive test suite for compute module functions**

## What's Left to Build
- Additional plugin types beyond audit
- More database dialect support
- Enhanced documentation
- Performance optimizations for very large datasets

## Current Status
- Project is stable and production-ready
- Active development on new features
- Regular maintenance and bug fixes

## Known Issues
- None currently reported

## Evolution of Project Decisions

### 2025-10-29: Added Comprehensive Tests for Compute Module
- Created test_compute.py with full test coverage for all compute functions
- Tests cover simple_metric, day_over_day, week_over_week, and stddev functions
- Added tests for helper functions _timeseries_check and _sparse_timeseries_check
- All tests pass with proper type hints and linting compliance
- Includes edge cases: missing data, division by zero, negative values, and empty databases

### 2025-10-28: Implemented UniqueCount Metric
- Added UniqueCount operation for counting distinct values
- Leveraged existing NonMergeable state class instead of creating new one
- Enhanced NonMergeable to support metric_type parameter
- Updated DuplicateCount to use NonMergeable state for consistency
- Comprehensive test coverage including edge cases

### 2025-01-27: Replaced Shell Script with Python Command
- Replaced `bin/run-hooks.sh` with `uv run hooks` command
- Implementation in `scripts/commands.py` provides better cross-platform support
- All functionality preserved with improved argument parsing
- Updated all documentation references to use new command

### Recent Major Changes
- Added data discrepancy display to AuditPlugin
- Implemented count values operation
- Removed Mock usage from test files
- Enhanced plugin execution context with trace analysis

### Architecture Decisions
- Plugin-based architecture for extensibility
- Separation of concerns between analysis, evaluation, and reporting
- Use of PyArrow for efficient data handling
- Batch SQL optimization for improved performance
