# Active Context

## Current Work Focus

### Batch SQL Optimization with MAP (Completed - 2025-10-23)
- Implemented MAP-based batch SQL optimization for DuckDB dialect
- Replaced UNPIVOT approach with MAP aggregation to reduce result set size
- Benefits: N rows instead of N*M rows (where N=dates, M=metrics)
- Maintains backward compatibility with existing analyzer interface

## Recent Changes

### Batch SQL MAP Optimization Implementation (2025-10-23)
- Modified DuckDBDialect.build_batch_cte_query to use MAP aggregation
- Key changes:
  - Helper methods: _build_cte_parts and _validate_metrics for cleaner code
  - MAP syntax creates dictionary of metric_name->value pairs per date
  - Result set is now [(date, {metric1: value1, metric2: value2, ...})]
- Updated analyzer.py to process MAP results instead of unpivot rows
- Added comprehensive tests for both dialect and analyzer changes
- Performance improvement: ~90% reduction in result rows for multi-metric queries

### Batch Analysis Lag Fix (Completed - 2025-10-23)
- Fixed critical bug where metrics with lagged dates were incorrectly deduplicated
- Root cause: SqlOp equality doesn't consider date context, causing value propagation issues
- Solution: Changed deduplication to group by (date, SqlOp) pairs instead of just SqlOp
- All 746 tests now passing with the fix
- Maintains batch SQL efficiency while ensuring correctness

### Check Decorator Simplification (2025-10-22)
- Removed tags parameter from @check decorator in api.py
- Updated CheckNode to remove tags field
- Cleaned up all references to tags in tests
- Decorator now only accepts name and severity parameters

### DataSource Consolidation (2025-10-22)
- Created new datasource.py module containing:
  - ArrowDataSource
  - DuckDataSource
  - InMemoryMetricDB
- Removed src/dqx/extensions/ directory
- Updated all imports from `dqx.extensions` to `dqx.datasource`
- Maintained all functionality with cleaner structure

### Extended Metric Symbol Display Fix (2025-10-22)
- Fixed bug where extended metrics (day_over_day, stddev) displayed only base metric names
- Issue: SymbolInfo was using `str(symbolic_metric.metric_spec)` instead of `symbolic_metric.name`
- Fix: Changed line 573 in api.py to use `metric=symbolic_metric.name`
- Added test_extended_metric_symbol_info.py to verify correct behavior

### Plugin Instance Registration Implementation (2025-10-21)
- Successfully implemented PostProcessor instance support in register_plugin method
- Added overloaded register_plugin method supporting both str and PostProcessor
- Implemented thorough validation including protocol checking and metadata validation

## Next Steps

1. Consider implementing PostgreSQL dialect support (planned for v0.4.0)
2. Add performance profiling tools to identify bottlenecks
3. Implement graph visualization capabilities
4. Create CLI interface for running checks from command line
5. Explore additional built-in plugins for the ecosystem

### Tags Parameter Removal from @check Decorator (2025-10-22)
- Successfully removed the unused tags parameter from @check decorator
- Simplified the decorator signature to only require name and severity
- Updated all tests and examples to remove tags usage
- Maintained backward compatibility is not needed as tags were never used

### DataSource Module Refactoring (2025-10-22)
- Moved data source implementations from extensions/ to datasource.py module
- Removed the extensions/ directory entirely after migration
- Updated all imports throughout the codebase
- Cleaner module structure with data sources at top level

## Important Patterns and Preferences

### Git Workflow
- Using conventional commits (enforced by commitizen)
- Pre-commit hooks run comprehensive checks including:
  - Python syntax validation
  - Code formatting (ruff)
  - Type checking (mypy)
  - Security checks
  - File quality checks
  - YAML validation (yamllint)
  - Shell script formatting (shfmt) and validation (shellcheck)

### Testing Standards
- Maintain 100% test coverage (currently at 731 passing tests)
- Use timer fallback patterns for resilient plugin execution
- Follow TDD approach for new features
- All changes must pass ruff and mypy checks

### Code Organization
- Data sources now live in datasource.py at top level
- Extensions directory removed for cleaner structure
- Plugins remain in their own module with protocol-based design
- Graph functionality organized under graph/ subdirectory

## Learnings and Insights

### Batch Analysis
- SqlOp equality is operation-based, not context-based (doesn't consider dates)
- Deduplication must consider the full context (date + operation) for correctness
- Batch SQL can still deduplicate across dates for efficiency, but value assignment must respect boundaries
- Empty metric handling is important to avoid unnecessary batch analysis calls

### MAP-based SQL Optimization
- DuckDB's MAP type is powerful for pivoting data without UNPIVOT
- MAP reduces result set from N*M rows to N rows (N=dates, M=metrics)
- Syntax: `MAP {key1: value1, key2: value2}` creates inline dictionaries
- Result processing is simpler with MAP - one row per date with all metrics
- Maintains same analyzer interface while improving performance

### Module Organization
- Moving data sources to top level improves discoverability
- Removing unnecessary directory structure (extensions/) simplifies imports
- Protocol-based design allows clean separation without deep hierarchies

### API Design
- Removing unused parameters (like tags) simplifies the API
- Required parameters (name, severity) enforce good practices
- Two-stage assertion building pattern works well for ensuring names

### Testing Strategy
- Comprehensive test coverage catches refactoring issues early
- Pre-commit hooks prevent bad code from entering the repository
- Running tests with -v flag helps track progress during large test suites
- End-to-end tests are crucial for catching integration issues
