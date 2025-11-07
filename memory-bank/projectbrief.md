# DQX Project Brief

## Project Name
DQX - Data Quality eXcellence (dqlib)

## Version
0.5.9

## Core Purpose
DQX is a data quality validation framework that enables developers to write quality checks as mathematical expressions in Python, validate data efficiently using SQL backends, and get instant feedback on data integrity issues.

## Key Requirements

### Functional Requirements
1. **Native Python API**: Define data quality checks as Python functions, not configuration files
2. **Symbolic Mathematics**: Express complex business rules using SymPy symbolic expressions
3. **SQL Backend**: Execute validation queries directly on data warehouses via DuckDB
4. **Extensible Architecture**: Support custom metrics, validators, and data sources via protocols
5. **Cross-Time Analysis**: Compare metrics across different time periods (lag functionality)
6. **Cross-Dataset Validation**: Combine metrics from multiple data sources in single expressions
7. **Severity Levels**: P0-P3 severity classification for prioritizing issues
8. **Result Persistence**: Store metrics and results in configurable databases
9. **Date Exclusion**: Support for excluding specific dates from calculations (skip_dates)
10. **Custom SQL Operations**: Execute user-defined SQL expressions as metrics

### Non-Functional Requirements
1. **Performance**: Single-pass SQL execution for multiple metrics
2. **Type Safety**: Full type hints and mypy validation
3. **Code Quality**: 100% test coverage, pre-commit hooks, linting
4. **Documentation**: Clear examples and API documentation
5. **Simplicity**: KISS/YAGNI principles - start simple, evolve thoughtfully

## Project Scope

### In Scope
- Data quality validation for structured data (tables, dataframes)
- SQL-based computation on DuckDB, BigQuery, PyArrow
- Metric computation (sum, average, count, cardinality, custom SQL, etc.)
- Assertion validation (equals, greater than, between, etc.)
- Result collection and persistence
- Graph-based dependency resolution
- Date exclusion for data availability handling

### Out of Scope (Removed)
- Batch processing support (removed to simplify architecture)
- Distributed computing via Spark
- Real-time streaming validation
- Unstructured data validation

## Target Users
- Data Engineers validating data pipelines
- Data Scientists ensuring data quality for ML
- Analytics Engineers monitoring data warehouse health
- QA Engineers testing data transformations

## Success Criteria
1. Developers can define complex validation rules in pure Python
2. Validation runs efficiently on large datasets via SQL pushdown
3. Clear, actionable error messages guide remediation
4. Extensible without modifying core framework
5. Minimal operational overhead (no clusters required)

## Core Design Decisions
1. **Code over Configuration**: Python functions instead of YAML/JSON
2. **SQL over Spark**: Direct database execution without cluster overhead
3. **Symbolic Expressions**: Mathematical formulas for business rules
4. **Graph Architecture**: Dependency resolution for optimal execution
5. **Protocol-Based Extensions**: Clean interfaces for customization
6. **3D Methodology**: Design-Driven Development - Think before you build, build with intention, ship with confidence

## Constraints
- Python 3.11+ required (up to 3.12)
- Must maintain backward compatibility within major versions
- All changes must maintain 100% test coverage
- Must follow project coding standards (PEP 8, type hints, modern syntax)
- Development managed via uv package manager
- Distributed as `dqlib` package on PyPI
