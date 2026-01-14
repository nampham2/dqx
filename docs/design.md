# DQX Design Document

## 1. Overview

### What is DQX

DQX is a data quality analysis framework that enables developers to define and run data quality checks using native Python code. It provides a SQL-based backend for efficient computation and supports extensible metrics, validators, and data sources.

### Core Design Philosophy

DQX prioritizes simplicity, composability, and performance:
- **Code over configuration**: Define checks in Python, not through configuration
- **SQL-powered**: Execute queries directly on data warehouses
- **Modular architecture**: Extend with custom metrics and validators
- **Symbolic computation**: Express complex business rules naturally

## 2. Native Python API vs Configuration

### Design Decision: Code over Config

Unlike configuration-based systems that use YAML or JSON, DQX uses Python code as the primary interface for defining data quality checks. This fundamental design choice drives the entire architecture.

### Advantages

- **IDE support**: Full autocomplete, type checking, and inline documentation
- **Composable & reusable functions**: Build complex checks from simple components
- **Version control friendly**: Standard code review processes apply
- **Dynamic check generation**: Use loops, conditions, and functions to create checks programmatically

## 3. Modular & Extensible Architecture

### Plugin-Based Design

DQX uses a protocol-based architecture that allows users to extend the framework without modifying core code.

### Extension Points

- **Custom Metrics**: Implement the `MetricSpec` protocol to create new metrics
- **Custom Validators**: Implement validation functions following the standard signature
- **Custom Data Sources**: Implement the `SqlDataSource` protocol for new data backends
- **Custom SQL Dialects**: Add support for database-specific SQL syntax

## 4. Symbolic Metrics

### How Symbolic Metrics Work

Metrics in DQX are symbolic expressions powered by SymPy, Python's symbolic mathematics library.
The framework evaluates metrics by collecting symbol values through SQL queries at execution time.

### Key Advantages

- **Mathematical operations on metrics**: Combine metrics using any SymPy functions
- **Cross-datasource validation**: Combine metrics from different data sources in a single expression
- **Lazy evaluation**: SQL generation happens only when needed
- **Declarative expressions**: Write business rules as mathematical formulas

## 5. SQL Backend Architecture

### Why SQL over Spark

DQX uses SQL as its computation backend instead of distributed frameworks like Spark:

- **Faster execution for single-node operations**: Direct SQL execution without cluster overhead
- **No Spark cluster required**: Simpler infrastructure and lower operational costs
- **Lower operational complexity**: No JVM tuning or cluster management
- **Rich analytical functions**: Leverage database-native window functions and aggregations

### Performance Optimizations

- Single-pass computation for multiple metrics for multiple days.
- Efficient CTE-based query generation

## 6. Severity Levels

### P0-P3 Definitions

- **P0 (Crisis)**: Data corruption or complete failure requiring immediate intervention
- **P1 (Major)**: Significant issues affecting data reliability and business decisions
- **P2 (Degraded)**: Quality degradation that needs investigation but not immediate action
- **P3 (Minor Noise)**: Informational alerts for monitoring trends and anomalies

## 7. API Reference & Supported Operations

### Supported Dialects

| Dialect | Description |
|---------|-------------|
| DuckDB | Default dialect, in-memory analytical database |
| BigQuery | Google Cloud data warehouse |
| PyArrow | Arrow tables via DuckDB integration |

### Supported Metrics

| Metric | Method | Description |
|--------|--------|-------------|
| Row Count | `num_rows` | Total number of rows |
| First | `first` | First value in column |
| Average | `average` | Mean value of column |
| Sum | `sum` | Sum of column values |
| Minimum | `minimum` | Minimum value |
| Maximum | `maximum` | Maximum value |
| Variance | `variance` | Statistical variance |
| Null Count | `null_count` | Count of null values |
| Duplicate Count | `duplicate_count` | Count of duplicate rows |

### Assertion Methods

| Method | Description |
|--------|-------------|
| `is_eq` | Assert equals with tolerance |
| `is_neq` | Assert not equal to |
| `is_gt` | Assert greater than |
| `is_lt` | Assert less than |
| `is_geq` | Assert greater than or equal to |
| `is_leq` | Assert less than or equal to |
| `is_between` | Assert in range (inclusive) |
| `is_positive` | Assert value > 0 |
| `is_negative` | Assert value < 0 |
| `is_zero` | Assert value is effectively zero |
| `is_none` | Assert value is None |
| `is_not_none` | Assert value is not None |
| `within_tol` | Assert within relative or absolute tolerance |
