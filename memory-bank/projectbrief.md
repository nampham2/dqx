# DQX Project Brief

## Project Identity
**Name**: DQX (Data Quality eXcellence)
**Type**: High-performance data quality framework
**Repository**: git@gitlab.com:booking-com/personal/nam.pham/dqx.git
**License**: MIT

## Core Purpose
DQX is a production-ready data quality framework designed to provide fast, scalable data validation and monitoring capabilities for large datasets. It combines the analytical power of DuckDB with the efficiency of PyArrow to deliver sub-second query performance on large-scale data.

## Key Requirements

### Functional Requirements
1. **Data Validation**: Declarative API for defining data quality checks
2. **Metric Computation**: Support for statistical metrics (avg, sum, min, max, variance, cardinality)
3. **Assertion Framework**: Flexible assertion system with mathematical expressions
4. **Efficient Processing**: Handle large datasets through optimized single-pass processing
5. **Cross-Dataset Validation**: Compare metrics across multiple data sources
6. **Time-Series Support**: Track metrics over time with historical comparisons
7. **Persistence**: Store computed metrics for historical analysis

### Non-Functional Requirements
1. **Performance**: Sub-second query performance on large datasets
2. **Scalability**: Handle large-scale data through statistical sketching
3. **Memory Efficiency**: Use HyperLogLog and DataSketches for approximate computations
4. **Extensibility**: Plugin architecture for custom metrics and data sources
5. **Production Ready**: Comprehensive error handling and monitoring
6. **Developer Experience**: Intuitive, fluent API design

## Target Users
- Data Engineers building data quality pipelines
- Analytics teams needing data validation
- Platform teams requiring scalable data monitoring
- Organizations with large-scale data processing needs

## Success Criteria
1. Process billion-row datasets in seconds
2. Memory usage <1% of dataset size for cardinality estimation
3. 100% test coverage for core modules
4. Clear, actionable error messages
5. Seamless integration with existing data pipelines

## Constraints
- Python 3.11 or 3.12 required
- Depends on DuckDB for SQL execution
- Uses SymPy for symbolic mathematics
- Requires PyArrow for columnar data processing

## Project Scope
### In Scope
- Data quality validation framework
- Metric computation and storage
- Efficient single-pass data processing
- SQL-based computation engine
- Graph-based dependency management
- Statistical approximation algorithms

### Out of Scope
- Data transformation/ETL capabilities
- Data catalog management
- Workflow orchestration
- Real-time alerting system (planned for future)
- Web UI (planned for future)

## Key Deliverables
1. Core validation framework with graph-based architecture
2. Comprehensive metric library
3. Multiple data source adapters (PyArrow, DuckDB)
4. Persistence layer for historical metrics
5. Documentation and examples
6. Pre-configured development environment
