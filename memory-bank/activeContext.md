# Active Context

## Current Focus
The project has successfully removed all batch processing support to simplify the analyzer for the first release. The system now focuses on efficient single-pass processing using DuckDB's query engine, which provides sufficient performance for most use cases while maintaining a cleaner, more maintainable architecture.

## Recently Completed Work (October 15, 2025)

### Batch Support Removal
Completed comprehensive removal of batch processing infrastructure:
- Removed `BatchSqlDataSource` protocol from `common.py`
- Removed `ArrowBatchDataSource` class from `pyarrow_ds.py`
- Removed batch processing methods from analyzer (`_analyze_batches`, `_analyze_batches_threaded`)
- Simplified `analyze()` method to directly call `analyze_single()`
- Removed all threading infrastructure for batch processing
- Updated documentation to remove batch processing references
- All 571 tests passing after removal

### Key Decisions in Batch Removal
- **Retained `merge()` functionality**: Still useful for combining analysis reports from different time periods
- **Kept internal batching in `analyze_sketch_ops`**: The `batch_size` parameter is for memory-efficient Arrow processing, not related to removed batch feature
- **Simplified error messages**: Clear messages when unsupported data sources are used

## Previous Work (October 14, 2025)

### Graph Property Implementation
- Added `_graph_built` flag to track when graph has been constructed
- Graph property raises `DQXError` if accessed before `build_graph()` or `run()`
- Property provides read-only access to the internal `Graph` instance

### Renamed collect() to build_graph()
- Renamed for better clarity of purpose
- Method sets `_graph_built = True` after successful validation
- Updated all internal references and test files

### Removed validate() Method
- Validation now happens automatically during `build_graph()`
- Simplified workflow - impossible to skip validation
- Errors raise `DQXError`, warnings are logged but don't fail

## Current API Usage Patterns (from e2e test)

### Check Decorator Pattern
```python
@check(name="Simple Checks", datasets=["ds1"])
def simple_checks(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.null_count("delivered")).where(
        name="Delivered null count is less than 100"
    ).is_leq(100)
```

### Direct Suite Instantiation
```python
suite = VerificationSuite(checks, db, name="Simple test suite")
suite.run({"ds1": ds1, "ds2": ds2}, key)
suite.graph.print_tree()  # Safe access after run()
```

### Result Collection Methods
- `suite.collect_results()` - Collect assertion results
- `suite.collect_symbols()` - Collect symbol information
- Both methods work after graph is built

## Key Technical Decisions

### Simplified Architecture
- Single-pass processing model
- No threading complexity for batch operations
- DuckDB handles performance optimization internally
- Cleaner, more maintainable codebase

### Graph Access Pattern
- Graph is only accessible after it's been built
- Prevents access to incomplete or invalid graph states
- Clear error messages guide proper usage

### Validation Integration
- Validation is no longer a separate step
- Happens automatically during graph building
- Simplifies the API and ensures validation always occurs

### API Simplification History
- v0.5.0+: Batch support removal
- v0.5.0: Removed VerificationSuiteBuilder
- v0.4.0: Removed assertion chaining, made assertions require names
- Current: Defensive graph access, integrated validation, no batch processing

## Next Steps
- Monitor adoption of simplified analyzer
- Continue improving error messages and debugging tools
- Focus on single-source performance optimizations
- Explore streaming data support (planned for Q1 2025)

## Important Patterns and Preferences

### Assertion Naming
All assertions must have descriptive names using the two-stage pattern:
```python
ctx.assert_that(metric).where(name="Description").is_gt(0)
```

### Cross-Dataset Validation
Supported through dataset parameter:
```python
mp.average("tax", dataset="ds1")
mp.average("tax", dataset="ds2")
```

### Time-Series Comparisons
Using key.lag() for historical comparisons:
```python
mp.average("tax", key=ctx.key.lag(1))
```

### Extension Methods
Custom metrics available through mp.ext:
```python
mp.ext.day_over_day(specs.Average("tax"))
```

## Current State
All tests passing, documentation updated, and implementation complete. The DQX analyzer now has:
- Simplified single-pass processing
- No batch processing complexity
- Clear, maintainable architecture
- Defensive graph property access
- Renamed `build_graph()` method (from `collect()`)
- Integrated validation (no separate `validate()`)
- Direct instantiation (no builder pattern)
- Mandatory assertion names
- Focus on DuckDB's efficient query engine for performance
