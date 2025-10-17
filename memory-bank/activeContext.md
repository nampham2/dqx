# DQX Active Context

## Current Focus
As of October 2025, DQX is at version 0.2.0 preparing for its first public release. The focus has been on simplifying the architecture and ensuring a solid foundation.

## Recent Changes

### Removed Features (Simplification)
1. **Batch Processing Support** (October 2025)
   - Removed `BatchSqlDataSource` protocol
   - Removed `ArrowBatchDataSource` implementation
   - Removed threading infrastructure
   - Simplified to single-pass execution only
   - Rationale: Reduce complexity for initial release

### Added Features
1. **is_between Assertion** (October 2025)
   - Added `is_between(lower, upper, tol)` function
   - Inclusive bounds with tolerance
   - Validates range before creation
   - Can check exact values with equal bounds

2. **Enhanced Display Functions** (2024)
   - `print_assertion_results()` for formatted output
   - `print_symbols()` for symbol value display
   - Rich terminal formatting support

## Active Decisions

### API Design Choices
- **Two-stage assertion building**: `assert_that()` → `where()` → assertion
  - Enforces meaningful names for all assertions
  - Prevents unnamed assertions in reports
  - Better error messages and debugging

### Graph Architecture
- **Visitor pattern** for extensibility
- **Node hierarchy**: Root → Check → Assertion → Symbol
- **Dataset imputation**: Automatic inference of dataset associations

### Error Handling
- **Returns library** for functional error handling
- **Result types** instead of exceptions for expected failures
- **Detailed failure context** in assertion results

## Next Steps

### Immediate Tasks
1. **Documentation Enhancement**
   - More real-world examples
   - Tutorial series
   - API reference completion

2. **Performance Optimization**
   - Query plan analysis
   - Caching strategy for repeated metrics
   - Connection pooling for databases

3. **Dialect Support**
   - PostgreSQL dialect implementation
   - Snowflake dialect consideration
   - MySQL/MariaDB evaluation

### Future Considerations
1. **Streaming Support**
   - Real-time validation capabilities
   - Incremental metric updates
   - Event-driven checks

2. **Advanced Metrics**
   - Statistical metrics (percentiles, correlation)
   - Machine learning metrics (drift detection)
   - Custom aggregation functions

3. **Integration Features**
   - dbt integration package
   - Airflow operators
   - Prefect tasks

## Important Patterns

### Check Writing Best Practices
```python
@check(name="Meaningful name", severity="P1")
def validate_data(mp: MetricProvider, ctx: Context) -> None:
    # Group related assertions
    price = mp.average("price")

    # Use descriptive names
    ctx.assert_that(price).where(name="Average price is reasonable").is_between(
        10, 1000
    )
```

### Metric Reuse Pattern
```python
# Define once, use multiple times
total = mp.sum("amount")
yesterday_total = mp.sum("amount", key=ctx.key.lag(1))

# Use in multiple assertions
ctx.assert_that(total).where(name="Has sales").is_positive()
ctx.assert_that((total - yesterday_total) / yesterday_total).where(
    name="Growth rate reasonable"
).is_between(-0.5, 0.5)
```

### Error Investigation Pattern
```python
# Collect all results for analysis
results = suite.collect_results()
failures = [r for r in results if r.status == "FAILURE"]

# Examine failure details
for failure in failures:
    error = failure.metric.failure()
    print(f"{failure.check}/{failure.assertion}: {error}")
```

## Development Workflow

### Adding New Features
1. Write comprehensive tests first (TDD)
2. Implement minimal working solution
3. Run full test suite
4. Check type hints with mypy
5. Run pre-commit hooks
6. Update documentation
7. Add examples if appropriate

### Code Review Checklist
- [ ] Tests added/updated
- [ ] Type hints complete
- [ ] Docstrings updated
- [ ] No backward compatibility breaks
- [ ] Examples work correctly
- [ ] Documentation updated

## Known Issues

### Current Limitations
1. **No partial evaluation**: All metrics in suite must succeed
2. **Limited error recovery**: Single failure stops evaluation
3. **Memory metrics**: Large result sets need pagination

### Workarounds
1. **Partial evaluation**: Split into multiple suites
2. **Error recovery**: Wrap metrics in try/except patterns
3. **Large results**: Use database-side aggregation

## Key Insights

### What Works Well
- Symbolic expression API is intuitive
- Graph-based execution is efficient
- Protocol-based extensions are clean
- Error messages are helpful

### What Needs Improvement
- Better debugging tools for complex expressions
- More built-in metrics (median, mode, etc.)
- Performance profiling capabilities
- Visualization of dependency graphs

## Contact & Collaboration
- Lead Developer: Nam Pham (phamducnam@gmail.com)
- Development tracked in GitLab
- Following TDD and clean code principles
- Open to contributions after public release
