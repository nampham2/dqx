# DQX Active Context

## Current Focus
As of October 2025, DQX has reached version 0.3.0 with significant architectural enhancements. The focus has been on adding extensibility through plugins, improving validation capabilities, and enhancing the developer experience.

## Recent Changes

### Plugin System Implementation (October 2025)
- **PluginManager**: Centralized plugin lifecycle management
- **PostProcessor Protocol**: Clean interface for result processing plugins
- **Built-in AuditPlugin**: Rich-formatted execution reports with statistics
- **Time-limited Execution**: 60-second timeout for plugin safety
- **Plugin Discovery**: Automatic registration of built-in plugins

### Enhanced Validation Framework (October 2025)
- **Comprehensive Validators**: Four specialized validators for different issues
  - `DuplicateCheckNameValidator`: Detects duplicate check names (error)
  - `EmptyCheckValidator`: Warns about checks with no assertions (warning)
  - `DuplicateAssertionNameValidator`: Catches duplicate assertion names within checks (error)
  - `DatasetValidator`: Detects dataset mismatches and ambiguities (error)
- **CompositeValidationVisitor**: Efficient single-pass validation
- **Structured Reports**: ValidationReport with errors, warnings, and structured output

### Evaluation Failure Improvements (October 2025)
- **EvaluationFailure Dataclass**: Rich error context with expression and symbol info
- **SymbolInfo Dataclass**: Complete metadata for each symbol including dataset and suite
- **Complex Number Handling**: Proper detection and error reporting for complex/infinite values
- **Enhanced Error Messages**: Include symbol values and expressions in failures

### Symbol Natural Ordering (October 2025)
- Symbols now sort numerically: x_1, x_2, ..., x_10 (not x_1, x_10, x_2)
- Consistent ordering in reports and displays

### SQL Formatting (October 2025)
- Integration with `sqlparse` library
- Formatted SQL output for better readability
- Automatic formatting in query generation

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

- **Mandatory Severity**: All assertions must have a severity level
  - No longer accepts None
  - Defaults to "P1" if not specified
  - Breaking change from previous versions

- **Mandatory Check Names**: @check decorator requires name parameter
  - `@check(name="My Check")` is required
  - No default or automatic naming
  - Improves clarity and debugging

### Graph Architecture
- **Visitor pattern** for extensibility
- **Node hierarchy**: Root → Check → Assertion → Symbol
- **Dataset imputation**: Automatic inference of dataset associations
- **Defensive graph property**: Explicit `_graph_built` flag for safety

### Error Handling
- **Returns library** for functional error handling
- **Result types** instead of exceptions for expected failures
- **Detailed failure context** in assertion results
- **EvaluationFailure** for rich error information

## Next Steps

### Immediate Tasks
1. **Documentation Enhancement**
   - More real-world examples
   - Tutorial series
   - API reference completion
   - Plugin development guide

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

### Plugin Development Pattern
```python
class MyPlugin:
    @staticmethod
    def metadata() -> PluginMetadata:
        return PluginMetadata(
            name="my-plugin",
            version="1.0.0",
            author="Your Name",
            description="Description",
            capabilities={"reporting"},
        )

    def process(self, context: PluginExecutionContext) -> None:
        # Access results and symbols
        failed = context.failed_assertions()
        pass_rate = context.assertion_pass_rate()
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
    if failure.metric.is_failure():
        errors = failure.metric.failure()
        for error in errors:
            print(f"Expression: {error.expression}")
            for symbol in error.symbols:
                print(f"  {symbol.name}: {symbol.value}")
```

## Development Workflow

### Adding New Features
1. Write comprehensive tests first (TDD)
2. Implement minimal working solution
3. Run full test suite
4. Check type hints with mypy
5. Run pre-commit hooks (now includes yamllint, shfmt, shellcheck)
6. Update documentation
7. Add examples if appropriate

### Code Review Checklist
- [ ] Tests added/updated
- [ ] Type hints complete
- [ ] Docstrings updated
- [ ] No backward compatibility breaks
- [ ] Examples work correctly
- [ ] Documentation updated
- [ ] Plugin compatibility verified

## Known Issues

### Current Limitations
1. **No partial evaluation**: All metrics in suite must succeed
2. **Limited error recovery**: Single failure stops evaluation
3. **Memory metrics**: Large result sets need pagination

### Workarounds
1. **Partial evaluation**: Split into multiple suites
2. **Error recovery**: Use plugins for custom error handling
3. **Large results**: Use database-side aggregation

## Key Insights

### What Works Well
- Symbolic expression API is intuitive
- Graph-based execution is efficient
- Protocol-based extensions are clean
- Error messages are helpful
- Plugin system provides flexibility
- Validation framework catches issues early

### What Needs Improvement
- Better debugging tools for complex expressions
- More built-in metrics (median, mode, etc.)
- Performance profiling capabilities
- Visualization of dependency graphs
- Plugin marketplace/registry

## Contact & Collaboration
- Lead Developer: Nam Pham (phamducnam@gmail.com)
- Development tracked in GitLab
- Following TDD and clean code principles
- Open to contributions after public release
