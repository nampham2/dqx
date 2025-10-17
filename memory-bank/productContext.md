# DQX Product Context

## Why DQX Exists

### The Problem
Data quality issues cost organizations billions annually through:
- Bad business decisions based on incorrect data
- Failed ML models due to data drift
- Broken dashboards and reports
- Time wasted debugging data issues
- Loss of stakeholder trust in data

Traditional approaches fall short:
- **SQL scripts**: Ad-hoc, hard to maintain, no reusability
- **Config-based tools**: Limited expressiveness, poor IDE support
- **Spark-based solutions**: Operational complexity, slow for small datasets
- **Manual testing**: Not scalable, prone to human error

### The Solution
DQX provides a developer-friendly framework for data quality that:
- Uses native Python code for full expressiveness
- Leverages SQL for efficient computation
- Enables mathematical expressions for complex rules
- Requires no infrastructure beyond a database
- Integrates with existing workflows

## User Experience Goals

### 1. Intuitive API
```python
@check("Orders have valid prices")
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.average("price")).where(name="Price check").is_positive()
```
- Reads like natural language
- Full IDE support with autocomplete
- Type-safe with immediate feedback

### 2. Instant Feedback
- Run validation → See results immediately
- Clear pass/fail status with severity levels
- Actionable error messages
- Visual progress tracking

### 3. Powerful Expressions
```python
# Compare across time
today = mp.sum("revenue")
yesterday = mp.sum("revenue", key=ctx.key.lag(1))
change = (today - yesterday) / yesterday

# Complex business rules
error_rate = sp.Abs(calculated - reported) / reported
ctx.assert_that(error_rate).where(name="Accuracy").is_lt(0.001)
```

### 4. Zero Friction Integration
- Install with pip
- No clusters to manage
- Works with existing databases
- Minimal configuration

## Core User Workflows

### Daily Validation
1. Define checks for critical metrics
2. Schedule suite execution
3. Get alerts on failures
4. Investigate issues with detailed context
5. Track quality trends over time

### Development Testing
1. Write new transformation code
2. Add quality checks alongside
3. Run validation in CI/CD
4. Catch issues before production

### Data Monitoring
1. Set up comprehensive checks
2. Persist metrics to database
3. Build quality dashboards
4. Alert on anomalies

## Product Principles

### 1. Developer First
- Code is the interface
- Excellent error messages
- Comprehensive documentation
- Testable by design

### 2. Progressive Disclosure
- Simple tasks are simple
- Complex tasks are possible
- Advanced features don't clutter basics

### 3. Fast Feedback Loop
- Quick to write checks
- Fast to run validation
- Immediate actionable results

### 4. Composable Building Blocks
- Small, focused functions
- Combine for complex logic
- Reuse across projects

## Success Metrics

### User Success
- Time to first check: < 5 minutes
- Time to production: < 1 day
- Check reusability: > 80%
- False positive rate: < 5%

### Technical Success
- Query performance: < 30s for typical datasets
- Memory usage: Constant regardless of data size
- Test coverage: 100%
- Zero runtime dependencies issues

## Comparison to Alternatives

### vs Great Expectations
- **DQX**: Code-first, symbolic math, SQL-native
- **GE**: Config-heavy, limited expressions, Python execution

### vs dbt tests
- **DQX**: Full programming power, cross-dataset, time-series
- **dbt**: SQL-only, single query scope, limited assertions

### vs Apache Griffin
- **DQX**: Lightweight, no infrastructure, instant start
- **Griffin**: Complex setup, requires Spark cluster

## Future Vision
DQX becomes the standard way to express and validate data quality requirements, making "bad data in production" as rare as syntax errors in compiled code. Every data pipeline includes DQX checks, every notebook validates its inputs, and data quality becomes a solved problem.
