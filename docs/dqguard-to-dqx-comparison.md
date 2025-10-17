# DQGuard to DQX: Evolution of Data Quality at Scale

## Executive Summary

DQX represents the next generation of data quality tooling, evolving from DQGuard's configuration-based approach to a code-first framework. While DQGuard pioneered automated quality monitoring through JSON configurations, DQX advances the field with mathematical expressions, type safety, and modern architecture.

This document guides teams through understanding the evolution, key improvements, and migration considerations.

## The Evolution Story

### Where We Started: DQGuard

DQGuard solved critical problems:
- High overhead in quality monitoring across teams
- Need for long-term reliability measurement
- Slow incident response times
- Unreliable business-critical data

The solution: JSON-based configurations that automated metric collection and validation.

### Where We're Going: DQX

DQX addresses modern challenges:
- Complex validation logic requiring mathematical expressions
- Efficient data processing through modern architecture
- Developer productivity through type safety and IDE support
- Flexible assertions beyond time-series patterns

## Key Improvements

### 1. From Configuration to Code

**DQGuard**: Define checks in JSON
```json
{
    "name": "default.reservation_flatter",
    "type": "table",
    "metrics": ["num_rows"],
    "validators": [{
        "name": "is_geq",
        "threshold": 1000000
    }]
}
```

**DQX**: Express checks as Python functions
```python
@check("Reservations have sufficient volume")
def validate_reservations(mp: MetricProvider, ctx: Context) -> None:
    ctx.assert_that(mp.num_rows()).where(name="Daily volume check").is_geq(1000000)
```

Benefits:
- Type checking catches errors before runtime
- IDE autocompletion speeds development
- Version control shows meaningful diffs
- Reusable logic through standard Python patterns

### 2. Mathematical Expressions

**DQGuard**: Fixed validator patterns
```json
{
    "validators": ["within_2_sd", "wow_change"]
}
```

**DQX**: Arbitrary mathematical assertions
```python
# Revenue integrity check
calculated = mp.sum("price") * mp.sum("quantity")
reported = mp.sum("revenue")
error_rate = sp.Abs(calculated - reported) / reported

ctx.assert_that(error_rate).where(name="Revenue calculation accuracy").is_lt(0.001)
```

### 3. Processing Architecture

**DQGuard**:
- Spark-based processing
- Separate metric collection and validation passes
- Scales with cluster resources

**DQX**:
- DuckDB columnar engine
- Single-pass optimization
- Efficient query execution
- Statistical sketching for memory efficiency

### 4. Developer Experience

**DQGuard**:
```bash
# Edit JSON
vim lib/quality_check.json

# Run in workflow
<action name="quality_check">
    <sub-workflow>
        <app-path>${workflowsBaseDir}/data-quality-library/production-app</app-path>
    </sub-workflow>
</action>
```

**DQX**:
```python
# Write with IDE support
def validate_orders(mp: MetricProvider, ctx: Context) -> None:
    # Autocompletion shows available metrics
    ctx.assert_that(mp.average("price")).where(name="Price validation").is_positive()


# Run directly
suite = VerificationSuite([validate_orders], db)
suite.run(datasources, key)
```

## Feature Comparison

| Feature | DQGuard | DQX |
|---------|---------|-----|
| **Configuration** | JSON files | Python code |
| **Type Safety** | Runtime validation | Compile-time checking |
| **Metrics** | 15 predefined | Extensible + custom |
| **Validators** | 12 patterns | Unlimited expressions |
| **Engine** | Spark clusters | DuckDB |
| **Processing Model** | Multiple passes | Single pass |
| **Architecture** | Distributed | Columnar + sketches |
| **IDE Support** | JSON schemas | Full Python tooling |
| **Testing** | Manual validation | Standard unit tests |
| **Debugging** | Log analysis | Interactive debugging |
| **Deployment** | Oozie workflows | Any Python environment |

## Migration Guide

### Assessment Phase

1. **Inventory Current Checks**
   - List all DQGuard configurations
   - Identify custom validators
   - Note integration points

2. **Complexity Analysis**
   - Simple threshold checks → Direct migration
   - Time-series validators → May need custom logic
   - Complex preprocessors → Evaluate alternatives

### Migration Patterns

#### Pattern 1: Simple Metrics
**DQGuard**:
```json
{
    "metrics": ["num_rows", "null_count"],
    "validators": [{"name": "is_geq", "threshold": 0}]
}
```

**DQX**:
```python
ctx.assert_that(mp.num_rows()).where(name="Has rows").is_positive()
ctx.assert_that(mp.null_count("id")).where(name="No null IDs").is_zero()
```

#### Pattern 2: Time-Series Validation
**DQGuard**:
```json
{
    "validators": ["within_2_sd"],
    "time_range": "7 days"
}
```

**DQX**:
```python
# Collect historical metrics
history = [mp.sum("revenue", key=ctx.key.lag(i)) for i in range(1, 8)]
mean = sum(history) / len(history)
std = calculate_std(history)

# Apply validation
current = mp.sum("revenue")
z_score = abs(current - mean) / std
ctx.assert_that(z_score).where(name="Within 2 SD").is_lt(2)
```

#### Pattern 3: Duplicate Detection
**DQGuard**:
```json
{
    "metrics": [{
        "name": "has_duplicate",
        "columns": ["transaction_id"]
    }]
}
```

**DQX**:
```python
# Built-in duplicate detection (coming in v0.6)
# Current approach:
duplicate_count = mp.count("transaction_id") - mp.approx_cardinality("transaction_id")
ctx.assert_that(duplicate_count).where(name="No duplicates").is_zero()
```

### Coexistence Strategy

Teams can run both systems during migration:

1. **Phase 1**: Shadow Mode
   - Keep DQGuard running
   - Add DQX checks in parallel
   - Compare results

2. **Phase 2**: Gradual Cutover
   - Migrate one dataset at a time
   - Maintain critical DQGuard checks
   - Build confidence in DQX

3. **Phase 3**: Full Migration
   - Retire DQGuard configurations
   - Leverage DQX-only features
   - Optimize for performance

## Advanced DQX Capabilities

### 1. Cross-Dataset Validation
```python
@check("Production matches staging")
def compare_environments(mp: MetricProvider, ctx: Context) -> None:
    prod = mp.sum("revenue", dataset="production")
    staging = mp.sum("revenue", dataset="staging")

    ctx.assert_that(prod).where(name="Environment parity").is_eq(staging, tol=0.01)
```

### 2. Custom Metric Extensions
```python
# DQX supports custom metrics through extensions
day_over_day = mp.ext.day_over_day(specs.Average("response_time"))
ctx.assert_that(day_over_day).where(name="Response time trend").is_between(-0.1, 0.1)
```

### 3. Symbolic Mathematics
```python
# Complex business rules as mathematical expressions
margin = (mp.sum("revenue") - mp.sum("cost")) / mp.sum("revenue")
target_margin = 0.3

ctx.assert_that(margin).where(name="Profit margin target", severity="P0").is_geq(
    target_margin
)
```


## Best Practices

### 1. Name Every Assertion
```python
# Good: Clear, specific names
ctx.assert_that(metric).where(name="Daily revenue within 10% of average")

# Avoid: Generic names
ctx.assert_that(metric).where(name="Revenue check")
```

### 2. Group Related Checks
```python
@check("Payment integrity", datasets=["payments"])
def validate_payments(mp: MetricProvider, ctx: Context) -> None:
    # All payment validations in one check
    ctx.assert_that(mp.null_count("payment_id")).where(
        name="Payment ID completeness"
    ).is_zero()

    ctx.assert_that(mp.average("amount")).where(
        name="Average payment reasonable"
    ).is_between(10, 1000)
```

### 3. Use Severity Levels
```python
ctx.assert_that(critical_metric).where(
    name="Critical business metric", severity="P0"  # Pages on-call
).is_positive()

ctx.assert_that(quality_metric).where(
    name="Data quality indicator", severity="P2"  # Daily review
).is_within_range()
```

## Conclusion

DQX builds upon DQGuard's foundation while addressing modern data quality challenges. The evolution from configuration to code enables:

- **Better expressiveness** through mathematical assertions
- **Modern architecture** with DuckDB's columnar processing
- **Enhanced productivity** with type safety and IDE support
- **Greater flexibility** for complex business rules

Teams should evaluate their current DQGuard usage and plan migration based on complexity and criticality. The coexistence strategy allows gradual adoption while maintaining quality coverage.

For teams starting fresh, DQX provides a modern, flexible solution for data quality validation.
