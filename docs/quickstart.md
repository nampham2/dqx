# Quick Start

Get started with DQX in 5 minutes! This guide shows you how to create your first data quality checks.

## Basic Example

Here's a simple example checking data quality on a pandas DataFrame:

```python
import pandas as pd
from dqx import DataQualityValidator

# Sample data
df = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, -5, 150, 35],
        "email": [
            "john@example.com",
            "invalid-email",
            "jane@test.com",
            None,
            "bob@demo.com",
        ],
        "score": [85.5, 92.0, 78.5, None, 88.0],
    }
)

# Create validator
validator = DataQualityValidator()

# Define checks
checks = (
    validator.create_checks(df)
    .is_not_null("user_id")
    .is_between("age", 0, 120)
    .matches_pattern("email", r"^[\w\.-]+@[\w\.-]+\.\w+$")
    .is_not_null("score")
)

# Run validation
results = validator.validate(df, checks)

# Display results
print(results.summary())
```

## Understanding Results

DQX provides detailed validation results:

```python
# Check if validation passed
if results.passed:
    print("✅ All checks passed!")
else:
    print(f"❌ {results.failed_count} checks failed")

# Get detailed results
for check_result in results.details:
    print(f"{check_result.check_name}: {check_result.status}")
    if not check_result.passed:
        print(f"  Failed rows: {check_result.failed_rows}")
```

## Common Data Quality Checks

### 1. Null Value Checks

```python
# Check for nulls
validator.create_checks(df).is_not_null(
    "column_name"
).has_no_nulls()  # Check all columns
```

### 2. Range Validation

```python
# Numeric ranges
validator.create_checks(df).is_between("age", 0, 120).is_positive(
    "amount"
).is_greater_than("score", 0)
```

### 3. Pattern Matching

```python
# String patterns
validator.create_checks(df).matches_pattern(
    "email", r"^[\w\.-]+@[\w\.-]+\.\w+$"
).matches_pattern("phone", r"^\d{3}-\d{3}-\d{4}$").has_length("zip_code", 5)
```

### 4. Uniqueness Checks

```python
# Unique values
validator.create_checks(df).is_unique("user_id").has_no_duplicates(
    ["first_name", "last_name"]
)
```

### 5. Statistical Checks

```python
# Statistical validation
validator.create_checks(df).mean_between("score", 70, 90).std_dev_less_than(
    "price", 100
).percentile_between("income", 0.25, 10000, 50000)
```

## Working with Different Data Sources

### SQL Databases

```python
from dqx import SQLDataSource

# Connect to database
datasource = SQLDataSource(connection_string="postgresql://...")

# Validate query results
query = "SELECT * FROM users WHERE created_at > '2024-01-01'"
results = validator.validate_query(datasource, query, checks)
```

### CSV Files

```python
# Validate CSV directly
results = validator.validate_file("data.csv", checks)

# Or load and validate
df = pd.read_csv("data.csv")
results = validator.validate(df, checks)
```

### Parquet Files

```python
# Validate Parquet files
results = validator.validate_file("data.parquet", checks)
```

## Custom Validation Rules

Create custom validation logic:

```python
from dqx import custom_check


@custom_check
def business_rule_check(df):
    """Custom business rule validation"""
    mask = (df["status"] == "active") & (df["balance"] > 0)
    return mask


# Use custom check
validator.create_checks(df).add_custom_check(
    business_rule_check, "active_positive_balance"
)
```

## Validation Reporting

### Summary Report

```python
# Get summary statistics
summary = results.summary()
print(f"Total checks: {summary['total']}")
print(f"Passed: {summary['passed']}")
print(f"Failed: {summary['failed']}")
print(f"Pass rate: {summary['pass_rate']:.1%}")
```

### Detailed Report

```python
# Generate detailed HTML report
results.to_html("validation_report.html")

# Or get DataFrame of results
results_df = results.to_dataframe()
```

### Export Results

```python
# Export to various formats
results.to_json("results.json")
results.to_csv("results.csv")
```

## Error Handling

DQX provides clear error messages:

```python
try:
    results = validator.validate(df, checks)
except ValidationError as e:
    print(f"Validation error: {e}")
except DataSourceError as e:
    print(f"Data source error: {e}")
```

## Best Practices

1. **Start Simple**: Begin with basic null and range checks
2. **Incremental Validation**: Add checks gradually
3. **Use Descriptive Names**: Name your checks clearly
4. **Set Appropriate Thresholds**: Be realistic with ranges
5. **Monitor Trends**: Track validation results over time

## Next Steps

- Explore the [User Guide](user-guide.md) for advanced features
- Learn about [Plugin System](plugin_system.md) for extending DQX
- Check [API Reference](api-reference.md) for detailed documentation
- See [Examples](https://github.com/yourusername/dqx/tree/main/examples) for real-world use cases

## Getting Help

- 📖 [Full Documentation](https://dqx.readthedocs.io)
- 💬 [GitHub Discussions](https://github.com/yourusername/dqx/discussions)
- 🐛 [Report Issues](https://github.com/yourusername/dqx/issues)
- 📧 [Contact Support](mailto:support@dqx.dev)

---

Ready to dive deeper? Check out the [User Guide](user-guide.md) for comprehensive documentation.
