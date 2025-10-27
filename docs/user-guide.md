# User Guide

Welcome to the DQX User Guide! This comprehensive guide covers all aspects of using DQX for data quality validation.

## Overview

DQX (Data Quality Excellence) transforms data validation into mathematical expressions, making it easy to find data quality issues before they impact your business.

## Core Concepts

### 1. Validators

Validators are the heart of DQX. They orchestrate the validation process:

```python
from dqx import DataQualityValidator

validator = DataQualityValidator()
```

### 2. Checks

Checks define what to validate. DQX provides many built-in checks:

- **Completeness**: Null checks, missing value detection
- **Consistency**: Pattern matching, format validation
- **Accuracy**: Range checks, statistical validation
- **Uniqueness**: Duplicate detection, key validation

### 3. Data Sources

DQX supports multiple data sources:

- Pandas DataFrames
- SQL databases (PostgreSQL, MySQL, SQLite)
- CSV, Parquet, JSON files
- Cloud storage (S3, GCS, Azure)

### 4. Results

Validation results provide detailed insights:

- Pass/fail status
- Row-level details
- Statistical summaries
- Trend analysis

## Validation Workflow

```mermaid
graph LR
    A[Data Source] --> B[Define Checks]
    B --> C[Run Validation]
    C --> D[Analyze Results]
    D --> E[Take Action]
```

## Detailed Features

### Completeness Validation

Check for missing or null values:

```python
# Single column
checks.is_not_null("customer_id")

# Multiple columns
checks.are_not_null(["name", "email", "phone"])

# Completeness threshold
checks.has_completeness("optional_field", threshold=0.95)
```

### Consistency Validation

Ensure data follows expected patterns:

```python
# Email format
checks.matches_pattern("email", r"^[\w\.-]+@[\w\.-]+\.\w+$")

# Date format
checks.matches_date_format("birth_date", "%Y-%m-%d")

# Custom formats
checks.matches_format("product_code", "XXX-####")
```

### Accuracy Validation

Validate data accuracy and ranges:

```python
# Numeric ranges
checks.is_between("age", 0, 120)
checks.is_positive("amount")
checks.is_not_negative("balance")

# Date ranges
checks.date_between("order_date", "2020-01-01", "2024-12-31")

# Statistical checks
checks.mean_between("score", 70, 90)
checks.std_dev_less_than("variance", 10)
```

### Uniqueness Validation

Detect duplicates and ensure uniqueness:

```python
# Single column uniqueness
checks.is_unique("user_id")

# Composite key uniqueness
checks.has_unique_combination(["first_name", "last_name", "birth_date"])

# Duplicate detection
checks.has_no_duplicates()
checks.duplicate_count_less_than("email", 5)
```

### Cross-Column Validation

Validate relationships between columns:

```python
# Column comparison
checks.column_greater_than("end_date", "start_date")
checks.columns_match("billing_address", "shipping_address")

# Conditional validation
checks.when('status == "active"').then("balance > 0")

# Complex rules
checks.satisfies(lambda df: df["price"] * df["quantity"] == df["total"])
```

## Advanced Features

### Custom Validation Functions

Create domain-specific validations:

```python
@validator.custom_check
def validate_business_rule(df):
    """Custom business logic"""
    valid_mask = (
        (df["customer_type"] == "premium") & (df["credit_limit"] >= 10000)
    ) | ((df["customer_type"] == "standard") & (df["credit_limit"] <= 5000))
    return valid_mask
```

### Validation Pipelines

Chain multiple validations:

```python
pipeline = validator.create_pipeline()

# Stage 1: Basic checks
pipeline.add_stage("basic", [checks.is_not_null("id"), checks.is_unique("id")])

# Stage 2: Business rules
pipeline.add_stage(
    "business",
    [checks.is_positive("revenue"), checks.date_not_future("transaction_date")],
)

# Run pipeline
results = pipeline.run(df)
```

### Conditional Validation

Apply checks based on conditions:

```python
# Different rules for different segments
checks.when('region == "US"').then(
    checks.matches_pattern("phone", r"^\d{3}-\d{3}-\d{4}$")
)

checks.when('region == "UK"').then(checks.matches_pattern("phone", r"^\+44\d{10}$"))
```

### Sampling and Performance

Optimize validation for large datasets:

```python
# Sample validation
validator.validate_sample(df, sample_size=10000, checks=checks)

# Chunk processing
validator.validate_chunks(df, chunk_size=50000, checks=checks)

# Parallel validation
validator.validate_parallel(df, checks=checks, n_jobs=4)
```

## Reporting and Monitoring

### HTML Reports

Generate interactive reports:

```python
results.generate_report(
    "validation_report.html", include_charts=True, include_failed_rows=True
)
```

### Dashboard Integration

Export metrics for monitoring:

```python
# Prometheus metrics
metrics = results.to_prometheus()

# JSON for APIs
json_results = results.to_json()

# Time series data
time_series = results.to_time_series()
```

### Alerting

Set up alerts for validation failures:

```python
# Email alerts
if not results.passed:
    validator.send_alert(
        to=["data-team@company.com"], subject="Data Quality Alert", results=results
    )

# Slack integration
validator.notify_slack(webhook_url="https://hooks.slack.com/...", results=results)
```

## Best Practices

### 1. Start with Critical Checks

Begin with the most important validations:
- Primary key uniqueness
- Not null for required fields
- Basic range validations

### 2. Use Descriptive Names

```python
# Good
checks.is_not_null("customer_id").with_name("Customer ID Required")

# Better
checks.is_not_null("customer_id").with_name(
    "Customer ID Required", description="Every order must have a valid customer ID"
)
```

### 3. Set Appropriate Thresholds

```python
# Allow some nulls in optional fields
checks.has_completeness("middle_name", threshold=0.8)

# Strict for critical fields
checks.has_completeness("email", threshold=1.0)
```

### 4. Group Related Checks

```python
customer_checks = CheckGroup("Customer Validation")
customer_checks.add(
    [
        checks.is_not_null("customer_id"),
        checks.matches_pattern("email", email_regex),
        checks.is_between("age", 18, 120),
    ]
)
```

### 5. Version Your Validations

```python
# Track validation rules in version control
validator.save_rules("validations/v1.0.0/customer_rules.json")

# Load versioned rules
validator.load_rules("validations/v1.0.0/customer_rules.json")
```

## Integration Examples

### With Pandas

```python
# Direct DataFrame validation
df = pd.read_csv("data.csv")
results = validator.validate(df, checks)
```

### With SQL Databases

```python
# PostgreSQL
conn_string = "postgresql://user:pass@localhost/db"
results = validator.validate_query(conn_string, "SELECT * FROM customers", checks)
```

### With Apache Spark

```python
# Spark DataFrame validation
spark_df = spark.read.parquet("data.parquet")
results = validator.validate_spark(spark_df, checks)
```

### With Airflow

```python
# DQX Airflow operator
from dqx.airflow import DQXValidationOperator

validation_task = DQXValidationOperator(
    task_id="validate_customer_data",
    source="s3://bucket/data.parquet",
    checks=customer_checks,
    fail_on_error=True,
)
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**
   - Use sampling or chunk processing
   - Increase available memory
   - Use distributed processing

2. **Slow validation performance**
   - Create indexes on validated columns
   - Use parallel processing
   - Optimize regex patterns

3. **Connection issues**
   - Check database credentials
   - Verify network connectivity
   - Check firewall rules

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
validator = DataQualityValidator(debug=True)
```

## Next Steps

- Explore the [Plugin System](plugin_system.md) to extend DQX
- Read the [API Reference](api-reference.md) for detailed documentation
- Check out [real-world examples](https://github.com/yourusername/dqx/tree/main/examples)
- Join the [community](https://github.com/yourusername/dqx/discussions)

---

Need help? Check our [FAQ](https://dqx.readthedocs.io/faq) or [open an issue](https://github.com/yourusername/dqx/issues).
