# API Reference

Complete API documentation for DQX (Data Quality Excellence).

## Core API

### DataQualityValidator

The main entry point for data validation.

```python
from dqx import DataQualityValidator

validator = DataQualityValidator(
    name="MyValidator",
    description="Validates customer data",
    fail_fast=False,
    parallel=True,
)
```

#### Parameters

- `name` (str, optional): Name of the validator instance
- `description` (str, optional): Description of the validator's purpose
- `fail_fast` (bool, default=False): Stop validation on first failure
- `parallel` (bool, default=True): Enable parallel processing

#### Methods

##### validate()

Run validation checks on data.

```python
results = validator.validate(data, checks, sample_size=None, context=None)
```

**Parameters:**
- `data`: DataFrame, file path, or data source to validate
- `checks`: List of Check objects or CheckGroup
- `sample_size` (int, optional): Number of rows to sample
- `context` (dict, optional): Additional context for validation

**Returns:** `ValidationResults` object

##### create_checks()

Create a check builder for fluent API.

```python
checks = (
    validator.create_checks(data).is_not_null("column1").is_unique("column2").build()
)
```

### Check Classes

#### BaseCheck

Abstract base class for all checks.

```python
class BaseCheck:
    def __init__(self, name, description=None, severity="error"):
        self.name = name
        self.description = description
        self.severity = severity
```

#### Built-in Checks

##### NotNullCheck

```python
from dqx.checks import NotNullCheck

check = NotNullCheck(column="user_id", name="User ID Required", severity="critical")
```

##### RangeCheck

```python
from dqx.checks import RangeCheck

check = RangeCheck(
    column="age", min_value=0, max_value=120, inclusive=True, name="Valid Age Range"
)
```

##### PatternCheck

```python
from dqx.checks import PatternCheck

check = PatternCheck(
    column="email", pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", name="Valid Email Format"
)
```

##### UniqueCheck

```python
from dqx.checks import UniqueCheck

check = UniqueCheck(
    columns=["user_id"], name="Unique User ID"  # Can be single column or list
)
```

### CheckBuilder API

Fluent interface for building checks.

```python
checks = (
    CheckBuilder(df)
    # Null checks
    .is_not_null("column")
    .are_not_null(["col1", "col2"])
    .has_no_nulls()
    # Range checks
    .is_between("age", 0, 120)
    .is_positive("amount")
    .is_negative("loss")
    .is_greater_than("score", 50)
    .is_less_than("cost", 1000)
    # Pattern checks
    .matches_pattern("email", r"^[\w\.-]+@[\w\.-]+\.\w+$")
    .matches_date_format("date", "%Y-%m-%d")
    .has_length("zip_code", 5)
    .starts_with("product_code", "PRD")
    .ends_with("filename", ".csv")
    # Uniqueness checks
    .is_unique("id")
    .has_unique_combination(["first_name", "last_name"])
    .has_no_duplicates()
    # Statistical checks
    .mean_between("score", 70, 90)
    .std_dev_less_than("variance", 10)
    .percentile_between("income", 0.25, 10000, 50000)
    # Build final check list
    .build()
)
```

### ValidationResults

Results container with analysis methods.

```python
class ValidationResults:
    @property
    def passed(self) -> bool:
        """Whether all checks passed"""

    @property
    def failed_count(self) -> int:
        """Number of failed checks"""

    @property
    def pass_rate(self) -> float:
        """Percentage of passed checks"""

    def summary(self) -> dict:
        """Get summary statistics"""

    def failed_checks(self) -> List[CheckResult]:
        """Get only failed check results"""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""

    def to_json(self, filepath: str = None) -> str:
        """Export results as JSON"""

    def to_html(self, filepath: str) -> None:
        """Generate HTML report"""
```

### Data Sources

#### PandasDataSource

```python
from dqx.sources import PandasDataSource

source = PandasDataSource(df)
```

#### SQLDataSource

```python
from dqx.sources import SQLDataSource

source = SQLDataSource(
    connection_string="postgresql://user:pass@host/db", query="SELECT * FROM table"
)
```

#### FileDataSource

```python
from dqx.sources import FileDataSource

source = FileDataSource(filepath="data.csv", format="csv", **read_options)
```

#### CloudDataSource

```python
from dqx.sources import CloudDataSource

source = CloudDataSource(
    uri="s3://bucket/path/data.parquet", credentials=aws_credentials
)
```

## Advanced Features

### Custom Checks

Create custom validation logic.

```python
from dqx import custom_check


@custom_check
def business_rule_check(df):
    """Custom validation function"""
    return (df["status"] == "active") & (df["balance"] > 0)


# Or as a class
class CustomBusinessCheck(BaseCheck):
    def validate(self, df):
        valid_mask = self._apply_business_logic(df)
        return CheckResult(
            check_name=self.name,
            passed=valid_mask.all(),
            failed_rows=df[~valid_mask].index.tolist(),
        )
```

### Conditional Validation

Apply checks conditionally.

```python
from dqx import ConditionalCheck

check = ConditionalCheck(
    condition=lambda df: df["region"] == "US",
    check=PatternCheck("phone", r"^\d{3}-\d{3}-\d{4}$"),
)
```

### Check Groups

Organize related checks.

```python
from dqx import CheckGroup

customer_checks = CheckGroup("Customer Validation")
customer_checks.add_checks(
    [
        NotNullCheck("customer_id"),
        UniqueCheck("customer_id"),
        PatternCheck("email", email_regex),
    ]
)

order_checks = CheckGroup("Order Validation")
order_checks.add_checks([NotNullCheck("order_id"), RangeCheck("amount", min_value=0)])
```

### Validation Pipelines

Chain validation stages.

```python
from dqx import ValidationPipeline

pipeline = ValidationPipeline()

# Add stages
pipeline.add_stage("basic", basic_checks, fail_fast=True)
pipeline.add_stage("business", business_checks)
pipeline.add_stage("statistical", stats_checks)

# Run pipeline
results = pipeline.run(data)
```

## Configuration

### Global Configuration

```python
from dqx import config

# Set global defaults
config.set_defaults(fail_fast=False, parallel=True, max_workers=4, sample_size=100000)

# Configure logging
config.set_logging(
    level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Validator Configuration

```python
# From file
validator = DataQualityValidator.from_config("config.yaml")

# Or programmatically
validator.configure(parallel=True, max_workers=8, memory_limit="4GB")
```

## Metrics and Monitoring

### Metrics Collection

```python
from dqx.metrics import MetricsCollector

collector = MetricsCollector()
validator.add_metrics_collector(collector)

# After validation
metrics = collector.get_metrics()
print(f"Validation time: {metrics['duration_ms']}ms")
print(f"Rows processed: {metrics['rows_processed']}")
```

### Export Formats

```python
# Prometheus format
prometheus_metrics = results.to_prometheus()

# StatsD format
statsd_metrics = results.to_statsd()

# Custom format
custom_metrics = results.export_metrics(formatter=lambda m: f"{m['name']}:{m['value']}")
```

## Error Handling

### Exception Types

```python
from dqx.exceptions import (
    ValidationError,
    DataSourceError,
    CheckConfigurationError,
    ConnectionError,
)

try:
    results = validator.validate(data, checks)
except ValidationError as e:
    print(f"Validation failed: {e}")
except DataSourceError as e:
    print(f"Data source error: {e}")
```

### Error Recovery

```python
# Retry logic
validator.validate_with_retry(data, checks, max_retries=3, retry_delay=1.0)

# Fallback handling
results = validator.validate_with_fallback(
    primary_source=sql_source, fallback_source=file_source, checks=checks
)
```

## Utilities

### Data Profiling

```python
from dqx.utils import DataProfiler

profiler = DataProfiler()
profile = profiler.analyze(df)

print(profile.summary())
# Shows: column types, null counts, unique values, etc.
```

### Check Generator

```python
from dqx.utils import CheckGenerator

generator = CheckGenerator()
suggested_checks = generator.suggest_checks(
    df, include_statistical=True, confidence_level=0.95
)
```

### Migration Tools

```python
from dqx.utils import migrate_rules

# Migrate from other formats
dqx_checks = migrate_rules(source="great_expectations", rules_file="expectations.json")
```

## Integration APIs

### REST API Client

```python
from dqx.api import DQXClient

client = DQXClient(base_url="https://dqx-api.company.com", api_key="your-api-key")

# Submit validation job
job_id = client.submit_validation(
    dataset_id="customers", check_suite_id="customer_checks"
)

# Get results
results = client.get_results(job_id)
```

### Webhook Integration

```python
from dqx.integrations import WebhookNotifier

notifier = WebhookNotifier(
    url="https://hooks.company.com/dqx", headers={"Authorization": "Bearer token"}
)

validator.add_notifier(notifier)
```

## Type Definitions

```python
from typing import TypedDict, List, Optional, Union


class CheckResult(TypedDict):
    check_name: str
    passed: bool
    failed_count: int
    failed_rows: List[int]
    error_message: Optional[str]


class ValidationSummary(TypedDict):
    total_checks: int
    passed_checks: int
    failed_checks: int
    pass_rate: float
    duration_ms: float


DataSource = Union[pd.DataFrame, str, SQLDataSource, FileDataSource]
```

## Constants and Enums

```python
from dqx.constants import Severity, CheckType


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckType(Enum):
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    UNIQUENESS = "uniqueness"
    CUSTOM = "custom"
```

---

For more examples and detailed usage, see the [User Guide](user-guide.md) and [examples directory](https://github.com/yourusername/dqx/tree/main/examples).
