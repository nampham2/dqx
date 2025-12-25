# YAML Configuration: Declarative Data Quality Suites

## Problem

Python code couples check definitions to the runtime environment. Teams cannot:

1. **Share checks** without sharing Python modules
2. **Generate checks** from external tools or AI systems
3. **Version checks** separately from application code
4. **Run checks remotely** by sending definitions over the network

## Solution

YAML configuration defines verification suites as data. A suite specifies checks, assertions, and profiles in a format that humans read, machines parse, and AI generates.

```yaml
name: "Order Validation Suite"
data_av_threshold: 0.9

checks:
  - name: "Price Checks"
    datasets: ["orders"]
    assertions:
      - name: "Average price is positive"
        metric: average(price)
        expect: "> 0"
        severity: P1
```

Load this configuration in Python:

```python
suite = VerificationSuite.from_yaml("suite.yaml", db=my_db)
suite.run([orders_datasource], key)
```

## Structure

A suite configuration contains three top-level sections:

```yaml
name: "Suite Name"           # Metadata
data_av_threshold: 0.9       # Metadata

checks:                      # Check definitions
  - name: "Check 1"
    assertions: [...]

profiles:                    # Suite-level behavior modifiers
  - name: "Profile 1"
    rules: [...]
```

Profiles apply to the entire suite, not individual assertions. An assertion's `tags` field determines which profile rules match it.

### Metadata

```yaml
name: "Order Validation Suite"    # Required. Identifies the suite.
data_av_threshold: 0.9            # Optional. Default: 0.9 (90%).
```

The `data_av_threshold` sets the minimum data availability ratio. Assertions skip evaluation when data falls below this threshold.

### Checks

Each check groups related assertions:

```yaml
checks:
  - name: "Price Checks"          # Required. Unique within the suite.
    datasets: ["orders"]          # Optional. Restricts check to these datasets.
    assertions:
      - name: "Average price is positive"
        metric: average(price)
        expect: "> 0"
```

### Assertions

An assertion validates one metric against one expectation:

```yaml
assertions:
  - name: "Null percentage below 10%"   # Required. Describes the rule.
    metric: null_count(email) / num_rows()   # Required. Expression to evaluate.
    expect: "< 0.1"                      # Required. Validation condition.
    severity: P0                         # Optional. Default: P1.
    tolerance: 0.01                      # Optional. For equality checks.
    tags: ["completeness", "critical"]   # Optional. For profile targeting.
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Human-readable description |
| `metric` | Yes | Metric expression (see Metric Syntax) |
| `expect` | Yes | Validation rule (see Expect Syntax) |
| `severity` | No | P0, P1, P2, or P3. Default: P1 |
| `tolerance` | No | Numeric tolerance for comparisons. Default: `1e-9` |
| `tags` | No | List of tags for profile rule matching |

## Profiles

Profiles are **suite-level** configuration. They modify assertion behavior during specific periods. Define profiles alongside `checks`, not inside them:

```yaml
profiles:
  - name: "Christmas 2024"
    type: holiday
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - check: "Volume Checks"
        action: disable
      - tag: "seasonal"
        metric_multiplier: 1.5
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Profile identifier |
| `type` | Yes | Profile type. Currently: `holiday` |
| `start_date` | Yes | Activation start (ISO format) |
| `end_date` | Yes | Activation end (ISO format) |
| `rules` | No | List of rules to apply |

### Profile Rules

Rules select assertions and specify modifications:

```yaml
rules:
  # Disable by check name
  - check: "Volume Checks"
    action: disable

  # Disable specific assertion
  - check: "Price Checks"
    assertion: "Minimum price above threshold"
    action: disable

  # Modify by tag
  - tag: "seasonal"
    metric_multiplier: 2.0      # Scale metric before validation
    severity: P2                # Override severity
```

| Field | Description |
|-------|-------------|
| `check` | Match assertions in this check |
| `assertion` | Match this specific assertion |
| `tag` | Match assertions with this tag |
| `action` | `disable` skips matched assertions |
| `metric_multiplier` | Scale factor applied before validation |
| `severity` | Override the assertion's severity |

## Metric Syntax

Metric expressions use function-call notation. The parser evaluates these expressions against the `MetricProvider` API.

### Simple Metrics

```yaml
metric: average(price)
metric: sum(revenue)
metric: minimum(quantity)
metric: maximum(temperature)
metric: num_rows()
metric: null_count(email)
metric: variance(score)
metric: unique_count(customer_id)
metric: first(timestamp)
```

### Parameters

Add dataset, lag, or custom parameters:

```yaml
metric: average(price, dataset=orders)
metric: average(price, lag=1)
metric: minimum(quantity, min_value=10)
```

| Parameter | Description |
|-----------|-------------|
| `dataset` | Restrict metric to this dataset |
| `lag` | Days offset (1 = yesterday) |
| Other | Passed to metric's `parameters` dict |

### Multi-Column Metrics

```yaml
metric: duplicate_count(columns=[order_id, date])
metric: count_values(status, "pending")
metric: count_values(active, true)
```

### Custom SQL

```yaml
metric: custom_sql("SUM(amount) / COUNT(*)")
metric: custom_sql("COUNT(DISTINCT user_id)", min_users=100)
```

### Extended Metrics

Extended metrics wrap base metrics:

```yaml
# Day-over-day absolute percentage change: abs((today - yesterday) / yesterday)
# Returns 0.1 for 10% change, 0.5 for 50% change
metric: day_over_day(average(price))

# Week-over-week absolute percentage change: abs((today - week_ago) / week_ago)
# Returns 0.1 for 10% change, 0.5 for 50% change
metric: week_over_week(sum(revenue))

# Standard deviation over time window
# offset: starting lag (default: 0), n: window size (default: 7)
metric: stddev(day_over_day(average(tax)), offset=1, n=7)
metric: stddev(average(price))  # equivalent to offset=0, n=7
```

### Arithmetic

Combine metrics with arithmetic operators:

```yaml
metric: null_count(email) / num_rows()
metric: sum(revenue) - sum(cost)
metric: average(price) * 1.1
```

### Math Functions

```yaml
metric: abs(day_over_day(maximum(price)) - 1.0)
metric: sqrt(variance(score))
```

Supported functions: `abs`, `sqrt`, `log`, `exp`, `min`, `max`.

## Expect Syntax

The `expect` field specifies the validation condition:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `"> N"` | Greater than N | `"> 0"` |
| `">= N"` | Greater than or equal | `">= 100"` |
| `"< N"` | Less than N | `"< 0.1"` |
| `"<= N"` | Less than or equal | `"<= 1000"` |
| `"= N"` | Equal to N | `"= 0"` |
| `"between A and B"` | In range [A, B] | `"between 0.8 and 1.2"` |
| `"positive"` | Greater than zero | `"positive"` |
| `"negative"` | Less than zero | `"negative"` |
| `"collect"` | No validation; store value | `"collect"` |

Use `tolerance` for approximate equality:

```yaml
- name: "Ratio near 1.0"
  metric: average(price) / average(price, lag=1)
  expect: "= 1.0"
  tolerance: 0.05
```

## Complete Example

This configuration validates an e-commerce orders dataset:

```yaml
name: "E-Commerce Data Quality"
data_av_threshold: 0.8

checks:
  - name: "Completeness"
    datasets: ["orders"]
    assertions:
      - name: "No null customer IDs"
        metric: null_count(customer_id)
        expect: "= 0"
        severity: P0

      - name: "Email null rate below 5%"
        metric: null_count(email) / num_rows()
        expect: "< 0.05"

  - name: "Volume"
    datasets: ["orders"]
    assertions:
      - name: "At least 1000 orders"
        metric: num_rows()
        expect: ">= 1000"
        tags: ["volume"]

      - name: "Day-over-day volume stable"
        metric: day_over_day(num_rows())
        expect: "between 0.5 and 2.0"
        tags: ["volume", "trend"]

  - name: "Revenue"
    datasets: ["orders"]
    assertions:
      - name: "Total revenue positive"
        metric: sum(amount) + sum(tax)
        expect: "positive"
        severity: P0

      - name: "Average order value in range"
        metric: sum(amount) / num_rows()
        expect: "between 10 and 500"

  - name: "Cross-Dataset"
    datasets: ["orders", "returns"]
    assertions:
      - name: "Return rate below 15%"
        metric: num_rows(dataset=returns) / num_rows(dataset=orders)
        expect: "< 0.15"

profiles:
  - name: "Black Friday"
    type: holiday
    start_date: "2024-11-29"
    end_date: "2024-12-02"
    rules:
      - tag: "volume"
        metric_multiplier: 3.0

  - name: "System Maintenance"
    type: holiday
    start_date: "2024-12-15"
    end_date: "2024-12-15"
    rules:
      - check: "Volume"
        action: disable
```

## Python API

### Load Configuration

```python
from dqx import VerificationSuite

# From file
suite = VerificationSuite.from_yaml("suite.yaml", db=my_db)

# From string
yaml_content = """
name: "Quick Check"
checks:
  - name: "Basic"
    assertions:
      - name: "Has data"
        metric: num_rows()
        expect: "> 0"
"""
suite = VerificationSuite.from_yaml_string(yaml_content, db=my_db)
```

### Run Suite

```python
from dqx.common import ResultKey
from datetime import date

key = ResultKey(yyyy_mm_dd=date.today(), tags={"env": "prod"})
suite.run([orders_datasource], key)

results = suite.collect_results()
for r in results:
    print(f"{r.check}/{r.assertion}: {r.status}")
```

### Validate Configuration

```python
from dqx.config import validate_config

errors = validate_config("suite.yaml")
if errors:
    for e in errors:
        print(f"Error: {e}")
```

## JSON Schema

The JSON Schema validates configuration files at two levels:

Schema location: `src/dqx/suite.schema.json`

### Editor Integration

IDEs use the schema for real-time feedback:

- **Autocompletion** — Suggests field names and enum values
- **Error highlighting** — Marks invalid fields immediately
- **Hover documentation** — Shows field descriptions

Configure VS Code by adding to `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./src/dqx/suite.schema.json": "dqx-*.yaml"
  }
}
```

### Programmatic Validation

DQX validates configurations against the schema before loading:

```python
from dqx.config import validate_config_schema

# Returns list of validation errors (empty if valid)
errors = validate_config_schema("suite.yaml")
for error in errors:
    print(f"Schema error: {error}")
```

Schema validation catches structural errors:

- Missing required fields (`name`, `checks`)
- Invalid enum values (`severity: "P5"`)
- Wrong types (`data_av_threshold: "high"`)
- Extra unknown fields

The `load_config()` function validates automatically. To skip schema validation (e.g., for performance), use `load_config(..., validate_schema=False)`.

## Design Decisions

### String-Based Metric Syntax

Metrics use string expressions (`average(price)`) rather than nested objects. This choice favors readability and AI generation over strict typing.

The parser uses `sympy.sympify()` with a restricted namespace. This approach:

1. Handles arithmetic naturally
2. Supports nested function calls
3. Avoids custom parser complexity

### Whitelisted Functions

Math functions (`abs`, `sqrt`) come from a whitelist, not the full sympy namespace. This prevents arbitrary code execution when loading untrusted configurations.

### Profile Types

Only `holiday` profiles exist today. The `type` field reserves space for future types: `weekly`, `regional`, `percentage` (A/B tests).

## Future Extensions

| Feature | Status | Description |
|---------|--------|-------------|
| `includes` | Planned | Import checks from other files |
| `variables` | Planned | Parameterize thresholds |
| `environments` | Planned | Override values per environment |
| `custom_metrics` | Considered | Define metrics in configuration |
| `alerts` | Considered | Route failures to channels |
