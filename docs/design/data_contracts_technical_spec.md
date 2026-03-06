# Data Contracts Technical Specification

| Field | Value |
|-------|-------|
| **Author** | Data Platform Team |
| **Created** | 2025-02-15 |
| **Last Updated** | 2025-03-03 |
| **Version** | 2.2 |
| **Status** | Designing |
| **Tags** | data-contracts, data-quality, sla |
| **Related** | None |

---

> **Document Status**: Restructured for clarity (Type System before SLA Specification)
>
> **Key Improvements**:
> - 12 flexible PyArrow types (down from 25) with compatibility matching
> - Structured SLA metadata with cron-based scheduling and auto-generated freshness checks
> - Function-based check registry with improved error messages
> - Runtime schema validation for suite reusability across environments

---

## Problem Statement

DQX requires Python code for data quality checks. Teams need **declarative YAML contracts** that integrate with DQX's suite → checks → assertions hierarchy while defining schemas (PyArrow types with nullability), table-level checks (row counts, freshness), and SLA guarantees with automatic validation.

Contracts generate standard `VerificationSuite` instances that return `AssertionResult` objects, support custom check types via plugins, and make requirements explicit, version-controlled, and portable.

---

## Architecture Overview

### Core Design Principle

**Contracts are column-centric YAML specifications that generate VerificationSuites following the suite → checks → assertions pattern.**

```text
Contract YAML (schema + checks)
    ↓ Contract.from_yaml()
Contract instance (with PyArrow schema)
    ↓ contract.to_verification_suite(db)
VerificationSuite (standard, reusable)
    ↓ suite.run([datasource], result_key)
    ↓ [Schema validation at runtime via PyArrow]
AssertionResult[] (standard)
```

Contracts treat columns as first-class citizens with co-located schema and checks. The simplified PyArrow schema uses 12 flexible types for validation, accepting compatible variations (e.g., `int` accepts any integer width). All fields explicitly specify type and description; nullable defaults to true. Runtime schema validation enables suite reuse across environments.

---

## YAML Contract Structure

### Complete Schema Structure

```yaml
# Required: Contract metadata
name: string              # Contract name (1-255 characters)
version: string           # Semantic version (e.g., "1.0.0")
description: string       # Contract/table description
owner: string            # Team or individual owner
dataset: string          # Dataset name to validate (must match datasource.name)
tags: [string, ...]      # Optional tags (e.g., ["revenue", "core"])

# Optional: Structured SLA (see SLA Specification section)
sla:
  schedule: string               # Cron expression for data arrival schedule
  lag_hours: int                 # Hours after schedule until data available

# Optional: Table-level metadata (flat at top level)
metadata:
  partitioned_by: [string, ...]  # Column names used for partitioning
  # ... custom metadata key-value pairs

# Required: Unified columns (schema + checks together)
columns:
  - name: string                   # Required: Column name
    type: string | object          # Required: Simple type (string) or complex type (object)
    nullable: true|false           # Optional: Defaults to true if not specified
    description: string            # Required: Column description

    # Optional: Field-level metadata
    metadata:
      # ... custom metadata key-value pairs

    # Optional: Column checks (can be omitted for schema-only columns)
    checks:
      - name: string                  # Check name (required)
        type: string                  # Check type (e.g., "unique", "min")
        severity: "P0"|"P1"|"P2"|"P3"  # Default: "P1"
        # Type-specific parameters...

# Optional: Table-level checks
checks:
  - name: string                      # Check name (required)
    type: string                      # Check type (e.g., "num_rows", "freshness")
    severity: "P0"|"P1"|"P2"|"P3"  # Default: "P1"
    # Type-specific parameters...
```

**Key points:**
- Schema and checks co-located in `columns` section
- Checks are optional; omit for schema-only columns
- Single `description` field describes both contract and table
- Table metadata at top level (sibling to `columns`)
- Checks defined only on top-level columns, not nested struct fields

### Type Field Format

Simple types use strings; complex types use objects with a `kind` field:

```yaml
# Simple type (string)
- name: order_id
  type: int
  nullable: false
  description: "Order ID"

# Complex type (object)
- name: created_at
  type:
    kind: timestamp
    tz: "UTC"
  nullable: false
  description: "Creation timestamp"
```

This unifies type information and signals intent clearly.

---

### Basic Contract Example

```yaml
name: "Orders Contract"
version: "1.0.0"
description: "Daily order records"
owner: "data-platform-team"
dataset: "orders"
tags: ["revenue"]

metadata:
  partitioned_by: ["order_date"]

columns:
  - name: order_id
    type: int
    nullable: false
    description: "Unique order identifier"
    metadata:
      primary_key: "true"
    checks:
      - name: "Order ID is unique"
        type: duplicates
        max: 0
        severity: P0

      - name: "Order ID is positive"
        type: min
        min: 1
        severity: P0

  - name: customer_id
    type: int
    nullable: false
    description: "Customer identifier"
    checks:
      - name: "Customer ID is positive"
        type: min
        min: 1
        severity: P1

  - name: total_amount
    type: decimal
    nullable: false
    description: "Total order amount in USD"
    checks:
      - name: "Amount is non-negative"
        type: min
        min: 0.0
        severity: P1

      - name: "Amount is reasonable"
        type: max
        max: 1000000.0
        severity: P1

  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      - name: "Status is valid"
        type: whitelist
        values: ["pending", "processing", "shipped", "delivered", "cancelled"]
        severity: P0

  # Schema-only columns (no checks)
  - name: order_date
    type: date
    nullable: false
    description: "Order date (for partitioning)"

  - name: payment_method
    type: string
    nullable: false
    description: "Payment method used"

  - name: is_gift
    type: bool
    nullable: false
    description: "Whether order is a gift"

  - name: notes
    type: string
    nullable: true
    description: "Order notes from customer"

# Table-level checks
checks:
  - name: "Daily volume within bounds"
    type: num_rows
    min: 100
    max: 1000000
    severity: P1
```

This contract generates a suite with five CheckNodes: one per column with checks (order_id, customer_id, total_amount, status), plus one for table checks. Schema-only columns validate type and nullability only.

---

## Type System

### Design Philosophy

The type system prioritizes validation flexibility over exact matching. Types accept compatible variations (e.g., `int` accepts int8 through int64), require parameters only when semantically necessary (e.g., timezone for timestamp), and default to the simplest form.

### Type Reference Table

| Type | YAML Format | Validates Against | Notes |
|------|-------------|-------------------|-------|
| **Primitive Types** | | | |
| `int` | `type: int` | int8, int16, int32, int64, uint8, uint16, uint32, uint64 | Any integer width, signed or unsigned |
| `float` | `type: float` | float32, float64 | Any float precision |
| `bool` | `type: bool` | bool | Boolean exact match |
| `string` | `type: string` | string, utf8 | UTF-8 text |
| `bytes` | `type: bytes` | binary, large_binary | Binary data |
| **Temporal Types** | | | |
| `date` | `type: date` | date32, date64 | Any date representation |
| `time` | `type: time` | time32(s/ms), time64(us/ns) | Any time representation |
| `timestamp` | `type: timestamp` or `type: {kind: timestamp}` or `type: {kind: timestamp, tz: "America/New_York"}` | timestamp(any unit) | Simple form is flexible; object form validates timezone (defaults to UTC) |
| **Decimal Type** | | | |
| `decimal` | `type: decimal` | decimal128(any), decimal256(any) | Any precision/scale |
| **Complex Types** | | | |
| `list` | `type: {kind: list, value_type: T}` | list\<T\> | Recursive validation of element type |
| `struct` | `type: {kind: struct, fields: [...]}` | struct\<fields\> | Recursive validation of field structure |
| `map` | `type: {kind: map, key_type: K, value_type: V}` | map\<K, V\> | Recursive validation of key/value types |

### Primitive Types

Five primitive types validate flexibly:

```yaml
# Integer - accepts any width (int8-int64, uint8-uint64)
- name: user_id
  type: int
  nullable: false
  description: "User ID"

# Float - accepts float32 or float64
- name: price
  type: float
  description: "Product price"

# Boolean
- name: is_active
  type: bool
  nullable: false
  description: "Active flag"

# String (UTF-8 text)
- name: name
  type: string
  nullable: false
  description: "User name"

# Bytes (binary data)
- name: thumbnail
  type: bytes
  nullable: true
  description: "Image thumbnail bytes"
```

For validation, exact widths and precisions are implementation details. We verify the semantic type, not storage format.

### Temporal Types

```yaml
# Date - accepts date32 or date64
- name: birth_date
  type: date
  nullable: false
  description: "Date of birth"

# Timestamp - three ways to use
# 1. Simple form (flexible - accepts any timezone or no timezone)
- name: event_time
  type: timestamp
  description: "Event timestamp"

# 2. Complex form with default UTC timezone
- name: created_at
  type:
    kind: timestamp
    # tz defaults to "UTC" (omitted)
  nullable: false
  description: "Creation timestamp in UTC"

# 3. Complex form with explicit timezone
- name: created_at_ny
  type:
    kind: timestamp
    tz: "America/New_York"
  nullable: false
  description: "Creation timestamp in New York time"

# Time - accepts time32 or time64 with any unit
- name: daily_event_time
  type: time
  nullable: false
  description: "Time of day when event occurred"
```

Most timestamps should be stored in UTC for consistency. Use simple form when timezone doesn't matter or varies; use complex form with UTC (default) for most cases; use explicit timezone when data must be in a specific timezone.

### Decimal Type

```yaml
# Decimal - accepts any precision/scale
- name: amount
  type: decimal
  nullable: false
  description: "Transaction amount"

# Validates: decimal128(10, 2), decimal128(18, 4), decimal256(38, 6), etc.
```

For initial validation, we verify it's a decimal type. Precision/scale parameters can be added later if needed for strict financial validation.

### Complex Types

#### List Type

```yaml
# Simple list (primitive element type)
- name: tags
  type:
    kind: list
    value_type: string
  nullable: true
  description: "Product tags"

# List with integer elements (any integer width accepted)
- name: item_ids
  type:
    kind: list
    value_type: int
  nullable: false
  description: "Item IDs"

# List with complex element type (struct)
- name: events
  type:
    kind: list
    value_type:
      kind: struct
      fields:
        - name: timestamp
          type:
            kind: timestamp
          nullable: false
          description: "Event timestamp"

        - name: event_type
          type: string
          nullable: false
          description: "Event type"

        - name: value
          type: float
          nullable: true
          description: "Event value"
  nullable: false
  description: "Event history"
```

#### Struct Type

```yaml
# Simple struct (flat)
- name: location
  type:
    kind: struct
    fields:
      - name: latitude
        type: float
        nullable: false
        description: "Latitude coordinate"

      - name: longitude
        type: float
        nullable: false
        description: "Longitude coordinate"

      - name: label
        type: string
        nullable: true
        description: "Location label"
  nullable: true
  description: "Geographic location"

# Nested struct
- name: address
  type:
    kind: struct
    fields:
      - name: street
        type: string
        nullable: false
        description: "Street address"

      - name: city
        type: string
        nullable: false
        description: "City name"

      - name: coordinates
        type:
          kind: struct
          fields:
            - name: lat
              type: float
              nullable: false
              description: "Latitude"

            - name: lon
              type: float
              nullable: false
              description: "Longitude"
        nullable: true
        description: "GPS coordinates"
  nullable: false
  description: "Complete address"
```

#### Map Type

```yaml
# Simple map (string keys and values)
- name: properties
  type:
    kind: map
    key_type: string
    value_type: string
  nullable: true
  description: "Custom properties"

# Map with integer keys (any integer type accepted)
- name: item_quantities
  type:
    kind: map
    key_type: int
    value_type: int
  nullable: false
  description: "Item ID to quantity mapping"

# Map with complex value type (struct)
- name: metrics
  type:
    kind: map
    key_type: string
    value_type:
      kind: struct
      fields:
        - name: value
          type: float
          nullable: false
          description: "Metric value"

        - name: unit
          type: string
          nullable: false
          description: "Unit of measurement"
  nullable: false
  description: "Performance metrics"
```

Complex types can be nested arbitrarily (e.g., list of structs with maps). All types validate recursively.

---

## SLA Specification

### Overview

Contracts support optional SLA metadata defining when data should arrive. SLAs auto-generate freshness checks, convert business requirements into executable validation, and document availability guarantees. The system uses standard cron expressions for scheduling and infers partition vs. table mode from `metadata.partitioned_by`.

SLA is optional—omit it for ad-hoc datasets or when freshness isn't a concern. When specified, two fields are required: `schedule` and `lag_hours`.

### SLA Structure

```yaml
# REQUIRED CONTRACT FIELDS
name: string
version: string
description: string
owner: string
dataset: string
tags: [string, ...]

# OPTIONAL: Structured SLA (can be omitted entirely)
sla:
  schedule: string               # REQUIRED (if sla specified): Cron expression (5-field format)
  lag_hours: int                 # REQUIRED (if sla specified): Hours after scheduled time until data available

# Optional table metadata
metadata:
  partitioned_by: [string, ...]  # If present → partition SLA, if absent → table SLA
```

**SLA Type Inference:**
- If `metadata.partitioned_by` exists → Partition-based SLA (incremental data)
- If `metadata.partitioned_by` absent → Table-based SLA (full table refresh)

### SLA Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schedule` | `string` | Yes* | Cron expression (5-field format) defining when data arrives |
| `lag_hours` | `int` | Yes* | Hours after scheduled time until data is available |

\* Required if `sla` block is specified. The entire `sla` block is optional at contract level.

### Auto-Generated Checks

When SLA metadata is specified, it automatically generates a freshness check. Gap detection is a separate concern that users add explicitly in the `checks` section.

**For Partitioned Tables** (`partitioned_by` exists):
```text
max_age_hours = lag_hours + period_hours + buffer

where:
  period_hours = inferred from cron schedule
    - Hourly (0 * * * *) → 1 hour
    - Daily (0 0 * * *) → 24 hours
    - Weekly (0 0 * * 1) → 168 hours
  buffer = 1 hour (default tolerance)
```

**For Non-Partitioned Tables** (`partitioned_by` absent):
```text
max_age_hours = lag_hours + buffer
```

**Generated Check:**
```yaml
checks:
  - name: "SLA: Freshness check"
    type: freshness
    max_age_hours: <calculated>
    timestamp_column: <inferred from partitioned_by or specified in check>
    severity: P0
```

**Note:** The `timestamp_column` is inferred from the first column in `metadata.partitioned_by` for partitioned tables, or must be explicitly specified in the freshness check for non-partitioned tables.

Gap detection is NOT auto-generated. Users explicitly add `completeness` checks in the `checks` section if needed.

### Complete Examples

#### Example 1: Daily Partitioned Table (T-1 Availability)

```yaml
name: "E-commerce Orders Contract"
version: "2.0.0"
description: "Daily order records from e-commerce platform"
owner: "data-platform-team"
dataset: "orders"
tags: ["revenue", "core", "pii"]

sla:
  schedule: "0 0 * * *"          # Every day at midnight
  lag_hours: 24                  # T-1 data (data for day D by end of D+1)

metadata:
  partitioned_by: ["order_date"] # ← Indicates partition-based SLA
  owner_team: "finance"
  pii_contains: "true"

columns:
  - name: order_date
    type: date
    nullable: false
    description: "Order date (partition key)"

  - name: order_id
    type: int
    nullable: false
    description: "Unique order identifier"
    checks:
      - name: "Order ID is unique"
        type: duplicates
        max: 0
        severity: P0

  - name: customer_id
    type: int
    nullable: false
    description: "Customer identifier"
    metadata:
      foreign_key: "customers.id"

  - name: total_amount
    type: decimal
    nullable: false
    description: "Total order amount in USD"
    checks:
      - name: "Amount is non-negative"
        type: min
        min: 0.0
        severity: P1

  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      - name: "Status is valid"
        type: whitelist
        values: ["pending", "processing", "shipped", "delivered", "cancelled"]
        severity: P0

  - name: created_at
    type:
      kind: timestamp
    nullable: false
    description: "Order creation timestamp"

  - name: updated_at
    type:
      kind: timestamp
    nullable: true
    description: "Last update timestamp"

  # Complex types
  - name: tags
    type:
      kind: list
      value_type: string
    nullable: true
    description: "Order tags for categorization"

  - name: shipping_address
    type:
      kind: struct
      fields:
        - name: street
          type: string
          nullable: false
          description: "Street address"

        - name: city
          type: string
          nullable: false
          description: "City name"

        - name: postal_code
          type: string
          nullable: false
          description: "Postal/ZIP code"

        - name: country
          type: string
          nullable: false
          description: "ISO country code"
    nullable: false
    description: "Shipping address information"

  - name: line_items
    type:
      kind: list
      value_type:
        kind: struct
        fields:
          - name: item_id
            type: int
            nullable: false
            description: "Product item ID"

          - name: quantity
            type: int
            nullable: false
            description: "Quantity ordered"

          - name: unit_price
            type: decimal
            nullable: false
            description: "Price per unit"
    nullable: false
    description: "Order line items"

# AUTO-GENERATED CHECK FROM SLA:
# checks:
#   - name: "SLA: Freshness check"
#     type: freshness
#     max_age_hours: 49              # 24 lag + 24 period + 1 buffer
#     timestamp_column: order_date   # Inferred from partitioned_by
#     severity: P0

# User can optionally add gap detection:
# checks:
#   - name: "No missing partitions in last 7 days"
#     type: completeness
#     partition_column: order_date
#     granularity: daily
#     lookback_days: 7
#     allow_future_gaps: true
#     severity: P1
```

**Interpretation:**
- Schedule: Daily at midnight (cron: `0 0 * * *`)
- Lag: 24 hours after day ends
- Freshness check: Auto-generated, latest partition must be within 49 hours (24 + 24 + 1)
- Gap check: Optional, user adds manually if needed

---

#### Example 2: Hourly Event Stream (2-hour Lag)

```yaml
name: "Clickstream Events Contract"
version: "1.0.0"
description: "Hourly clickstream events"
owner: "analytics-team"
dataset: "events"
tags: ["clickstream"]

sla:
  schedule: "0 * * * *"          # Every hour
  lag_hours: 2                   # Hour H data available by H+2

metadata:
  partitioned_by: ["event_hour"]

columns:
  - name: event_hour
    type:
      kind: timestamp
    nullable: false
    description: "Event hour (partition key)"

  - name: event_id
    type: int
    nullable: false
    description: "Unique event ID"

  - name: user_id
    type: int
    nullable: false
    description: "User identifier"

  - name: event_type
    type: string
    nullable: false
    description: "Event type"

# AUTO-GENERATED:
# - Freshness: max_age_hours = 2 + 1 + 1 = 4
```

---

#### Example 3: Business Days Only (Mon-Fri)

```yaml
name: "Business Day Reports Contract"
version: "1.0.0"
description: "Business day reports (Mon-Fri)"
owner: "reporting-team"
dataset: "daily_reports"

sla:
  schedule: "0 6 * * 1-5"        # Mon-Fri at 6 AM
  lag_hours: 24                  # T-1 business day

metadata:
  partitioned_by: ["report_date"]

columns:
  - name: report_date
    type: date
    nullable: false
    description: "Report date (partition key)"

  - name: total_revenue
    type: decimal
    nullable: false
    description: "Total revenue for the day"

# AUTO-GENERATED:
# - Freshness: max_age_hours = 24 + 24 + 1 = 49
```

---

#### Example 4: Non-Partitioned Daily Refresh (6 AM)

```yaml
name: "Customer Aggregates Contract"
version: "1.0.0"
description: "Daily customer aggregate table"
owner: "analytics-team"
dataset: "customer_agg"

sla:
  schedule: "0 6 * * *"          # Daily at 6 AM
  lag_hours: 0                   # Available promptly at 6 AM

# NO metadata.partitioned_by → Table-based SLA
# timestamp_column must be specified in the freshness check

columns:
  - name: customer_id
    type: int
    nullable: false
    description: "Customer identifier"

  - name: last_updated
    type:
      kind: timestamp
    nullable: false
    description: "Last refresh timestamp"

  - name: total_orders
    type: int
    nullable: false
    description: "Total number of orders"

# AUTO-GENERATED:
# - Freshness: max_age_hours = 0 + 1 = 1 (must be within 1 hour)
```

### Validation Rules

**SLA Validation:**
1. SLA is optional at contract level
2. If SLA specified, both fields (`schedule`, `lag_hours`) required
3. `schedule` must be valid 5-field cron expression
4. `lag_hours` should be reasonable for schedule frequency (warn if > 168 hours for hourly/daily)
5. For partitioned tables, freshness check uses first column in `partitioned_by` as timestamp_column
6. For non-partitioned tables, freshness check must explicitly specify `timestamp_column`

### SLA vs Manual Checks

**With SLA** (auto-generates freshness only):
```yaml
sla:
  schedule: "0 0 * * *"
  lag_hours: 24

metadata:
  partitioned_by: ["order_date"]  # timestamp_column inferred from this

# Auto-generates freshness check with timestamp_column: order_date
```

**Without SLA** (manual checks):
```yaml
checks:
  - name: "Latest partition is fresh"
    type: freshness
    max_age_hours: 49
    timestamp_column: order_date  # Explicitly specified in check
    severity: P0
```

SLA metadata reduces boilerplate, ensures correct `max_age_hours` calculation, self-documents availability requirements, and provides machine-readable data for orchestration tools.

---

## Check Types Reference

### Overview

Data contracts support two categories of checks:

1. **Table-Level Checks**: Validate table-wide properties (row counts, freshness, duplicates, partitions)
2. **Column-Level Checks**: Validate individual column values (nullability, uniqueness, ranges, patterns, distributions)

**Note:** Cross-dataset validation (referential integrity, schema consistency, aggregate reconciliation) should be handled at the orchestration/workflow level rather than in individual data contracts. This keeps contracts focused on single-dataset quality while allowing orchestration tools to manage relationships between datasets.

**Naming Conventions:**
- Check names should be descriptive business statements (e.g., "Order ID is unique")
- Severity levels: `P0` (critical), `P1` (important), `P2` (nice-to-have), `P3` (informational)
- All checks support optional `tags` parameter for categorization
- Check type names in YAML contracts use `snake_case` (e.g., `num_rows`, `duplicates`, `nulls`).

**Check Structure:**

Every check shares a common structure:

```yaml
- name: "Descriptive business name"   # required
  type: <check_type>                   # required
  severity: P0 | P1 | P2 | P3         # required
  <validator>: <value>                 # optional — exactly one of: min, max, between, equals
  tolerance: <value>                   # optional
  # check-specific params (e.g. return: count/pct, columns:, values:, pattern:)
```

The `name` is a descriptive business statement. The `type` identifies which check to run. `severity` sets the priority level. A single **validator** — `min`, `max`, `between`, or `equals` — defines the acceptance condition; only one may be specified per check. The optional `tolerance` parameter allows acceptable variance. Check-specific parameters (such as `return`, `columns`, `values`, `pattern`) are documented in each check's detail section.

---

### Check Types Summary

#### Table-Level Checks

Validate table-wide properties and aggregates. Specified in the top-level `checks` array.

| Check Type | Description | Return |
|------------|-------------|--------|
| [`num_rows`](#num_rows) | Total row count validation | count |
| [`duplicates`](#duplicates) | Count of duplicate rows | count or pct |
| [`freshness`](#freshness) | Data recency validation | age_hours |
| [`completeness`](#completeness) | Partition gap detection | gap_count |

**Total: 4 table-level checks**

---

#### Column-Level Checks

Validate individual column values and properties. Specified within the `checks` array of a column definition.

##### Statistical Checks

Statistical checks compute aggregate metrics over the entire column and return a single numeric value.

| Check Type | Description |
|------------|-------------|
| [`cardinality`](#cardinality) | Count of distinct values |
| [`min`](#min) | Minimum value in column |
| [`max`](#max) | Maximum value in column |
| [`mean`](#mean) | Average value validation |
| [`sum`](#sum) | Sum of values validation |
| [`count`](#count) | Count of non-null values |
| [`variance`](#variance) | Variance validation |
| [`percentile`](#percentile) | Percentile value validation |

##### Value Checks

Value checks validate individual values within a column. Each check returns either an absolute count (`count`) or percentage (`pct`) controlled by the `return` parameter. For `nulls` and `duplicates`, the return value represents the count/percentage of null or duplicate values found. For `whitelist`, `blacklist`, `pattern`, and `length`, the return value represents the count/percentage of valid (conforming) rows.

| Check Type | Description |
|------------|-------------|
| [`nulls`](#nulls) | Null value validation |
| [`duplicates`](#duplicates) | Duplicate value validation |
| [`whitelist`](#whitelist) | Values in allowed set |
| [`blacklist`](#blacklist) | Values in disallowed set |
| [`pattern`](#pattern) | Values matching regex pattern |
| [`length`](#length) | Values within length bounds |

**Total: 14 column-level checks** (8 Statistical + 6 Value Checks)

---

**Note:**
- All checks support standard validators: `min`, `max`, `between`, `equals`, and `tolerance`
- `freshness` returns `age_hours` and validates implicitly with `max` = `max_age_hours`
- `completeness` returns `gap_count` and validates implicitly with `max` = `max_gap_count`
- Check type names are linked to their detailed definitions below

---

### Parameter Conventions

Data contract checks use two intuitive parameter patterns: **min/max** (explicit bounds) and **between** (range shorthand). These patterns work together—use `between` as a convenience for specifying both bounds.

#### Pattern 1: Explicit Min/Max

Use `min` and `max` parameters to set explicit bounds. Use only the bound you need, or both for range validation.

```yaml
# Lower bound only (for "higher is better" checks)
checks:
  - name: "At least 10 distinct merchants"
    type: cardinality
    min: 10  # At least 10 distinct values
    severity: P1

# Upper bound only (for "lower is better" checks)
checks:
  - name: "Low null percentage"
    type: nulls
    return: pct
    max: 0.05  # 5% null percentage
    severity: P1

# Both bounds (range validation)
checks:
  - name: "Stable row count"
    type: num_rows
    min: 1000
    max: 100000
    severity: P1
```

**When to use which bound:**
- **`max` for "lower is better"** - nulls (count or pct), variance, duplicates
- **`min` for "higher is better"** - cardinality (for diversity)
- **Both for stability** - num_rows, mean, sum, percentile

#### Pattern 2: Between (Convenience)

Use `between` as shorthand for inclusive ranges. This is equivalent to specifying both `min` and `max`.

```yaml
checks:
  - name: "Revenue in expected range"
    type: sum
    between: [1000000, 5000000]
    severity: P0

# Equivalent to:
checks:
  - name: "Revenue in expected range"
    type: sum
    min: 1000000
    max: 5000000
    severity: P0
```

#### Parameter Guidelines

**Mutual Exclusivity:** `between` cannot be combined with `min` or `max`:

```yaml
# ✅ Valid: min only
min: 100

# ✅ Valid: max only
max: 1000

# ✅ Valid: both
min: 100
max: 1000

# ✅ Valid: between
between: [100, 1000]

# ❌ Invalid: between + min
between: [100, 1000]
min: 50

# ❌ Invalid: between + max
between: [100, 1000]
max: 2000
```

**Error Handling:**

When parameters conflict, validation will fail with a clear message:
```text
ValidationError: Cannot use 'between' with 'min' or 'max'.
Use 'between: [100, 1000]' OR 'min: 100, max: 1000', not both.
```

#### Choosing the Right Parameter

**Use `max` for "lower is better" checks:**
- Metrics where lower values indicate better quality
- Examples: `nulls` (with count return), `variance`, `duplicates`, `completeness` (uses `max_gap_count`)
- Rationale: You want to set an upper bound on "bad" metrics

**Use `min` for "higher is better" checks:**
- Metrics where higher values indicate better quality
- Examples: `cardinality` (for diversity)
- Rationale: You want to set a lower bound on "good" metrics

**Use both `min` and `max` (or `between`) for stability:**
- Metrics that can be too low OR too high
- Examples: `num_rows`, `mean`, `sum`, `percentile`
- Rationale: You want to detect drift in either direction

---

#### Validator Parameters (Mutually Exclusive)

All checks support four validator patterns (see Check Structure in the Overview):

> **Note:** Validators (`min`, `max`, `between`, `equals`) are optional for value checks (e.g., `whitelist`, `blacklist`, `pattern`, `length`) that define their own required parameters such as `values` or `pattern`. When a validator is omitted from a value check, the check acts as a boolean assertion — it passes if all rows conform (equivalent to requiring the conforming count to equal the total row count).

**1. Lower Bound (`min`)**
```yaml
type: mean
min: 10.0
```

**2. Upper Bound (`max`)**
```yaml
type: variance
max: 100.0
```

**3. Range (`between`)**
```yaml
type: num_rows
between: [1000, 100000]
```

**4. Exact Match (`equals`)**
```yaml
type: sum
column: revenue
equals: 1000000.0
```

---

#### Tolerance Parameter (Universal)

The `tolerance` parameter applies to ALL numeric comparisons and validators. It provides flexibility for floating-point comparisons and allows for acceptable variance in validations.

**Default Values:**
- Floating-point checks: `tolerance: 1e-6`
- Integer checks: `tolerance: 0`

**Semantics:**
- For `equals`: Passes if `|actual - expected| ≤ tolerance`
- For `min`: Passes if `actual ≥ (min - tolerance)`
- For `max`: Passes if `actual ≤ (max + tolerance)`
- For `between`: Passes if `actual ≥ (min - tolerance) AND actual ≤ (max + tolerance)`

**Examples:**

```yaml
# Equals with tolerance
columns:
  - name: unit_price
    checks:
      - type: mean
        equals: 50.0
        tolerance: 0.1  # Passes if 49.9 ≤ actual ≤ 50.1

# Min with tolerance (cardinality)
columns:
  - name: merchant_id
    checks:
      - type: cardinality
        min: 100  # At least 100 distinct merchants
        tolerance: 5  # Passes if actual ≥ 95

# Max with tolerance (nulls with pct return uses 0-1 scale)
columns:
  - name: email
    checks:
      - type: nulls
        return: pct
        max: 0.05  # 5% null percentage
        tolerance: 0.001  # Passes if actual ≤ 0.051

# Between with tolerance
checks:
  - type: num_rows
    between: [1000, 2000]
    tolerance: 10  # Passes if 990 ≤ actual ≤ 2010
```

---

### Table-Level Checks

Table-level checks are specified in the top-level `checks` array (sibling to `columns`). They validate table-wide properties and aggregates.

---

#### Volume Checks

##### `num_rows`

Validates that the total row count is within specified bounds.

**Parameters:**

**Note:** This check does NOT support `threshold` because direction is context-dependent. Most use cases require range validation to detect both missing data (too few rows) and duplicates/anomalies (too many rows).

**Example 1: Range Validation**

```yaml
checks:
  - name: "Daily volume within bounds"
    type: num_rows
    between: [100, 1000000]  # Between 100 and 1M rows
    severity: P1
```

**Example 2: Minimum Volume**

```yaml
checks:
  - name: "At least 1000 transactions per day"
    type: num_rows
    min: 1000  # At least 1000 rows
    severity: P0
```

**Example 3: Exact Row Count**

```yaml
checks:
  - name: "Partition has exactly 1000 records"
    type: num_rows
    equals: 1000  # Exactly 1000 rows
    tolerance: 0
    severity: P1
```

---

##### `duplicates`

Validates that the number of duplicate rows (based on specified columns) is within bounds.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[string]` | Yes | None | Columns to check for duplicates (composite key) |
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |

**Example 1: No Duplicates (Most Common)**

```yaml
checks:
  - name: "No duplicate orders"
    type: duplicates
    columns: ["order_id"]
    max: 0  # No duplicates allowed
    severity: P0
```

**Example 2: Allow Small Number of Duplicates**

```yaml
checks:
  - name: "Few duplicate user-date combinations"
    type: duplicates
    columns: ["user_id", "event_date"]
    max: 10  # At most 10 duplicate rows
    severity: P1
```

**Example 3: Exact Duplicate Count**

```yaml
checks:
  - name: "Expected duplicate rate"
    type: duplicates
    columns: ["transaction_id", "timestamp"]
    equals: 50  # Exactly 50 duplicates expected
    tolerance: 5  # Within 5
    severity: P2
```

**Example 4: Duplicate Percentage**

```yaml
checks:
  - name: "Low duplicate rate"
    type: duplicates
    columns: ["order_id"]
    return: pct
    max: 0.01  # At most 1% duplicate rows
    severity: P1
```

---

#### Freshness Checks

##### `freshness`

Validates that data is not stale (most recent timestamp is within acceptable age).

**Returns:** `age_hours` — the age of the most recent (or oldest, depending on `aggregation`) record in hours. Validated implicitly: the check passes when `age_hours <= max_age_hours`.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_age_hours` | `float` | Yes | None | Maximum allowed age in hours |
| `timestamp_column` | `string` | No | "created_at" | Column containing timestamps |
| `aggregation` | `string` | No | "max" | Aggregation method ("max" or "min") |

**Example 1: Basic Usage (Auto-generated from SLA)**

```yaml
checks:
  - name: "Data is fresh"
    type: freshness
    max_age_hours: 24
    timestamp_column: created_at
    severity: P0
```

**Example 2: Partition Freshness**

```yaml
checks:
  - name: "Latest partition is within 2 hours"
    type: freshness
    max_age_hours: 2.0
    timestamp_column: event_hour
    severity: P0
```

**Example 3: Oldest Record Within 7 Days**

```yaml
checks:
  - name: "Oldest record within 7 days"
    type: freshness
    max_age_hours: 168.0  # oldest record must be no more than 7 days old
    timestamp_column: ingestion_time
    aggregation: min
    severity: P2
```

---

##### `completeness`

Validates that partitioned data has no missing partitions within a time range (gap detection). This check only applies to tables with `metadata.partitioned_by` defined.

**Returns:** `gap_count` — the number of missing partitions found. Validated implicitly: the check passes when `gap_count <= max_gap_count`.

**Note:** For column-level null validation, use `nulls` check (with `return: count` for absolute counts or `return: pct` for percentages).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `partition_column` | `string` | Yes | None | Column used for partitioning (date/timestamp) |
| `granularity` | `string` | Yes | None | Expected granularity ("hourly", "daily", "weekly", "monthly") |
| `lookback_days` | `int` | No | `30` | Days to check for gaps |
| `allow_future_gaps` | `bool` | No | `true` | Ignore gaps in future dates |
| `max_gap_count` | `int` | No | `0` | Maximum allowed missing partitions |

**Example 1: Daily Partitions (No Gaps)**

```yaml
checks:
  - name: "No missing daily partitions in last 7 days"
    type: completeness
    partition_column: order_date
    granularity: daily
    lookback_days: 7
    allow_future_gaps: true
    severity: P1
```

**Example 2: Hourly Partitions**

```yaml
checks:
  - name: "Hourly partitions are complete"
    type: completeness
    partition_column: event_hour
    granularity: hourly
    lookback_days: 1
    severity: P0
```

**Example 3: Allow Some Gaps (Best Effort)**

```yaml
checks:
  - name: "Weekly partitions mostly complete"
    type: completeness
    partition_column: report_week
    granularity: weekly
    lookback_days: 90
    max_gap_count: 2
    severity: P2
```

---

### Column-Level Checks

Column-level checks are specified within the `checks` array of a column definition. They validate individual column values and statistical properties.

---

#### Value Checks

Value checks validate individual values within a column (rather than computing aggregates over the column). Each check defines its own return semantics. For `nulls` and `duplicates`, the return value represents null or duplicate values found. For `whitelist`, `blacklist`, `pattern`, and `length`, the return value represents conforming (valid) rows. Use the `return` parameter to specify the return type (`count` or `pct`).

---

##### `nulls`

Validates null values in a column. Returns count or percentage of null values based on the `return` parameter.

**Return Parameter:** Use `return: count` (default) to return absolute null count, or `return: pct` to return null percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |

**Example 1: Maximum Bound (Most Common)**

```yaml
columns:
  - name: notes
    type: string
    nullable: true
    description: "Optional customer notes"
    checks:
      - name: "Most orders have notes"
        type: nulls
        max: 1000  # At most 1000 nulls
        severity: P2
```

**Example 2: Range Validation**

```yaml
columns:
  - name: deleted_at
    type:
      kind: timestamp
    nullable: true
    description: "Deletion timestamp (null if not deleted)"
    checks:
      - name: "Few deletions expected"
        type: nulls
        between: [0, 50]  # Between 0 and 50 nulls
        severity: P1
```

**Example 3: Exact Null Count**

```yaml
columns:
  - name: optional_metadata
    type: string
    nullable: true
    description: "Optional metadata field"
    checks:
      - name: "Exactly 100 records have metadata"
        type: nulls
        equals: 900  # Expecting 900 nulls (100 non-null)
        tolerance: 0
        severity: P2
```

**Example 4: Using Percentage Return**

```yaml
columns:
  - name: email
    type: string
    nullable: true
    description: "Customer email address"
    checks:
      - name: "Low null percentage"
        type: nulls
        return: pct
        max: 0.10  # At most 10% nulls
        severity: P1
```

---

##### `whitelist`

Validates that non-null values match a whitelist of allowed values. Returns count or percentage of rows matching the whitelist.

**Return Parameter:** Use `return: count` (default) to return the count of matching rows, or `return: pct` to return the percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `values` | `list` | Yes | None | List of allowed values |
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |
| `case_sensitive` | `bool` | No | `true` | Whether comparison is case-sensitive (strings only) |

**Example 1: Basic Usage**

```yaml
columns:
  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      - name: "Status is valid"
        type: whitelist
        values: ["pending", "processing", "shipped", "delivered", "cancelled"]
        severity: P0
```

**Example 2: Case-Insensitive**

```yaml
columns:
  - name: country_code
    type: string
    nullable: false
    description: "ISO country code"
    checks:
      - name: "Country code is valid"
        type: whitelist
        values: ["US", "CA", "MX", "GB", "DE", "FR"]
        case_sensitive: false
        severity: P1
```

**Example 3: Numeric Enum**

```yaml
columns:
  - name: priority
    type: int
    nullable: false
    description: "Task priority level"
    checks:
      - name: "Priority is valid (1-5)"
        type: whitelist
        values: [1, 2, 3, 4, 5]
        severity: P0
```

**Example 4: Using Percentage Return**

```yaml
columns:
  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      - name: "Most statuses are valid"
        type: whitelist
        values: ["pending", "processing", "shipped", "delivered", "cancelled"]
        return: pct
        min: 0.95  # At least 95% match
        severity: P1
```

---

##### `blacklist`

Validates that non-null values do NOT match a blacklist. Returns count or percentage of rows NOT in the blacklist (passing rows).

**Return Parameter:** Use `return: count` (default) to return the count of passing rows, or `return: pct` to return the percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `values` | `list` | Yes | None | List of forbidden values |
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |
| `case_sensitive` | `bool` | No | `true` | Whether comparison is case-sensitive (strings only) |

**Example 1: Basic Usage**

```yaml
columns:
  - name: username
    type: string
    nullable: false
    description: "User username"
    checks:
      - name: "Username is not reserved"
        type: blacklist
        values: ["admin", "root", "system", "null", "undefined"]
        case_sensitive: false
        severity: P0
```

**Example 2: Exclude Test Data**

```yaml
columns:
  - name: email
    type: string
    nullable: false
    description: "Customer email address"
    checks:
      - name: "Email is not test account"
        type: blacklist
        values: ["test@example.com", "noreply@example.com", "admin@test.com"]
        severity: P1
```

---

##### `duplicates`

Validates that the count of duplicate values in a column is within specified bounds. This is the column-level version of the table-level `duplicates` check (which validates duplicates across multiple columns).

**Semantics:** Counts total duplicate occurrences. If value "A" appears 3 times, it contributes 3 to the duplicate count.

**Return Parameter:** Use `return: count` (default) to return absolute duplicate count, or `return: pct` to return duplicate percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |

**Note:** To enforce uniqueness (no duplicates), use `max: 0` or `equals: 0`.

**Example 1: No Duplicates (Uniqueness)**

```yaml
columns:
  - name: order_id
    type: int
    nullable: false
    description: "Unique order identifier"
    checks:
      - name: "Order ID is unique"
        type: duplicates
        max: 0  # No duplicates allowed
        severity: P0
```

**Example 2: Allow Some Duplicates**

```yaml
columns:
  - name: user_id
    type: int
    nullable: false
    description: "User identifier (repeat customers allowed)"
    checks:
      - name: "User ID has reasonable duplicates"
        type: duplicates
        max: 100  # Up to 100 duplicate user IDs
        severity: P1
```

**Example 3: Exact Duplicate Count (Testing)**

```yaml
columns:
  - name: test_category
    type: string
    nullable: false
    description: "Test category with expected duplicates"
    checks:
      - name: "Category has expected duplicates"
        type: duplicates
        equals: 50  # Expect exactly 50 duplicates
        tolerance: 0
        severity: P2
```

**Example 4: Duplicate Percentage**

```yaml
columns:
  - name: order_id
    type: int
    nullable: false
    description: "Order identifier"
    checks:
      - name: "Low duplicate rate"
        type: duplicates
        return: pct
        max: 0.01  # At most 1% duplicate values
        severity: P1
```

---

#### String Checks

String checks are part of Value Checks. See the Value Checks section above for context.

##### `pattern`

Validates that string values match a pattern. Supports explicit regex patterns or predefined format shortcuts. Returns count or percentage of values matching the pattern.

**Pattern Specification:** Use either:
- `pattern` parameter with a regex pattern (e.g., `"^[A-Z]{2}\\d{6}$"`)
- `format` parameter with a predefined format name (e.g., `"email"`, `"phone"`, `"uuid"`)

**Return Parameter:** Use `return: count` (default) to return count of matching values, or `return: pct` to return percentage (0-1 scale) of matching values.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pattern` | `string` | No* | None | Regular expression pattern to match |
| `format` | `string` | No* | None | Predefined format name ("email", "phone", "uuid", "url", "ipv4", "ipv6", "date", "datetime") |
| `flags` | `list[string]` | No | `[]` | Regex flags (e.g., "IGNORECASE", "MULTILINE") - only for pattern |
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |

\* Exactly ONE of `pattern` or `format` must be specified

**Example 1: Email Validation**

```yaml
columns:
  - name: email
    type: string
    nullable: false
    description: "Customer email address"
    checks:
      - name: "Email format is valid"
        type: pattern
        pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        severity: P0
```

**Example 2: Phone Number Format**

```yaml
columns:
  - name: phone
    type: string
    nullable: true
    description: "Phone number (US format)"
    checks:
      - name: "Phone number matches US format"
        type: pattern
        pattern: "^\\+1-\\d{3}-\\d{3}-\\d{4}$"
        severity: P1
```

**Example 3: Product SKU Pattern**

```yaml
columns:
  - name: sku
    type: string
    nullable: false
    description: "Product SKU code"
    checks:
      - name: "SKU follows naming convention"
        type: pattern
        pattern: "^[A-Z]{2}\\d{6}[A-Z]$"
        severity: P0
```

**Example 4: Percentage Return**

```yaml
columns:
  - name: email
    type: string
    nullable: false
    description: "Customer email address"
    checks:
      - name: "95% emails are valid"
        type: pattern
        pattern: "^[^@]+@[^@]+\\.[^@]+$"
        return: pct
        min: 0.95  # At least 95%
        severity: P1
```

**Example 5: Using Predefined Format (Email)**

```yaml
columns:
  - name: email
    type: string
    nullable: false
    description: "Customer email address"
    checks:
      - name: "All emails are valid"
        type: pattern
        format: email  # Predefined format instead of regex
        return: pct
        min: 0.95  # At least 95% valid
        severity: P1
```

**Example 6: Using Predefined Format (Phone)**

```yaml
columns:
  - name: phone
    type: string
    nullable: true
    description: "Phone number"
    checks:
      - name: "Phone numbers are valid"
        type: pattern
        format: phone  # Matches common phone formats
        return: count
        min: 1000  # At least 1000 valid phone numbers
        severity: P2
```

**Example 7: Using Predefined Format (UUID)**

```yaml
columns:
  - name: transaction_id
    type: string
    nullable: false
    description: "Transaction UUID"
    checks:
      - name: "Transaction IDs are valid UUIDs"
        type: pattern
        format: uuid  # Standard UUID format
        return: pct
        equals: 1.0  # 100% must be valid UUIDs
        severity: P0
```

**Example 8: Using Predefined Format (URL)**

```yaml
columns:
  - name: website
    type: string
    nullable: true
    description: "Website URL"
    checks:
      - name: "Websites are valid URLs"
        type: pattern
        format: url  # HTTP/HTTPS URLs
        return: pct
        min: 0.90  # At least 90% valid
        severity: P1
```

**Supported Format Shortcuts:**

| Format | Description | Equivalent Regex Pattern |
|--------|-------------|-------------------------|
| `email` | Email address | `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$` |
| `phone` | Phone number (international) | `^\\+?[1-9]\\d{1,14}$` |
| `uuid` | UUID (any version) | `^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$` |
| `url` | HTTP/HTTPS URL | `^https?://[^\\s/$.?#].[^\\s]*$` |
| `ipv4` | IPv4 address | `^(?:[0-9]{1,3}\\.){3}[0-9]{1,3}$` |
| `ipv6` | IPv6 address | `^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$` |
| `date` | ISO 8601 date (YYYY-MM-DD) | `^\\d{4}-\\d{2}-\\d{2}$` |
| `datetime` | ISO 8601 datetime | `^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}` |

**Note:** Format shortcuts provide common patterns for convenience. For custom formats or more specific validation, use the `pattern` parameter with an explicit regex.

---

##### `length`

Validates that string lengths are within specified bounds. Returns count or percentage of rows meeting the length criteria.

**Return Parameter:** Use `return: count` (default) to return count of rows within length bounds, or `return: pct` to return percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_length` | `int` | No* | None | Minimum allowed string length |
| `max_length` | `int` | No* | None | Maximum allowed string length |
| `return` | `string` | No | `count` | Return type: "count" or "pct" |

\* At least one of `min_length` or `max_length` required

**Example 1: Basic Usage**

```yaml
columns:
  - name: description
    type: string
    nullable: false
    description: "Product description"
    checks:
      - name: "Description is meaningful"
        type: length
        min_length: 10
        max_length: 500
        severity: P1
```

**Example 2: Fixed Length**

```yaml
columns:
  - name: country_code
    type: string
    nullable: false
    description: "ISO 3166-1 alpha-2 country code"
    checks:
      - name: "Country code is 2 characters"
        type: length
        min_length: 2
        max_length: 2
        severity: P0
```

**Example 3: Minimum Length Only**

```yaml
columns:
  - name: password_hash
    type: string
    nullable: false
    description: "Hashed password"
    checks:
      - name: "Password hash has minimum length"
        type: length
        min_length: 32
        severity: P0
```

**Example 4: Using Percentage Return**

```yaml
columns:
  - name: description
    type: string
    nullable: false
    description: "Product description"
    checks:
      - name: "Most descriptions are reasonable length"
        type: length
        min_length: 10
        max_length: 500
        return: pct
        min: 0.90  # At least 90% within bounds
        severity: P1
```

---

#### Statistical Checks

Statistical checks compute aggregate metrics over the entire column and return a single numeric value.

##### `cardinality`

Validates that the count of distinct (unique) non-null values is within specified bounds. This is a statistical aggregate over the entire column.

**Parameters:**

**Example 1: Low Cardinality (Categorical)**

```yaml
columns:
  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      - name: "Status is low-cardinality categorical"
        type: cardinality
        max: 10  # At most 10 distinct values
        severity: P1
```

**Example 2: Exact Cardinality**

```yaml
columns:
  - name: priority
    type: int
    nullable: false
    description: "Priority level"
    checks:
      - name: "Priority has exactly 5 levels"
        type: cardinality
        equals: 5  # Exactly 5 distinct values
        severity: P0
```

**Example 3: Cardinality Range**

```yaml
columns:
  - name: customer_segment
    type: string
    nullable: false
    description: "Customer segmentation tier"
    checks:
      - name: "Segment cardinality is stable"
        type: cardinality
        between: [3, 8]  # Between 3 and 8 segments
        severity: P1
```

---

##### `min`

Validates that the minimum value in the column meets specified criteria.

**Parameters:**

**Example 1: Minimum Must Be Non-Negative**

```yaml
columns:
  - name: price
    type: decimal
    nullable: false
    description: "Product price in USD"
    checks:
      - name: "All prices are non-negative"
        type: min
        min: 0.0
        severity: P0
```

**Example 2: Minimum Value in Range**

```yaml
columns:
  - name: temperature
    type: float
    nullable: false
    description: "Temperature in Celsius"
    checks:
      - name: "Minimum temperature is reasonable"
        type: min
        between: [-50.0, 0.0]
        severity: P1
```

---

##### `max`

Validates that the maximum value in the column meets specified criteria.

**Parameters:**

**Example 1: Maximum Must Be Reasonable**

```yaml
columns:
  - name: age
    type: int
    nullable: false
    description: "Customer age in years"
    checks:
      - name: "Maximum age is reasonable"
        type: max
        max: 120
        severity: P1
```

**Example 2: Maximum Value Equals Expected**

```yaml
columns:
  - name: priority
    type: int
    nullable: false
    description: "Task priority (1-5)"
    checks:
      - name: "Maximum priority is 5"
        type: max
        equals: 5
        severity: P0
```

---

##### `mean`

Validates that the arithmetic mean of numeric values is within specified bounds.

**Parameters:**

**Note:** This check does NOT support `threshold` because direction is context-dependent. Most use cases require range validation.

**Example 1: Range Validation**

```yaml
columns:
  - name: order_amount
    type: decimal
    nullable: false
    description: "Order amount in USD"
    checks:
      - name: "Average order value is typical"
        type: mean
        between: [50.0, 150.0]  # Between $50 and $150 average
        severity: P2
```

**Example 2: Monitoring Drift**

```yaml
columns:
  - name: response_time_ms
    type: int
    nullable: false
    description: "API response time in milliseconds"
    checks:
      - name: "Response time is acceptable"
        type: mean
        max: 500.0  # Average response under 500ms
        severity: P1
```

**Example 3: Exact Mean**

```yaml
columns:
  - name: rating
    type: int
    nullable: false
    description: "Product rating (1-5)"
    checks:
      - name: "Average rating matches target"
        type: mean
        equals: 3.5  # Exactly 3.5 average rating
        tolerance: 0.1  # Within 0.1
        severity: P2
```

---

##### `sum`

Validates that the sum of all non-null values in a column meets specified criteria.

**Parameters:**

**Example 1: Exact Sum**

```yaml
columns:
  - name: allocated_budget
    type: decimal
    nullable: false
    description: "Budget allocated to departments"
    checks:
      - name: "Total allocated budget is exactly $1M"
        type: sum
        equals: 1000000.0
        tolerance: 0.01
        severity: P0
```

**Example 2: Sum Range**

```yaml
columns:
  - name: daily_sales
    type: decimal
    nullable: false
    description: "Daily sales amount"
    checks:
      - name: "Daily sales within expected range"
        type: sum
        between: [10000.0, 100000.0]
        severity: P1
```

**Example 3: Minimum Sum**

```yaml
columns:
  - name: quantity
    type: int
    nullable: false
    description: "Order quantity"
    checks:
      - name: "Total quantity is positive"
        type: sum
        min: 0
        tolerance: 0
        severity: P0
```

---

##### `count`

Validates that the count of non-null values in a column meets specified criteria.

**Parameters:**

**Example 1: Exact Count**

```yaml
columns:
  - name: employee_id
    type: int
    nullable: false
    description: "Employee identifier"
    checks:
      - name: "Exactly 500 active employees"
        type: count
        equals: 500
        tolerance: 0
        severity: P1
```

**Example 2: Minimum Count**

```yaml
columns:
  - name: transaction_id
    type: int
    nullable: false
    description: "Transaction identifier"
    checks:
      - name: "At least 1000 transactions daily"
        type: count
        min: 1000
        severity: P2
```

**Example 3: Count Range**

```yaml
columns:
  - name: order_id
    type: int
    nullable: false
    description: "Order identifier"
    checks:
      - name: "Daily order count is stable"
        type: count
        between: [100, 1000]
        severity: P1
```

---

##### `variance`

Validates that the variance of numeric values is within specified bounds (measures spread).

**Parameters:**

**Example 1: Maximum Bound (Most Common)**

```yaml
columns:
  - name: temperature
    type: float
    nullable: false
    description: "Sensor temperature reading"
    checks:
      - name: "Temperature variance is stable"
        type: variance
        max: 10.0  # Variance ≤ 10.0
        severity: P2
```

**Example 2: Variance Range**

```yaml
columns:
  - name: daily_sales
    type: decimal
    nullable: false
    description: "Daily sales amount"
    checks:
      - name: "Sales variance is within expected range"
        type: variance
        between: [1000.0, 10000.0]  # Expected variance range
        severity: P2
```

**Example 3: Exact Variance**

```yaml
columns:
  - name: transaction_amount
    type: decimal
    nullable: false
    description: "Transaction amount"
    checks:
      - name: "Transaction variance matches expectation"
        type: variance
        equals: 50000.0  # Exactly 50000.0 variance
        tolerance: 100.0  # Within 100
        severity: P1
```

---

##### `percentile`

Validates that a specific percentile value is within specified bounds.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `percentile` | `float` | Yes | None | Percentile to check (0-100) |

**Note:** This check does NOT support `threshold` because direction is context-dependent. Most use cases require range validation.

**Example 1: P95 Response Time (Max Only)**

```yaml
columns:
  - name: response_time_ms
    type: int
    nullable: false
    description: "API response time in milliseconds"
    checks:
      - name: "P95 response time is acceptable"
        type: percentile
        percentile: 95.0
        max: 1000.0  # P95 under 1 second
        severity: P1
```

**Example 2: Median Value (Range)**

```yaml
columns:
  - name: order_amount
    type: decimal
    nullable: false
    description: "Order amount in USD"
    checks:
      - name: "Median order value is typical"
        type: percentile
        percentile: 50.0
        between: [30.0, 100.0]  # Median between $30-$100
        severity: P2
```

**Example 3: Exact Percentile Value**

```yaml
columns:
  - name: processing_time_sec
    type: float
    nullable: false
    description: "Processing time in seconds"
    checks:
      - name: "P99 processing time matches SLA"
        type: percentile
        percentile: 99.0
        equals: 300.0  # P99 exactly 300 seconds
        tolerance: 5.0  # Within 5 seconds
        severity: P0
```

---

### Check Composition Patterns

Checks can be combined to create comprehensive validation strategies. Here are common patterns:

#### Pattern 1: Layered Validation (Progressive Severity)

```yaml
columns:
  - name: email
    type: string
    nullable: false
    description: "Customer email address"
    checks:
      # Layer 1: Important - must have valid format
      - name: "Email format is valid"
        type: pattern
        format: email
        severity: P0

      # Layer 2: Nice-to-have - cardinality check
      - name: "Email domains are diverse"
        type: cardinality
        min: 10  # At least 10 distinct domains
        severity: P2
```

#### Pattern 2: Range Validation with Context

```yaml
columns:
  - name: order_amount
    type: decimal
    nullable: false
    description: "Order amount in USD"
    checks:
      # Hard constraint: Minimum value must be non-negative
      - name: "Minimum order amount is non-negative"
        type: min
        min: 0.0
        severity: P0

      # Business rule: Maximum value is reasonable
      - name: "Maximum order amount is reasonable"
        type: max
        max: 1000000.0
        severity: P1

      # Statistical monitoring: Average is stable
      - name: "Average order value is typical"
        type: mean
        min: 50.0
        max: 500.0
        severity: P2
```

#### Pattern 3: Multi-Level Validation

```yaml
columns:
  - name: phone_number
    type: string
    nullable: true
    description: "Customer phone number"
    checks:
      # Null percentage constraint (uses 0-1 scale)
      - name: "Most customers provide phone number"
        type: nulls
        return: pct
        max: 0.20  # At most 20% null
        severity: P1

      # Format validation (when present)
      - name: "Phone number format is valid"
        type: pattern
        pattern: "^\\+?[1-9]\\d{1,14}$"
        severity: P0

      # Length validation
      - name: "Phone number length is reasonable"
        type: length
        min: 10
        max: 15
        severity: P1
```

#### Pattern 4: Table-Level Reconciliation

```yaml
# Table-level checks
checks:
  # Volume check
  - name: "Daily volume within bounds"
    type: num_rows
    min: 1000
    max: 100000
    severity: P1

  # Freshness check
  - name: "Data arrived within SLA"
    type: freshness
    max_age_hours: 25.0
    timestamp_column: order_date
    severity: P0

  # Completeness check
  - name: "No missing daily partitions"
    type: completeness
    partition_column: order_date
    granularity: daily
    lookback_days: 7
    severity: P1
```

#### Pattern 5: Categorical with Cardinality

```yaml
columns:
  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      # Whitelist validation
      - name: "Status is valid"
        type: whitelist
        values: ["pending", "processing", "shipped", "delivered", "cancelled"]
        severity: P0

      # Low cardinality validation
      - name: "Status is low-cardinality"
        type: cardinality
        max: 10  # At most 10 distinct values
        severity: P1
```

---

---

## Summary

### Design Philosophy

DQX data contracts provide a declarative YAML interface defining schemas (PyArrow-based with 12 flexible types), SLAs (cron-based availability schedules), quality checks (column and table-level), and metadata (documentation and partitioning hints).

### Key Benefits

✅ **Declarative** - Express requirements in YAML, not Python code
✅ **Type-Safe** - PyArrow schema validation with complex type support
✅ **Flexible** - Types accept compatible variations (e.g., `int` accepts int8-int64)
✅ **Reusable** - Generated suites run against multiple datasources
✅ **Standard** - Produces standard `AssertionResult` objects
✅ **Extensible** - Custom check types via function registry
✅ **Self-Documenting** - Required descriptions enforce documentation culture

### Contract Structure Summary

```yaml
# Metadata
name: "Contract Name"
version: "1.0.0"
description: "What this data represents"
owner: "team-name"
dataset: "table_name"
tags: ["tag1", "tag2"]

# Optional SLA (2 fields)
sla:
  schedule: "0 0 * * *"        # Cron expression
  lag_hours: 24                # Availability lag

# Optional partitioning (timestamp_column inferred for freshness checks)
metadata:
  partitioned_by: ["event_date"]

# Schema with unified type field
columns:
  - name: column_name
    type: int                  # Simple type (string)
    nullable: false
    description: "Required description"

  - name: complex_column
    type:                      # Complex type (object)
      kind: list
      value_type: string
    nullable: true
    description: "Required description"
    checks:                    # Optional checks
      - name: "Check name"
        type: duplicates
        max: 0
        severity: P0

# Optional table-level checks
checks:
  - name: "Row count check"
    type: num_rows
    min: 100
    severity: P1
```

### Type System Summary

| Category | Types | Format |
|----------|-------|--------|
| **Primitive** | int, float, bool, string, bytes | `type: int` |
| **Temporal** | date, timestamp, time | `type: date` or `type: {kind: timestamp}` |
| **Decimal** | decimal | `type: decimal` |
| **Complex** | list, struct, map | `type: {kind: list, value_type: string}` |

### Next Steps

1. **Implementation** - Build YAML parser and contract validator
2. **Check Types** - Implement built-in check types (unique, min, max, etc.)
3. **SLA Integration** - Auto-generate freshness checks from SLA metadata
4. **Partition Validation** - Implement `completeness` check type for partition gap detection
5. **Testing** - Ensure 100% test coverage with TDD approach
6. **Documentation** - User guide with examples and best practices

---

## Appendix A: Cron Format Reference

### Cron Format

Standard 5-field cron format:

```text
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday = 0)
│ │ │ │ │
* * * * *
```

**Special Characters**:
- `*` - Any value (wildcard)
- `,` - List separator (e.g., `1,3,5`)
- `-` - Range (e.g., `1-5` = Monday through Friday)
- `/` - Step values (e.g., `*/6` = every 6 units)

### Common Cron Patterns

| Pattern | Cron Expression | Description |
|---------|-----------------|-------------|
| **Daily at midnight** | `0 0 * * *` | Every day at 00:00 |
| **Daily at 9 AM** | `0 9 * * *` | Every day at 09:00 |
| **Every hour** | `0 * * * *` | Top of every hour |
| **Every 6 hours** | `0 */6 * * *` | 00:00, 06:00, 12:00, 18:00 |
| **Business days** | `0 0 * * 1-5` | Mon-Fri at midnight |
| **Business days at 6 AM** | `0 6 * * 1-5` | Mon-Fri at 06:00 |
| **Monday only** | `0 0 * * 1` | Every Monday at midnight |
| **Tuesday and Thursday** | `0 0 * * 2,4` | Tue and Thu at midnight |
| **First of month** | `0 0 1 * *` | 1st day at midnight |
| **First and 15th** | `0 0 1,15 * *` | 1st and 15th at midnight |
| **Mon/Wed/Fri** | `0 0 * * 1,3,5` | Mon, Wed, Fri at midnight |

**Cron Testing Tools**:
- https://crontab.guru/ - Cron expression explainer
- https://crontab.cronhub.io/ - Cron validator

---

## Appendix B: Type Validation Rules

### Integer Type Compatibility

Contract type `int` validates against:
- `pa.int8()` - 8-bit signed integer (-128 to 127)
- `pa.int16()` - 16-bit signed integer (-32,768 to 32,767)
- `pa.int32()` - 32-bit signed integer (-2^31 to 2^31-1)
- `pa.int64()` - 64-bit signed integer (-2^63 to 2^63-1)
- `pa.uint8()` - 8-bit unsigned integer (0 to 255)
- `pa.uint16()` - 16-bit unsigned integer (0 to 65,535)
- `pa.uint32()` - 32-bit unsigned integer (0 to 2^32-1)
- `pa.uint64()` - 64-bit unsigned integer (0 to 2^64-1)

### Float Type Compatibility

Contract type `float` validates against:
- `pa.float32()` - 32-bit single precision (IEEE 754)
- `pa.float64()` - 64-bit double precision (IEEE 754)

### Date Type Compatibility

Contract type `date` validates against:
- `pa.date32()` - 32-bit signed integer, days since UNIX epoch
- `pa.date64()` - 64-bit signed integer, milliseconds since UNIX epoch

### Time Type Compatibility

Contract type `time` validates against:
- `pa.time32('s')` - 32-bit signed integer, seconds since midnight
- `pa.time32('ms')` - 32-bit signed integer, milliseconds since midnight
- `pa.time64('us')` - 64-bit signed integer, microseconds since midnight
- `pa.time64('ns')` - 64-bit signed integer, nanoseconds since midnight

### Timestamp Type Compatibility

**Simple form** (`type: timestamp`):
- Validates against any `pa.timestamp(unit, tz)` regardless of unit or timezone

**Complex form** (`type: {kind: timestamp}` or `type: {kind: timestamp, tz: "UTC"}`):
- Validates unit flexibility (accepts s, ms, us, ns)
- Validates timezone matches (default: "UTC")

**Complex form with explicit timezone** (`type: {kind: timestamp, tz: "America/New_York"}`):
- Validates unit flexibility (accepts s, ms, us, ns)
- Validates timezone exactly matches specified value

### Decimal Type Compatibility

Contract type `decimal` validates against:
- `pa.decimal128(precision, scale)` - Any precision/scale combination
- `pa.decimal256(precision, scale)` - Any precision/scale combination
