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

```
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
  timestamp_column: string       # Column to measure freshness

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
    type: string                      # Check type (e.g., "row_count", "freshness")
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
        type: unique
        severity: P0

      - name: "Order ID is positive"
        type: min
        value: 1
        severity: P0

  - name: customer_id
    type: int
    nullable: false
    description: "Customer identifier"
    checks:
      - name: "Customer ID is positive"
        type: min
        value: 1
        severity: P1

  - name: total_amount
    type: decimal
    nullable: false
    description: "Total order amount in USD"
    checks:
      - name: "Amount is non-negative"
        type: min
        value: 0.0
        severity: P1

      - name: "Amount is reasonable"
        type: max
        value: 1000000.0
        severity: P1

  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      - name: "Status is valid"
        type: allowed_values
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
    type: row_count
    min: 100
    max: 1000000
    severity: P1
```

This contract generates a suite with five CheckNodes: one per column with checks (order_id, customer_id, total_amount, status), plus one for table checks. Schema-only columns validate type and nullability only.

---

## PyArrow Type System

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

SLA is optional—omit it for ad-hoc datasets or when freshness isn't a concern. When specified, three fields are required: `schedule`, `lag_hours`, and `timestamp_column`.

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
  timestamp_column: string       # REQUIRED (if sla specified): Column to measure freshness

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
| `timestamp_column` | `string` | Yes* | Column to measure freshness (typically partition key) |

\* Required if `sla` block is specified. The entire `sla` block is optional at contract level.

### Auto-Generated Checks

When SLA metadata is specified, it automatically generates a freshness check. Gap detection is a separate concern that users add explicitly in the `checks` section.

**For Partitioned Tables** (`partitioned_by` exists):
```
max_age_hours = lag_hours + period_hours + buffer

where:
  period_hours = inferred from cron schedule
    - Hourly (0 * * * *) → 1 hour
    - Daily (0 0 * * *) → 24 hours
    - Weekly (0 0 * * 1) → 168 hours
  buffer = 1 hour (default tolerance)
```

**For Non-Partitioned Tables** (`partitioned_by` absent):
```
max_age_hours = lag_hours + buffer
```

**Generated Check:**
```yaml
checks:
  - name: "SLA: Freshness check"
    type: freshness
    max_age_hours: <calculated>
    timestamp_column: <from sla.timestamp_column>
    severity: P0
```

Gap detection is NOT auto-generated. Users explicitly add `partition_completeness` checks in the `checks` section if needed.

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
  timestamp_column: order_date

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
        type: unique
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
        value: 0.0
        severity: P1

  - name: status
    type: string
    nullable: false
    description: "Order status"
    checks:
      - name: "Status is valid"
        type: allowed_values
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
#     timestamp_column: order_date
#     severity: P0

# User can optionally add gap detection:
# checks:
#   - name: "No missing partitions in last 7 days"
#     type: partition_completeness
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
  timestamp_column: event_hour

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
  timestamp_column: report_date

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
  timestamp_column: last_updated

# NO metadata.partitioned_by → Table-based SLA

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
2. If SLA specified, all three fields (`schedule`, `lag_hours`, `timestamp_column`) required
3. `schedule` must be valid 5-field cron expression
4. `lag_hours` should be reasonable for schedule frequency (warn if > 168 hours for hourly/daily)
5. `timestamp_column` must reference a column in `columns[]`
6. For partitioned tables, `timestamp_column` should typically match one of `partitioned_by` columns (warn if not)

### SLA vs Manual Checks

**With SLA** (auto-generates freshness only):
```yaml
sla:
  schedule: "0 0 * * *"
  lag_hours: 24
  timestamp_column: order_date
# Auto-generates freshness check only
```

**Without SLA** (manual checks):
```yaml
checks:
  - name: "Latest partition is fresh"
    type: freshness
    max_age_hours: 49
    timestamp_column: order_date
    severity: P0
```

SLA metadata reduces boilerplate, ensures correct `max_age_hours` calculation, self-documents availability requirements, and provides machine-readable data for orchestration tools.

---

## Architecture Decisions

### Decision 1: Unified Column-Centric Structure

Columns are the primary organizational unit. Each column definition includes all information in one place: type, nullability, description, metadata, and optional checks. This creates a single source of truth per column and eliminates duplication.

We rejected separating schema and checks into different sections (duplicates column names), flat checks lists (less organized), and separate files (splits contract). Schema-only columns simply omit the `checks` field. Table metadata lives in the top-level `metadata` dictionary.

### Decision 2: Runtime Schema Validation

Column metadata (type, nullable) implicitly defines schema requirements. Schema validation happens at runtime when the suite executes, allowing suite reuse across multiple datasources (dev, staging, prod).

We rejected schema validation at suite generation (prevents reuse) and explicit schema checks in YAML (redundant, verbose). `suite.run()` validates all column types and nullability via DuckDB DESCRIBE, raising `DQXError` before check execution if schema is invalid.

### Decision 3: One Check Per Column

Following DQX's suite → checks → assertions hierarchy, each column with checks generates its own CheckNode. This provides clear grouping in results, ability to target specific column checks with profiles, and better parallelization opportunities.

We rejected a single "Column Checks" check for all columns (loses granularity), user-defined check grouping (adds complexity), and one check per assertion (too granular). A column with N checks generates a CheckNode named `"{column_name} Column Checks"` with N assertions.

### Decision 4: Table Checks Grouping

Table-level checks (row_count, freshness, etc.) are grouped into a single CheckNode named "Table Checks" for simplicity. We rejected one check per table assertion (creates many top-level checks), user-defined grouping (adds complexity), and nesting under columns (doesn't make logical sense).

### Decision 5: Multi-Column Checks in Table Section

Checks that operate on multiple columns (e.g., `duplicate_rate` on `[order_id, created_at]`) logically belong in table-level checks, not column-level checks. We rejected separate `multi_column_checks` section (unnecessary), nesting under one referenced column (arbitrary choice), and requiring separate check definition (verbose). Table checks can reference multiple columns via `columns: [...]` field.

### Decision 6: Schema-Only Columns Allowed

Not all columns require checks beyond schema validation. Columns can omit the `checks` field. We rejected requiring at least one check per column (overly restrictive), warnings on schema-only columns (false positives), and separate schema-only section (splits information). Columns without checks generate no CheckNode, only schema validation.

### Decision 7: PyArrow Schema as Foundation

PyArrow's mature, well-tested type system provides comprehensive types (25+), complex types (list, struct, map), metadata support, industry-standard validation, and interoperability with Parquet, Arrow, and DuckDB. We rejected custom type systems (reinventing the wheel), SQL DDL (dialect-specific, less portable), JSON Schema (not designed for columnar data), and Avro schema (less widespread in Python). This leverages PyArrow's validation, metadata, and tooling ecosystem.

### Decision 8: Simplified Type System (12 Types)

Use flexible, validation-focused types that accept compatible variations. This reduces complexity while supporting all real-world use cases. We include primitives (int, float, bool, string, bytes), temporals (date, timestamp, time), decimal, and complex types (list, struct, map). Types accept compatible variations: `int` accepts int8-int64 and uint variants, `float` accepts float32/float64, `date` accepts date32/date64, `timestamp` accepts any unit with optional timezone, `time` accepts time32/time64, and `decimal` accepts any precision/scale.

We excluded specific integer widths (use flexible `int`), specific float precisions (use flexible `float`), float16 (specialized hardware), large_string/large_binary (covered by flexible types), dictionary (encoding optimization), union (complex, rarely needed), fixed_size_binary (specialized), and interval (rarely used). This provides 52% reduction in type count (25 → 12) while database upgrades (e.g., INTEGER → BIGINT) don't break contracts.

### Decision 9: Nullable Defaults to True

Following SQL convention where columns are nullable by default unless explicitly constrained, this is the most permissive default and prevents breaking existing data pipelines. We rejected requiring explicit declaration (too verbose), defaulting to non-null like Avro (can break data unexpectedly), and inferring from datasource (contract should define requirements). If omitted, `nullable` defaults to `true`; specify `nullable: false` to require non-null.

### Decision 10: Description Required

Schema serves as documentation. Every field and table must have a description for self-documenting contracts. We rejected making description optional (results in undocumented schemas), generating from field names (inaccurate, lacks context), and allowing empty descriptions (defeats documentation purpose). YAML parsing fails if any field lacks description, enforcing documentation culture from day one.

### Decision 11: Partitioned_by Metadata

Physical data layout (partitioning) is important for query optimization, data validation context, pipeline documentation, and performance tuning. We rejected no partitioning information (loses context), inferring from datasource (contract should be authoritative), and separate partitioning file (splits information). Table metadata can specify `partitioned_by: [column_names]` for documentation, and future optimizations can leverage this information.

### Decision 12: Complex Types Fully Supported

Modern data formats (JSON, Parquet, Avro) use nested structures extensively. Data contracts must support lists (arrays of values), structs (nested objects), and maps (key-value pairs). We rejected flattening all structures (loses semantic meaning, verbose), supporting list only (struct and map equally important), and deferring complex types (cripples real-world adoption). Full support for nested structures with recursive validation enables arbitrary nesting (e.g., list of structs with maps).

### Decision 13: Type System Before SLA Specification

Readers must understand types before seeing SLA examples that use timestamp types extensively. This provides natural progression: basic structure → types → advanced features (SLA). We rejected SLA before types (examples reference unknown types) and interleaving (disrupts learning flow). Type system is foundational; SLA is an advanced feature that builds on type knowledge.

---

## Check Types Reference

### Column-Level Check Types

#### `unique`

Validates that all values in a column are unique.

```yaml
checks:
  - name: "Order ID is unique"
    type: unique
    severity: P0
```

**Parameters**: None

---

#### `min`

Validates that all values are greater than or equal to a minimum threshold.

```yaml
checks:
  - name: "Price is non-negative"
    type: min
    value: 0.0
    severity: P1
```

**Parameters**:
- `value` (required): Minimum allowed value

---

#### `max`

Validates that all values are less than or equal to a maximum threshold.

```yaml
checks:
  - name: "Price is reasonable"
    type: max
    value: 1000000.0
    severity: P1
```

**Parameters**:
- `value` (required): Maximum allowed value

---

#### `allowed_values`

Validates that all values are in an allowed set.

```yaml
checks:
  - name: "Status is valid"
    type: allowed_values
    values: ["pending", "processing", "shipped", "delivered", "cancelled"]
    severity: P0
```

**Parameters**:
- `values` (required): List of allowed values

---

### Table-Level Check Types

#### `row_count`

Validates that row count is within specified bounds.

```yaml
checks:
  - name: "Daily volume within bounds"
    type: row_count
    min: 100
    max: 1000000
    severity: P1
```

**Parameters**:
- `min` (optional): Minimum row count
- `max` (optional): Maximum row count

At least one of `min` or `max` must be specified.

---

#### `freshness`

Validates that data is not stale (most recent timestamp is recent enough).

```yaml
checks:
  - name: "Data is fresh"
    type: freshness
    max_age_hours: 24
    timestamp_column: created_at
    severity: P0
```

**Parameters**:
- `max_age_hours` (required): Maximum age in hours
- `timestamp_column` (optional): Column containing timestamps (default: "created_at")

---

#### `partition_completeness`

Validates that no partitions are missing in a time range.

```yaml
checks:
  - name: "No missing daily partitions"
    type: partition_completeness
    partition_column: order_date
    granularity: daily
    lookback_days: 7
    allow_future_gaps: true
    severity: P1
```

**Parameters**:
- `partition_column` (required): Column used for partitioning
- `granularity` (required): Expected partition granularity (daily, hourly, weekly, monthly)
- `lookback_days` (optional): Days to check for gaps (default: 30)
- `allow_future_gaps` (optional): Ignore future dates (default: true)

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

# Optional SLA (3 fields)
sla:
  schedule: "0 0 * * *"        # Cron expression
  lag_hours: 24                # Availability lag
  timestamp_column: event_date # Freshness column

# Optional partitioning
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
        type: unique
        severity: P0

# Optional table-level checks
checks:
  - name: "Row count check"
    type: row_count
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
4. **Partition Validation** - Implement `partition_completeness` check type
5. **Testing** - Ensure 100% test coverage with TDD approach
6. **Documentation** - User guide with examples and best practices

---

## Appendix A: Cron Format Reference

### Cron Format

Standard 5-field cron format:

```
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
