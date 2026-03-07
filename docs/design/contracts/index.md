# Data Contracts Technical Specification

| Field | Value |
|-------|-------|
| **Author** | DQX |
| **Created** | 2025-02-15 |
| **Last Updated** | 2025-03-06 |
| **Version** | 2.2 |
| **Status** | Ready for review |
| **Tags** | data-contracts, data-quality, sla |
| **Related** | None |

---

## Overview

A data contract is a versioned YAML document that defines the schema, quality checks, and freshness guarantees for a dataset. Each contract names its dataset, declares the PyArrow type and nullability of every column, specifies optional SLA schedules, and attaches quality checks directly to the columns or dataset they govern.

Teams need contracts because data requirements must be explicit, version-controlled, testable, and portable. Without contracts, quality rules live in ad-hoc Python scripts that differ across environments, drift from the actual schema, and cannot be reviewed as data specifications. A contract replaces that scattered Python code with a single declarative YAML file that any engineer can read, diff, and own.

Contracts generate lists of `DecoratedCheck` functions — the same type that `VerificationSuite` already accepts. The user combines contract-generated checks with hand-coded checks in a single suite, so contract-based and custom validations run together and produce `AssertionResult` objects identical to those from hand-coded suites.

---

## Architecture

### Core Design Principle

**Contracts are column-centric YAML specifications that generate checks composable with hand-coded checks inside a standard VerificationSuite.**

```text
Contract YAML (schema + checks)
    ↓ Contract.from_yaml()                                          [proposed]
Contract instance (with PyArrow schema)
    ↓ contract.to_checks()                                          [proposed]
list[DecoratedCheck]
    ↓ VerificationSuite(checks=contract.to_checks() + [...], ...)   [existing API]
VerificationSuite
    ↓ suite.run([datasource], result_key)                           [existing API]
None
    ↓ suite.collect_results()                                       [existing API]
list[AssertionResult]
```

The runtime flow combines proposed and existing API. First, `Contract.from_yaml()` parses the YAML and builds a `Contract` instance with a fully resolved PyArrow schema — **proposed**. Second, `contract.to_checks()` translates every column definition and check into a list of `DecoratedCheck` functions — **proposed**. From here, the flow uses the **existing** `VerificationSuite` API: the user merges contract-generated checks with any hand-coded checks — `VerificationSuite(checks=contract.to_checks() + [custom_check], db=db, name=...)` — and calls `suite.run([datasource], result_key)` to execute all checks. Results are collected separately via `suite.collect_results()`, which returns `list[AssertionResult]`. `VerificationSuite`, `suite.run()`, `suite.collect_results()`, and `AssertionResult` already exist in the codebase today; only `Contract.from_yaml()` and `contract.to_checks()` are new. Schema type mismatches raise `SchemaValidationError` (proposed); contract parse errors raise `ContractValidationError` (proposed). `SchemaValidationError` is raised when a column's actual storage type does not match the declared contract type. `ContractValidationError` is raised when the YAML cannot be parsed into a valid `Contract` (e.g., missing required fields, invalid cron expression, or unknown check type).

---

## Contract Structure

The complete contract below shows every top-level section. The prose paragraphs that follow explain each section.

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

# Optional table-level checks
checks:
  - name: "Row count check"
    type: num_rows
    min: 100
    severity: P1

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
```

**Metadata.** Every contract begins with five required metadata fields that identify the dataset and its owner: `name` (a human-readable label), `version` (a version string; semantic versioning is recommended but not enforced — e.g., `"1.0.0"` or `"2025-03-07"`), `description` (a plain-English statement of what the data represents), `owner` (the responsible team), and `dataset` (the table or view name used at query time). An optional `tags` field accepts a list of strings for filtering and discovery.

**SLA.** The optional `sla` block defines when data should arrive. It takes two fields: `schedule`, a standard 5-field cron expression that declares the expected delivery cadence, and `lag_hours`, the number of hours the data may lag behind the scheduled time before triggering a failure. When both fields are present, DQX auto-generates a freshness check — no additional configuration required. See [SLA Specification](sla.md) for cron format reference and examples.

**Partitioning.** The optional `metadata` block declares the partitioning columns for the dataset. DQX reads `partitioned_by` to infer which column carries the timestamp used in freshness and completeness checks. When the SLA block references a freshness check and `partitioned_by` is set, DQX selects the first listed column as the timestamp column automatically.

**Table-level checks.** The top-level `checks` section validates properties of the dataset as a whole. `num_rows` asserts that the row count falls within a specified range. `duplicates` asserts that duplicate rows stay below a threshold. `freshness` asserts that data is not stale by checking record age against `max_age_hours` (defaults to the most recent record; set `aggregation: min` to check the oldest). `completeness` asserts that partition gaps — missing dates or time windows — stay below a specified count. `num_rows` and `duplicates` accept standard validators (`min`, `max`, `between`, `equals`, `tolerance`). `freshness` uses the implicit `max_age_hours` parameter instead of standard validators; `completeness` uses the implicit `max_gap_count` parameter instead.

**Columns.** The `columns` section is the heart of the contract. Each entry co-locates four pieces of information that belong together: the column's `type` (one of 12 flexible PyArrow types that accept compatible storage variations — `int` accepts int8 through int64, `float` accepts float32 and float64), its `nullable` flag (defaults to `true` when omitted), its required `description`, and an optional `checks` list. Co-locating schema and checks in a single entry makes the contract self-documenting: a reader sees the column's semantics and its quality requirements in one place. See [Type System](types.md) for the full compatibility matrix.

### Complete Schema Structure

The annotated schema below shows every field a contract file accepts, with types and defaults.

```yaml
# Required: Contract metadata
name: string              # Contract name (1-255 characters)
version: string           # Version string (e.g., "1.0.0"); semantic versioning recommended but not enforced
description: string       # Contract/table description
owner: string            # Team or individual owner
dataset: string          # Dataset name to validate (must match datasource.name)
tags: [string, ...]      # Optional tags (e.g., ["revenue", "core"])

# Optional: Structured SLA (see SLA Specification section)
sla:
  schedule: string               # Cron expression for data arrival schedule
  lag_hours: number              # Hours after schedule until data available (fractional values allowed, e.g. 1.5)

# Optional: Table-level metadata (flat at top level)
metadata:
  partitioned_by: [string, ...]  # Column names used for partitioning
  # ... custom metadata key-value pairs

# Optional: Table-level checks
checks:
  - name: string                      # Check name (required)
    type: string                      # Check type (e.g., "num_rows", "freshness")
    severity: "P0"|"P1"|"P2"|"P3"  # Required
    # Type-specific parameters...

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
        type: string                  # Check type (e.g., "duplicates", "min")
        severity: "P0"|"P1"|"P2"|"P3"  # Required
        # Type-specific parameters...
```

Omitting the `checks` key from a column produces a schema-only column: DQX validates its type and nullability but runs no quality assertions against it. Checks attach only to top-level columns, not to nested struct fields.

### Co-location Principle

Schema definitions and quality checks live together inside each column entry by design. Proximity keeps related information together, so a reader sees a column's type, nullability, and constraints in one place without jumping between sections. It also eliminates a common class of authoring error: a check that references a column not present in the schema cannot be written, because the check must nest inside a column that already declares its type.

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

### Minimal Contract Example

A minimal contract defines only metadata and columns. Without checks, DQX applies PyArrow schema enforcement at load time but generates no quality checks.

```yaml
name: "Products Contract"
version: "1.0.0"
description: "Product catalog records"
owner: "catalog-team"
dataset: "products"

columns:
  - name: product_id
    type: int
    nullable: false
    description: "Unique product identifier"

  - name: name
    type: string
    nullable: false
    description: "Product display name"

  - name: price_usd
    type: decimal
    nullable: false
    description: "List price in USD"

  - name: discontinued
    type: bool
    nullable: false
    description: "Whether the product is discontinued"
```

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
    between: [100, 1000000]
    severity: P1
```

This contract generates checks for four columns (`order_id`, `customer_id`, `total_amount`, `status`) plus one table-level check. The four schema-only columns (`order_date`, `payment_method`, `is_gift`, `notes`) produce no checks; DQX validates their types and nullability via PyArrow schema enforcement at load time. Because the checks bind to the dataset name `orders`, the same checks run unchanged against any datasource whose registered name matches and whose schema satisfies the declared types.

---

## Type System Summary

DQX contracts use 12 types that validate semantically, not by exact storage format. A column declared as `int` passes validation for any integer width the storage engine chooses; only the semantic category matters.

| Category | Types | Format |
|----------|-------|--------|
| **Primitive** | int, float, bool, string, bytes | `type: int` |
| **Temporal** | date, timestamp, time | `type: date` or `type: {kind: timestamp}` |
| **Decimal** | decimal | `type: decimal` |
| **Complex** | list, struct, map | `type: {kind: list, value_type: string}` |

Simple types (primitive, temporal, decimal) use a plain string value. Complex types (list, struct, map) use an object with a `kind` field and optional subtype fields. See [Type System](types.md) for the full compatibility rules per type.

---

## Check Types Summary

DQX contracts define 18 check types across two scopes. 11 are implemented today; 7 are planned for upcoming releases (marked below).

**4 table-level checks** validate the dataset as a whole:

- `num_rows` — asserts total row count
- `duplicates` — asserts count of duplicate rows
- `freshness` — asserts that data is not stale (record age does not exceed `max_age_hours`; defaults to most recent, optionally oldest via `aggregation: min`) *[planned]*
- `completeness` — asserts absence of partition gaps *[planned]*

**14 column-level checks** validate individual columns. 8 are statistical:

- `cardinality` — distinct value count
- `min` — minimum value
- `max` — maximum value
- `mean` — arithmetic mean
- `sum` — column sum
- `count` — non-null count
- `variance` — statistical variance
- `percentile` — value at a specified percentile *[planned]*

6 are value checks:

- `nulls` — null value count
- `duplicates` — duplicate value count within the column
- `whitelist` — all values belong to an allowed set *[planned]*
- `blacklist` — no values belong to a forbidden set *[planned]*
- `pattern` — all values match a regular expression *[planned]*
- `length` — string, list, or map element count *[planned]*

Most checks, table-level or column-level, support validators: `min`, `max`, `between`, `equals`, and `tolerance`. Exceptions are `freshness` (uses `max_age_hours`) and `completeness` (uses `max_gap_count`), which use check-specific implicit parameters instead. See [Check Types Reference](checks.md) for validators and composition patterns.

---

## Detailed References

- [Type System](types.md) — PyArrow-based type definitions, primitive, temporal, decimal, and complex types
- [SLA Specification](sla.md) — Service level agreements, scheduling, auto-generated checks, and examples
- [Check Types Reference](checks.md) — Overview, parameter conventions, table-level checks, column-level checks, and composition patterns
