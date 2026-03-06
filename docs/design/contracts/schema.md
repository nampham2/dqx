# YAML Contract Structure

> Part of the [Data Contracts Technical Specification](index.md).

A contract YAML file has four sections: required metadata identifying the dataset, a `columns` section co-locating schema definitions and quality checks, an optional `sla` block for freshness guarantees, and an optional table-level `checks` array. This page documents the full structure and shows examples at two levels of complexity.

## Complete Schema Structure

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
        type: string                  # Check type (e.g., "duplicates", "min")
        severity: "P0"|"P1"|"P2"|"P3"  # Required
        # Type-specific parameters...

# Optional: Table-level checks
checks:
  - name: string                      # Check name (required)
    type: string                      # Check type (e.g., "num_rows", "freshness")
    severity: "P0"|"P1"|"P2"|"P3"  # Required
    # Type-specific parameters...
```

The `columns` section co-locates schema and checks for each column. A single `description` field describes both the contract and the table it governs. Table-level metadata sits at the top level as a sibling to `columns`. Checks attach only to top-level columns, not to nested struct fields. Omitting the `checks` key from a column produces a schema-only column: DQX validates its type and nullability but runs no quality assertions against it.

## Co-location Principle

Schema definitions and quality checks live together inside each column entry by design. Proximity keeps related information together, so a reader sees a column's type, nullability, and constraints in one place without jumping between sections. It also eliminates a common class of authoring error: a check that references a column not present in the schema cannot be written, because the check must nest inside a column that already declares its type. This co-location makes each column's full specification immediately visible and removes ambiguity about which checks apply to which columns.

## Type Field Format

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

## Minimal Contract Example

A minimal contract defines only metadata and columns. Without checks, each column validates type and nullability only. No suite is generated; DQX applies PyArrow schema enforcement at load time.

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

---

## Basic Contract Example

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

This contract generates a suite with five CheckNodes: one for each column that declares checks (`order_id`, `customer_id`, `total_amount`, `status`), plus one for the table-level `checks` array. The four schema-only columns (`order_date`, `payment_method`, `is_gift`, `notes`) produce no CheckNodes; DQX validates their types and nullability via PyArrow schema enforcement at load time. Because the suite binds to the dataset name `orders` rather than to a specific datasource, the same suite runs unchanged against any datasource whose registered name matches `orders` and whose schema satisfies the declared column types.

---
