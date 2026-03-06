# YAML Contract Structure

> Part of the [Data Contracts Technical Specification](index.md).

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

This contract generates a suite with five CheckNodes: one per column with checks (order_id, customer_id, total_amount, status), plus one for table checks. Schema-only columns validate type and nullability only.

---
