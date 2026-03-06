# SLA Specification

> Part of the [Data Contracts Technical Specification](index.md).

## Overview

Contracts support optional SLA metadata defining when data should arrive. SLAs auto-generate freshness checks, convert business requirements into executable validation, and document availability guarantees. The system uses standard cron expressions for scheduling and infers partition vs. table mode from `metadata.partitioned_by`.

SLA is optional—omit it for ad-hoc datasets or when freshness isn't a concern. When specified, two fields are required: `schedule` and `lag_hours`.

## SLA Structure

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

## SLA Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schedule` | `string` | Yes* | Cron expression (5-field format) defining when data arrives |
| `lag_hours` | `int` | Yes* | Hours after scheduled time until data is available |

\* Required if `sla` block is specified. The entire `sla` block is optional at contract level.

## Auto-Generated Checks

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

## Complete Examples

### Example 1: Daily Partitioned Table (T-1 Availability)

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

### Example 2: Hourly Event Stream (2-hour Lag)

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

### Example 3: Business Days Only (Mon-Fri)

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

### Example 4: Non-Partitioned Daily Refresh (6 AM)

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

## Validation Rules

**SLA Validation:**
1. SLA is optional at contract level
2. If SLA specified, both fields (`schedule`, `lag_hours`) required
3. `schedule` must be valid 5-field cron expression
4. `lag_hours` should be reasonable for schedule frequency (warn if > 168 hours for hourly/daily)
5. For partitioned tables, freshness check uses first column in `partitioned_by` as timestamp_column
6. For non-partitioned tables, freshness check must explicitly specify `timestamp_column`

## SLA vs Manual Checks

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
