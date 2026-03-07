# SLA Specification

> Part of the [Data Contracts Technical Specification](index.md).

## Overview

An SLA block defines when data should arrive. DQX auto-generates a freshness check from the two required SLA fields, converting business availability requirements into executable validation and documenting guarantees in a machine-readable form. The system uses standard 5-field cron expressions for scheduling and infers partition vs. table mode from `metadata.partitioned_by`.

Omit the `sla` block for ad-hoc datasets or when freshness is not a concern.

---

## SLA vs Manual Checks

Use the `sla` block when you want DQX to calculate `max_age_hours` for you and self-document the availability guarantee. Write a manual freshness check when you need direct control over every parameter.

**With SLA** (auto-generates freshness check):
```yaml
sla:
  schedule: "0 0 * * *"
  lag_hours: 24

metadata:
  partitioned_by: ["order_date"]  # timestamp_column inferred from first entry

# DQX auto-generates:
#   name: "SLA: Freshness check"
#   type: freshness
#   max_age_hours: 49              # 24 (lag) + 24 (daily period) + 1 (buffer)
#   timestamp_column: order_date
#   severity: P0
```

**Without SLA** (manual freshness check):
```yaml
checks:
  - name: "Latest partition is fresh"
    type: freshness
    max_age_hours: 49
    timestamp_column: order_date  # Must be specified explicitly
    severity: P0
```

The `sla` block eliminates the manual `max_age_hours` calculation, ensures the formula stays consistent across contracts, and gives orchestration tools a structured record of availability commitments.

---

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
  lag_hours: number                 # REQUIRED (if sla specified): Hours after scheduled time until data available

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
| `lag_hours` | `number` | Yes* | Hours after scheduled time until data is available |

\* Required if `sla` block is specified. The entire `sla` block is optional at contract level.

## Auto-Generated Checks

When SLA metadata is specified, DQX automatically generates a freshness check. Gap detection is a separate concern that users add explicitly in the `checks` section.

**For Partitioned Tables** (`partitioned_by` exists):
```text
max_age_hours = lag_hours + period_hours + buffer

where:
  period_hours = inferred from cron schedule
    - Hourly    (0 * * * *)       → 1 hour
    - Daily     (0 H * * *)       → 24 hours
    - Every N hours (0 */N * * *) → N hours
    - Business days (0 H * * 1-5) → 24 hours
    - Weekly    (0 H * * W)       → 168 hours
    - Monthly   (0 H 1 * *)       → 720 hours
    Cron expressions that do not match one of the above patterns
    (e.g. multi-day-of-week lists like "1,3,5") raise a ContractValidationError.
  buffer = 1 hour (fixed constant)
```

**For Non-Partitioned Tables** (`partitioned_by` absent):
```text
max_age_hours = lag_hours + buffer
```

**Generated Check:**

For the daily orders contract with `lag_hours: 24` and `schedule: "0 0 * * *"`, DQX generates:

```yaml
# For the daily orders contract with lag_hours: 24 and schedule: "0 0 * * *":
checks:
  - name: "SLA: Freshness check"
    type: freshness
    max_age_hours: 49          # 24 (lag) + 24 (daily period) + 1 (buffer)
    timestamp_column: order_date  # First column in metadata.partitioned_by
    severity: P0
```

**Note:** DQX infers `timestamp_column` from the first column in `metadata.partitioned_by` for partitioned tables. For non-partitioned tables, the freshness check must explicitly specify `timestamp_column`.

Gap detection is NOT auto-generated. Add `completeness` checks in the `checks` section explicitly when needed.

## Complete Examples

### Example 1: Daily Partitioned Table (T-1 Availability)

A daily e-commerce orders table where data for day D arrives by the end of day D+1 (24-hour lag). DQX generates a freshness check requiring the latest partition to be no older than 49 hours.

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

A clickstream events table partitioned by hour where each hour's data is available two hours after the hour closes. DQX generates a freshness check requiring the latest partition to be no older than 4 hours.

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

A reporting table that updates only on business days, with T-1 lag. DQX generates a freshness check using the daily 24-hour period, even though the schedule excludes weekends.

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

A daily customer aggregate table with a full-table refresh. Because there is no `partitioned_by`, DQX uses the table-based SLA formula and the freshness check must name the timestamp column explicitly.

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

# AUTO-GENERATED CHECK FROM SLA:
# checks:
#   - name: "SLA: Freshness check"
#     type: freshness
#     max_age_hours: 1              # 0 (lag) + 1 (buffer)
#     timestamp_column: last_updated  # Must be specified — cannot be inferred for non-partitioned tables
#     severity: P0
```

## Validation Rules

DQX enforces the following rules when it parses an `sla` block.

| Rule | Requirement |
|------|-------------|
| SLA is optional | Omit the `sla` block entirely for ad-hoc or non-time-sensitive datasets |
| Both fields required | If `sla` is present, both `schedule` and `lag_hours` must be specified |
| Valid cron expression | `schedule` must be a valid 5-field cron expression |
| Reasonable lag | `lag_hours` exceeding 168 hours on an hourly or daily schedule triggers a warning |
| Partitioned timestamp | For partitioned tables, DQX uses the first column in `partitioned_by` as `timestamp_column` |
| Non-partitioned timestamp | For non-partitioned tables, the freshness check must specify `timestamp_column` explicitly |

---

## Cron Format Reference

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
- `*` — Any value (wildcard)
- `,` — List separator (e.g., `1,3,5`)
- `-` — Range (e.g., `1-5` = Monday through Friday)
- `/` — Step values (e.g., `*/6` = every 6 units)

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
- https://crontab.guru/ — Cron expression explainer
- https://crontab.cronhub.io/ — Cron validator
