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

> **Document Status**: Split into focused documents for easier navigation
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

## Table of Contents

- [YAML Contract Structure](schema.md) — Schema structure, type field format, and basic contract examples
- [Type System](types.md) — PyArrow-based type definitions, primitive, temporal, decimal, and complex types
- [SLA Specification](sla.md) — Service level agreements, scheduling, auto-generated checks, and examples
- [Check Types Reference](checks.md) — Overview, parameter conventions, table-level checks, column-level checks, and composition patterns

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
