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
    ↓ VerificationSuite(checks=contract.to_checks() + [...], ...)
VerificationSuite
    ↓ suite.run([datasource], result_key)
AssertionResult[] (standard)
```

The proposed runtime flow has three steps. First, `Contract.from_yaml()` parses the YAML and builds a `Contract` instance with a fully resolved PyArrow schema. Second, `contract.to_checks()` translates every column definition and check into a list of `DecoratedCheck` functions — the same type `VerificationSuite` accepts — so the user merges them with any hand-coded checks: `VerificationSuite(checks=contract.to_checks() + [custom_check], db=db, name=...)`. Third, `suite.run([datasource], result_key)` executes all checks, validates the schema at runtime via PyArrow, and returns `AssertionResult[]` objects identical to those from hand-coded suites.

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

**Metadata.** Every contract begins with five required metadata fields that identify the dataset and its owner: `name` (a human-readable label), `version` (a semantic version string), `description` (a plain-English statement of what the data represents), `owner` (the responsible team), and `dataset` (the table or view name used at query time). An optional `tags` field accepts a list of strings for filtering and discovery.

**SLA.** The optional `sla` block defines when data should arrive. It takes two fields: `schedule`, a standard 5-field cron expression that declares the expected delivery cadence, and `lag_hours`, the number of hours the data may lag behind the scheduled time before triggering a failure. When both fields are present, DQX auto-generates a freshness check — no additional configuration required. See [SLA Specification](sla.md) for cron format reference and examples.

**Partitioning.** The optional `metadata` block declares the partitioning columns for the dataset. DQX reads `partitioned_by` to infer which column carries the timestamp used in freshness and completeness checks. When the SLA block references a freshness check and `partitioned_by` is set, DQX selects the first listed column as the timestamp column automatically.

**Columns.** The `columns` section is the heart of the contract. Each entry co-locates four pieces of information that belong together: the column's `type` (one of 12 flexible PyArrow types that accept compatible storage variations — `int` accepts int8 through int64, `float` accepts float32 and float64), its `nullable` flag (defaults to `true` when omitted), its required `description`, and an optional `checks` list. Co-locating schema and checks in a single entry makes the contract self-documenting: a reader sees the column's semantics and its quality requirements in one place. See [Type System](types.md) for the full compatibility matrix.

**Table-level checks.** The top-level `checks` section validates properties of the dataset as a whole. `num_rows` asserts that the row count falls within a specified range. `duplicates` asserts that duplicate rows stay below a threshold. `freshness` asserts that the most recent timestamp column value falls within an acceptable lag window. `completeness` asserts that partition gaps — missing dates or time windows — stay below a specified count. All four checks accept `severity` and standard validators (`min`, `max`, `between`, `equals`, `tolerance`).

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

DQX contracts provide 18 built-in check types across two scopes.

**4 table-level checks** validate the dataset as a whole:
- `num_rows` — asserts total row count
- `duplicates` — asserts count of duplicate rows
- `freshness` — asserts recency of the most recent timestamp value
- `completeness` — asserts absence of partition gaps

**14 column-level checks** validate individual columns. 8 are statistical:
- `cardinality` — distinct value count
- `min` — minimum value
- `max` — maximum value
- `mean` — arithmetic mean
- `sum` — column sum
- `count` — non-null count
- `variance` — statistical variance
- `percentile` — value at a specified percentile

6 are value checks:
- `nulls` — null value count
- `duplicates` — duplicate value count within the column
- `whitelist` — all values belong to an allowed set
- `blacklist` — no values belong to a forbidden set
- `pattern` — all values match a regular expression
- `length` — string or array length

Most checks, table-level or column-level, support validators: `min`, `max`, `between`, `equals`, and `tolerance`. Exceptions are `freshness` (uses `max_age_hours`) and `completeness` (uses `max_gap_count`), which use check-specific implicit parameters instead. See [Check Types Reference](checks.md) for parameter conventions and composition patterns.

---

## Detailed References

- [YAML Contract Structure](schema.md) — Schema structure, type field format, and basic contract examples
- [Type System](types.md) — PyArrow-based type definitions, primitive, temporal, decimal, and complex types
- [SLA Specification](sla.md) — Service level agreements, scheduling, auto-generated checks, and examples
- [Check Types Reference](checks.md) — Overview, parameter conventions, table-level checks, column-level checks, and composition patterns

---

## Design Philosophy

DQX data contracts provide a declarative YAML interface defining schemas (PyArrow-based with 12 flexible types), SLAs (cron-based availability schedules), quality checks (column and table-level), and metadata (documentation and partitioning hints). The design centers on one principle: keep schema and checks co-located so the contract reads as a complete specification of the dataset, not a fragmented collection of rules.

✅ **Declarative** - Express requirements in YAML, not Python code
✅ **Type-Safe** - PyArrow schema validation with complex type support
✅ **Flexible** - Types accept compatible variations (e.g., `int` accepts int8-int64)
✅ **Reusable** - Generated suites run against multiple datasources
✅ **Standard** - Produces standard `AssertionResult` objects
✅ **Extensible** - Custom check types via function registry
✅ **Self-Documenting** - Required descriptions enforce documentation culture
