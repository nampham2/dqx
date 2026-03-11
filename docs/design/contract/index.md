# ODCS Contract Technical Specification

| Field        | Value                              |
|--------------|------------------------------------|
| **Author**   | DQX                                |
| **Created**  | 2026-03-11                         |
| **Version**  | 1.0                                |
| **Status**   | Draft                              |
| **Tags**     | data-contracts, odcs, data-quality |

---

## Overview

> **⚠️ Proposed Design — Not Yet Implemented**
>
> This document describes the planned public API for ODCS contract support in DQX. The `dqx.contract` module and the `Contract` class do **not** yet exist as a public API. The current prototype lives in `src/dqx/contract_old.py` and is not exported from `src/dqx/__init__.py`. Code examples in this document use the **intended future API** and will raise `ImportError` if run against the current codebase. This specification serves as the design target for the upcoming implementation.

A DQX contract is an [ODCS](https://github.com/bitol-io/open-data-contract-standard) (Open Data Contract Standard) YAML document that DQX parses and translates into executable `DecoratedCheck` functions. Each contract is an ODCS-compliant file validated against the ODCS JSON Schema v3.1.0 before any DQX processing occurs. Structural errors — missing required fields, wrong types, invalid enums — are caught at parse time and reported as `ContractValidationError` before any check executes.

DQX adopts ODCS as its contract format for three reasons. First, ODCS is an open standard maintained by a vendor-neutral working group, which means contracts written for DQX are interoperable with other ODCS-aware tools and platforms without conversion. Second, the ODCS JSON Schema provides machine-readable validation — DQX vendors the schema (`odcs-json-schema-v3.1.0.json`) in `src/dqx/_schemas/` and validates every incoming contract before processing. Third, using ODCS allows DQX to avoid designing and maintaining a proprietary contract format, reducing the surface area users must learn.

DQX participates in ODCS as a `type: custom, engine: dqx` quality rule engine. Standard ODCS library metrics (`nullValues`, `missingValues`, `invalidValues`, `duplicateValues`, `rowCount`) are also supported and mapped directly to `MetricProvider` calls. Quality rules targeting other engines — `type: sql`, `type: custom` with an engine other than `dqx` — emit a `ContractWarning` and are skipped.

Once the public module is implemented, `Contract` will be imported from `dqx.contract`:

```python
# Proposed API — not yet available; Contract is not currently exported from dqx
from dqx.contract import Contract
```

One `Contract` instance is created per `schema[]` object in the ODCS document. A single ODCS file with two schema objects produces `Contract.from_odcs(path)` returning a `list[Contract]` of length two. Each `Contract` exposes a `to_checks()` method returning a `list[DecoratedCheck]` that can be composed freely with hand-coded checks inside a `VerificationSuite`.

---

## Architecture

### Processing Pipeline

```text
ODCS YAML
    ↓ Contract.from_odcs(path)  ← validated against ODCS JSON Schema v3.1.0
list[Contract]                  ← one Contract per schema[] object
    ↓ contract.to_checks()
list[DecoratedCheck]
    ↓ VerificationSuite(checks=contract.to_checks() + [...], db=db, name=...)
VerificationSuite
    ↓ suite.run([datasource], result_key)
    ↓ suite.collect_results()
list[AssertionResult]
```

### Key Design Principles

**ODCS JSON Schema validation first.** `Contract.from_odcs()` validates the raw YAML document against `odcs-json-schema-v3.1.0.json` (vendored in `src/dqx/_schemas/`) before any DQX-specific processing. This means structural errors — absent required fields, wrong value types, enum violations — surface as `ContractValidationError` with a message derived from the JSON Schema validator, before DQX touches the quality rules or schema properties. Only documents that pass JSON Schema validation proceed to DQX parsing.

**One `Contract` per schema object.** ODCS contracts may declare multiple tables in a single file via the `schema[]` list. DQX maps each schema object to an independent `Contract` instance. `Contract.from_odcs(path)` always returns a `list[Contract]`, even when the ODCS file contains only one schema object.

**One `DecoratedCheck` per `Contract`.** All quality rules attached to a schema object — table-level and property-level — are consolidated into a single `DecoratedCheck` node named `"Contract: {name or dataset}"`, where `name` is the top-level ODCS contract `name` field and `dataset` is the schema object's `physicalName` (preferred) or `name`. Each individual rule becomes an assertion child of that check node. This keeps the graph structure flat and predictable.

**Composable.** `contract.to_checks()` returns a plain `list[DecoratedCheck]`. Users concatenate this list with any hand-coded checks before passing the combined list to `VerificationSuite`. Contract-generated checks and custom checks run together in the same suite and produce `AssertionResult` objects through the same `suite.collect_results()` call.

---

## Contract Structure

### Annotated ODCS Contract

The YAML below shows every field DQX consumes, with inline comments explaining DQX's treatment of each field.

```yaml
apiVersion: v3.1.0           # required by ODCS JSON Schema
kind: DataContract            # required: must be "DataContract"
id: a1b2c3d4-e5f6-7890-abcd-ef1234567890  # required by ODCS JSON Schema
name: orders_contract         # used: check node name → "Contract: orders_contract"
version: 1.0.0                # required by ODCS JSON Schema
status: active                # required by ODCS JSON Schema
domain: commerce              # stored: not executed
dataProduct: order-mgmt       # stored: not executed
tags: [revenue]               # stored: not executed

description:                  # stored: not executed
  purpose: "Daily order records for revenue reporting"
  limitations: "T-1 data only"

schema:                       # used: each object → one Contract
  - name: orders              # used: contract.dataset (logical name)
    physicalName: orders_tbl  # used: contract.dataset (preferred over name if present)
    quality: [...]            # used: table-level quality rules
    properties: [...]         # used: column definitions + column-level quality rules

slaProperties:                # used: latency property only
  - property: latency         # used: → freshness check (raises NotImplementedError in to_checks())
    value: 24
    unit: h
  - property: retention       # stored: not executed
    value: 3
    unit: y

# Ignored entirely by DQX:
# servers, team, roles, support, price,
# authoritativeDefinitions, contractCreatedTs
```

### Required ODCS Fields

The ODCS JSON Schema mandates five top-level fields. DQX raises `ContractValidationError` when any of them is absent or violates its schema constraint.

| ODCS Field    | Type     | Constraint                                   | `ContractValidationError` message when absent                     |
|---------------|----------|----------------------------------------------|-------------------------------------------------------------------|
| `version`     | `string` | Non-empty string                             | `ContractValidationError: 'version' is a required property`       |
| `apiVersion`  | `string` | Must match `v3.1.0`                          | `ContractValidationError: 'apiVersion' is a required property`    |
| `kind`        | `string` | Must equal `"DataContract"`                  | `ContractValidationError: 'kind' is a required property`          |
| `id`          | `string` | UUID or non-empty string                     | `ContractValidationError: 'id' is a required property`            |
| `status`      | `string` | Enum: `active`, `deprecated`, `draft`, etc. | `ContractValidationError: 'status' is a required property`        |

---

## Error Surface

DQX exposes three distinct error types across the contract lifecycle. Understanding which error type is raised — and when — helps teams write correct contracts and handle failures gracefully.

### `ContractValidationError`

Raised during `Contract.from_odcs()`. This error means the contract document itself is invalid and cannot be processed. Execution does not begin.

**Trigger conditions:**

- The ODCS JSON Schema validator rejects the document (any required field missing, wrong value type, enum violation for `kind`, `status`, or `apiVersion`).
- An unknown `logicalType` value is encountered on a property (e.g., `logicalType: uuid` is not in the ODCS-defined enumeration).
- A quality rule contains a malformed operator value (e.g., `mustBeBetween` is provided a scalar instead of a two-element list `[a, b]`).
- A required ODCS field (`version`, `apiVersion`, `kind`, `id`, `status`) is absent.

**Example messages:**

```text
ContractValidationError: 'kind' is a required property
ContractValidationError: 'status' is a required property
ContractValidationError: Property 'order_id': unknown logicalType 'uuid'
ContractValidationError: Quality rule 'amount_range': mustBeBetween must be a list of [lower, upper], got: 100
```

### `SchemaValidationError`

Raised during `to_checks()` at runtime, when the check function executes against a live datasource. This error means the actual data schema does not conform to what the contract declares.

**Trigger conditions:**

- A column's actual PyArrow type does not match the declared `logicalType`. For example, a column declared as `logicalType: integer` is backed by `pa.string()` in the datasource.
- A column declared with `required: true` contains one or more null values.

**Example messages:**

```text
SchemaValidationError: Property 'order_id' type mismatch:
  expected logicalType 'integer' (int8–int64, uint8–uint64), got pa.string()

SchemaValidationError: Property 'order_id' is required (required: true)
  but contains null values
```

### `ContractWarning`

Emitted during `Contract.from_odcs()` as a non-fatal Python warning (via `warnings.warn`). Processing continues after each warning; the offending rule is skipped.

**Trigger conditions and example messages:**

- A quality rule has `type: sql` — DQX cannot execute raw SQL rules inline.

  ```text
  ContractWarning: Quality rule 'iban_check': type 'sql' is not executable in DQX; skipping
  ```

- A quality rule has `type: custom` with an engine other than `dqx`.

  ```text
  ContractWarning: Quality rule 'soda_check': type 'custom' with engine 'soda' is not executable in DQX; skipping
  ```

- A `metric: invalidValues` quality rule specifies `arguments.pattern` (regex-based invalid value detection is not yet implemented).

  ```text
  ContractWarning: Quality rule 'iban_check': metric 'invalidValues' with pattern argument is not executable in DQX; skipping
  ```

- A `metric: missingValues` quality rule specifies `arguments.missingValues` (custom sentinel value lists are not supported; use `type: custom, engine: dqx` with `check: blacklist` instead).

  ```text
  ContractWarning: Quality rule 'sentinel_check': metric 'missingValues' with missingValues argument is not executable in DQX; skipping
  ```

- An `slaProperties` latency entry cannot resolve a timestamp column (no `element:` field and no `partitioned: true` property with `partitionKeyPosition: 1`).

  ```text
  ContractWarning: SLA latency rule: no timestamp column could be resolved; freshness check skipped
  ```

---

## Usage

> **⚠️ Proposed API — Not Yet Implemented**
>
> The examples below use `from dqx.contract import Contract`, which is the **intended future API**. This module does not currently exist. Running these examples against the current codebase will raise `ImportError`. They are provided as the design target for the upcoming implementation.

### Single-Table Contract

The most common case: an ODCS file with one schema object, producing one `Contract`.

```python
from pathlib import Path

from dqx.contract import Contract  # Proposed API — not yet available
from dqx.api import VerificationSuite

# Parse the ODCS file — validated against JSON Schema, returns list[Contract]
contracts = Contract.from_odcs(Path("orders.odcs.yaml"))

# Build the suite, composing contract checks with a hand-coded check
suite = VerificationSuite(
    name="orders-suite",
    checks=contracts[0].to_checks() + [my_custom_check],
    db=db,
)

# Execute and collect results
suite.run([datasource], result_key)
results = suite.collect_results()
```

### Multi-Table Contract

An ODCS file with multiple schema objects — one `Contract` per schema, one suite per contract.

```python
from pathlib import Path

from dqx.contract import Contract  # Proposed API — not yet available
from dqx.api import VerificationSuite

contracts = Contract.from_odcs(Path("commerce.odcs.yaml"))

for contract in contracts:
    suite = VerificationSuite(
        name=f"{contract.dataset}-suite",
        checks=contract.to_checks(),
        db=db,
    )
    suite.run([datasource], result_key)
```

---

## Quality Rules

### 6.1 Rule Types Overview

ODCS quality rules carry a `type` field that DQX uses to decide whether to execute the rule, skip it, or store it for documentation purposes.

| ODCS `type`                         | DQX treatment                                                               |
|-------------------------------------|-----------------------------------------------------------------------------|
| `library` (or implicit when `metric:` present) | Executed — mapped to `MetricProvider` calls                    |
| `text`                              | Documentation only — stored in `Contract.text_rules`, no assertion generated |
| `sql`                               | Skipped — `ContractWarning` emitted                                         |
| `custom` with `engine: dqx`         | Executed — `implementation:` block parsed as DQX check                      |
| `custom` with any other engine      | Skipped — `ContractWarning` emitted                                         |

### 6.2 Library Metrics

ODCS library metrics (`metric:` field present, `type: library` or omitted) map to `MetricProvider` method calls. The table below describes each mapping and notes arguments that cannot be satisfied by the current `MetricProvider` API.

| ODCS `metric`      | Scope      | `MetricProvider` call                              | `unit: percent` behaviour              | Unsupported argument                                                              |
|--------------------|------------|----------------------------------------------------|----------------------------------------|-----------------------------------------------------------------------------------|
| `nullValues`       | `property` | `mp.null_count(col)`                               | `÷ mp.num_rows()`                      | —                                                                                 |
| `missingValues`    | `property` | `mp.null_count(col)`                               | `÷ mp.num_rows()`                      | `arguments.missingValues` sentinel list → `ContractWarning`                       |
| `invalidValues`    | `property` | `mp.num_rows() - mp.count_values(col, validValues)` | `÷ mp.num_rows()`                     | `arguments.pattern` → `ContractWarning`                                           |
| `duplicateValues`  | `property` | `mp.duplicate_count([col])`                        | `÷ mp.num_rows()`                      | —                                                                                 |
| `rowCount`         | `schema`   | `mp.num_rows()`                                    | —                                      | —                                                                                 |
| `duplicateValues` (with `arguments.properties`) | `schema` | `mp.duplicate_count(properties)` | `÷ mp.num_rows()`         | —                                                                                 |

#### `nullValues` Example

```yaml
quality:
  - name: order_id_no_nulls
    type: library
    metric: nullValues
    mustBe: 0
    severity: error
```

#### `missingValues` Example

```yaml
quality:
  - name: customer_id_no_missing
    type: library
    metric: missingValues
    mustBe: 0
    severity: error
```

#### `invalidValues` Example

```yaml
quality:
  - name: status_valid_values
    type: library
    metric: invalidValues
    mustBe: 0
    severity: error
    validValues: [pending, processing, shipped, delivered, cancelled]
```

#### `duplicateValues` (property-level) Example

```yaml
quality:
  - name: order_id_unique
    type: library
    metric: duplicateValues
    mustBe: 0
    severity: error
```

#### `rowCount` Example

```yaml
quality:
  - name: daily_row_count
    type: library
    metric: rowCount
    mustBeGreaterOrEqualTo: 100
    severity: warning
```

#### `duplicateValues` (schema-level with `arguments.properties`) Example

```yaml
quality:
  - name: order_composite_unique
    type: library
    metric: duplicateValues
    mustBe: 0
    severity: error
    arguments:
      properties: [order_id, order_date]
```

### 6.3 Operators

ODCS quality rules express their acceptance condition through operator fields. DQX maps each ODCS operator to the corresponding `AssertionReady` method.

| ODCS operator              | `AssertionReady` method    | Semantics                          |
|----------------------------|----------------------------|------------------------------------|
| `mustBe: v`                | `.is_eq(v)`                | value == v (within tolerance 1e-9) |
| `mustNotBe: v`             | `.is_neq(v)`               | value ≠ v                          |
| `mustBeGreaterThan: v`     | `.is_gt(v)`                | value > v                          |
| `mustBeGreaterOrEqualTo: v`| `.is_geq(v)`               | value ≥ v                          |
| `mustBeLessThan: v`        | `.is_lt(v)`                | value < v                          |
| `mustBeLessOrEqualTo: v`   | `.is_leq(v)`               | value ≤ v                          |
| `mustBeBetween: [a, b]`    | `.is_between(a, b)`        | a ≤ value ≤ b                      |
| `mustNotBeBetween: [a, b]` | `.is_not_between(a, b)`    | value < a or value > b             |
| *(none)*                   | `.noop()`                  | metric recorded, never fails       |

**Note on tolerance:** `mustBe`, `mustBeBetween`, and `mustNotBeBetween` use `tolerance=1e-9` by default. This prevents spurious failures due to floating-point representation differences. For exact integer comparisons the tolerance is effectively zero.

### 6.4 Severity Mapping

ODCS quality rules carry an optional `severity` field using string labels. DQX maps these labels to its internal `SeverityLevel` type.

| ODCS `severity` | DQX `SeverityLevel` |
|-----------------|---------------------|
| `"error"`       | `"P0"`              |
| `"warning"`     | `"P1"`              |
| omitted         | `"P1"`              |

When `severity` is absent from a quality rule, DQX defaults to `"P1"` (important but non-blocking).

### 6.5 DQX Custom Quality Rules (`type: custom, engine: dqx`)

ODCS library metrics cover five aggregate metrics. DQX custom quality rules expose the full DQX check vocabulary — all `MetricProvider` methods — through ODCS's extensible `type: custom` mechanism. When a rule carries `type: custom` and `engine: dqx`, DQX parses the `implementation:` field as a YAML string. The `check:` key inside that YAML string is the discriminator: it selects which `MetricProvider` method to call. ODCS operators (`mustBe`, `mustBeBetween`, etc.) from the parent rule provide the threshold.

#### Column-Level Check Vocabulary

Column-level checks require a `column:` field inside the `implementation:` YAML string.

| `check:`       | Description                                     | Extra required params                    | Extra optional params          |
|----------------|-------------------------------------------------|------------------------------------------|--------------------------------|
| `missing`      | null count in column                            | —                                        | `return: count\|pct`           |
| `duplicates`   | duplicate value count in column                 | —                                        | `return: count\|pct`           |
| `whitelist`    | count of values NOT in allowed set              | `values: [...]`                          | `return: count\|pct`           |
| `blacklist`    | count of values IN forbidden set                | `values: [...]`                          | `return: count\|pct`           |
| `cardinality`  | distinct value count                            | —                                        | —                              |
| `min`          | column minimum value                            | —                                        | —                              |
| `max`          | column maximum value                            | —                                        | —                              |
| `mean`         | arithmetic mean of column values                | —                                        | —                              |
| `sum`          | sum of column values                            | —                                        | —                              |
| `count`        | non-null value count                            | —                                        | —                              |
| `variance`     | statistical variance of column values           | —                                        | —                              |
| `stddev`       | standard deviation of column values             | —                                        | —                              |
| `percentile`   | value at specified percentile                   | `percentile: 0.0–1.0`                    | —                              |
| `min_length`   | minimum string/list/map length                  | `column_type: string\|list\|map`         | —                              |
| `max_length`   | maximum string/list/map length                  | `column_type: string\|list\|map`         | —                              |
| `avg_length`   | average string/list/map length                  | `column_type: string\|list\|map`         | —                              |

#### Table-Level Check Vocabulary

Table-level checks omit the `column:` field inside `implementation:`.

| `check:`    | Description              |
|-------------|--------------------------|
| `num_rows`  | total row count of table |

#### Example 1: `check: missing` (null rate as percentage)

```yaml
quality:
  - name: total_amount_not_null
    type: custom
    engine: dqx
    severity: error
    mustBe: 0
    implementation: |
      check: missing
      column: total_amount
      return: pct
```

#### Example 2: `check: whitelist` (valid values check)

```yaml
quality:
  - name: status_valid
    type: custom
    engine: dqx
    severity: error
    mustBe: 0
    implementation: |
      check: whitelist
      column: status
      values:
        - pending
        - processing
        - shipped
        - delivered
        - cancelled
```

#### Example 3: `check: percentile` (p95 response time)

```yaml
quality:
  - name: p95_latency_acceptable
    type: custom
    engine: dqx
    severity: warning
    mustBeLessOrEqualTo: 500
    implementation: |
      check: percentile
      column: response_time_ms
      percentile: 0.95
```

#### Example 4: `check: avg_length` (average string length)

```yaml
quality:
  - name: product_name_length_reasonable
    type: custom
    engine: dqx
    severity: warning
    mustBeBetween: [5, 100]
    implementation: |
      check: avg_length
      column: product_name
      column_type: string
```

#### Example 5: `check: num_rows` (table-level, no `column:`)

```yaml
quality:
  - name: daily_volume_check
    type: custom
    engine: dqx
    severity: warning
    mustBeGreaterOrEqualTo: 100
    implementation: |
      check: num_rows
```

### 6.6 Skipped Rules

Some ODCS quality rule patterns cannot be executed by DQX. For each pattern, DQX emits a `ContractWarning` and skips the rule. The table below shows the YAML pattern, the warning message, and the recommended alternative.

#### `type: sql`

DQX does not execute inline SQL quality rules.

```yaml
# Triggers ContractWarning — skipped
quality:
  - name: iban_check
    type: sql
    query: "SELECT COUNT(*) FROM orders WHERE iban NOT REGEXP '^[A-Z]{2}[0-9]{2}'"
    mustBe: 0
    severity: error
```

```text
ContractWarning: Quality rule 'iban_check': type 'sql' is not executable in DQX; skipping
```

**Use instead:** `type: custom, engine: dqx` with an appropriate `check:` such as `check: blacklist` or `check: whitelist` for set-based validation. For regex matching, a `check: pattern` implementation is deferred (see below).

#### `type: custom` with a non-DQX engine

DQX cannot execute rules intended for other engines such as Soda, Great Expectations, or Monte Carlo.

```yaml
# Triggers ContractWarning — skipped
quality:
  - name: soda_check
    type: custom
    engine: soda
    severity: warning
    implementation: |
      missing_count(order_id) = 0
```

```text
ContractWarning: Quality rule 'soda_check': type 'custom' with engine 'soda' is not executable in DQX; skipping
```

**Use instead:** Configure the Soda engine separately and keep DQX rules as `engine: dqx`.

#### `metric: invalidValues` with `arguments.pattern`

Regex-based invalid value detection is not yet implemented in `MetricProvider`.

```yaml
# Triggers ContractWarning — skipped
quality:
  - name: iban_format_check
    type: library
    metric: invalidValues
    mustBe: 0
    severity: warning
    arguments:
      pattern: "^[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}$"
```

```text
ContractWarning: Quality rule 'iban_format_check': metric 'invalidValues' with pattern argument is not executable in DQX; skipping
```

Pattern-based validation (`check: pattern`) is a planned future capability pending implementation of `MetricProvider.pattern_match()`. It is not currently available in DQX — there is no working workaround for `metric: invalidValues` with `arguments.pattern` at this time.

#### `metric: missingValues` with `arguments.missingValues` sentinel list

Custom sentinel-based missing value detection (treating specific non-null values as missing) is not supported in `MetricProvider.null_count()`.

```yaml
# Triggers ContractWarning — skipped
quality:
  - name: sentinel_missing_check
    type: library
    metric: missingValues
    mustBe: 0
    severity: warning
    arguments:
      missingValues: ["N/A", "n/a", "NULL", ""]
```

```text
ContractWarning: Quality rule 'sentinel_missing_check': metric 'missingValues' with missingValues argument is not executable in DQX; skipping
```

**Partial workaround:** If you only need to reject sentinel literals, use `type: custom, engine: dqx` with `check: blacklist` and the sentinel values listed under `values:`. Note that `check: blacklist` does **not** count actual `NULL` values — it only matches the explicit string literals in `values:`. To preserve the full semantics of `metric: missingValues` (sentinel literals **and** real nulls), pair the `blacklist` rule with a separate `nullValues` library metric or a `check: missing` custom rule.

```yaml
# Recommended replacement
quality:
  - name: sentinel_missing_check
    type: custom
    engine: dqx
    severity: warning
    mustBe: 0
    implementation: |
      check: blacklist
      column: status
      values:
        - "N/A"
        - "n/a"
        - "NULL"
        - ""
```

### 6.7 Complete Contract Example

The contract below is a full, runnable ODCS YAML showing all four rule types — library, text, sql (skipped), and custom DQX — in a single `orders` table contract. Each rule carries an inline comment showing DQX treatment.

```yaml
apiVersion: v3.1.0
kind: DataContract
id: f7a3b2c1-d4e5-6789-abcd-0123456789ef
name: orders_contract
version: 2.0.0
status: active
domain: commerce
dataProduct: order-management

description:
  purpose: "Daily order records for the commerce domain"
  limitations: "T-1 data; does not include cancelled test orders"

slaProperties:
  - property: latency             # used: DQX resolves freshness check (NotImplementedError in to_checks())
    value: 24
    unit: h
    element: orders_tbl.order_date
  - property: retention           # stored: Contract.sla_metadata; not executed
    value: 3
    unit: y

schema:
  - name: orders
    physicalName: orders_tbl      # preferred over name for contract.dataset

    quality:
      # library rule — executed: mp.num_rows() mustBeGreaterOrEqualTo 100
      - name: daily_row_count
        type: library
        metric: rowCount
        mustBeGreaterOrEqualTo: 100
        severity: warning

      # library rule — executed: mp.duplicate_count([order_id, order_date]) mustBe 0
      - name: composite_key_unique
        type: library
        metric: duplicateValues
        mustBe: 0
        severity: error
        arguments:
          properties: [order_id, order_date]

      # custom dqx rule — executed: mp.num_rows() mustBeGreaterOrEqualTo 100 (table-level)
      - name: volume_floor
        type: custom
        engine: dqx
        severity: warning
        mustBeGreaterOrEqualTo: 100
        implementation: |
          check: num_rows

      # sql rule — skipped, ContractWarning emitted
      - name: fraud_sql_check
        type: sql
        query: "SELECT COUNT(*) FROM orders_tbl WHERE total_amount > 99999"
        mustBe: 0
        severity: warning

    properties:
      - name: order_id
        logicalType: integer
        required: true            # SchemaValidationError if nulls present at runtime

        quality:
          # library rule — executed: mp.null_count(order_id) mustBe 0
          - name: order_id_no_nulls
            type: library
            metric: nullValues
            mustBe: 0
            severity: error

          # text rule — stored as documentation; no assertion generated
          - name: order_id_description
            type: text
            description: "order_id is the primary key; generated by the order-service integer ID generator"

          # custom dqx rule — executed: mp.min(order_id) mustBeGreaterOrEqualTo 1
          - name: order_id_positive
            type: custom
            engine: dqx
            severity: error
            mustBeGreaterOrEqualTo: 1
            implementation: |
              check: min
              column: order_id

      - name: status
        logicalType: string
        required: true

        quality:
          # library rule — executed: mp.num_rows() - mp.count_values(status, validValues) mustBe 0
          - name: status_valid_values
            type: library
            metric: invalidValues
            mustBe: 0
            severity: error
            validValues: [pending, processing, shipped, delivered, cancelled]

          # custom dqx rule — executed: mp.cardinality(status) mustBeLessOrEqualTo 10
          - name: status_low_cardinality
            type: custom
            engine: dqx
            severity: warning
            mustBeLessOrEqualTo: 10
            implementation: |
              check: cardinality
              column: status

      - name: total_amount
        logicalType: number
        required: true

        quality:
          # custom dqx rule — executed: mp.mean(total_amount) mustBeBetween [10.0, 500.0]
          - name: average_order_value_reasonable
            type: custom
            engine: dqx
            severity: warning
            mustBeBetween: [10.0, 500.0]
            implementation: |
              check: mean
              column: total_amount

          # custom dqx rule — executed: mp.null_count(total_amount) / mp.num_rows() mustBe 0
          - name: total_amount_complete
            type: custom
            engine: dqx
            severity: error
            mustBe: 0
            implementation: |
              check: missing
              column: total_amount
              return: pct

      - name: product_name
        logicalType: string
        required: false

        quality:
          # custom dqx rule — executed: mp.avg_length(product_name) mustBeBetween [3, 120]
          - name: product_name_length_sanity
            type: custom
            engine: dqx
            severity: warning
            mustBeBetween: [3, 120]
            implementation: |
              check: avg_length
              column: product_name
              column_type: string

      - name: order_date
        logicalType: date
        required: true
        partitioned: true         # used by SLA latency resolution
        partitionKeyPosition: 1   # used by SLA latency resolution → timestamp_column = order_date
```

---

## Type System

### 7.1 `logicalType` → PyArrow Compatibility Matrix

ODCS properties declare their semantic type via the `logicalType` field. DQX validates at runtime that the column's actual PyArrow type is compatible with the declared `logicalType`.

| ODCS `logicalType` | PyArrow types accepted                                                                                      | Notes                                                                  |
|--------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `string`           | `pa.string()`, `pa.utf8()`, `pa.large_string()`, `pa.large_utf8()`                                         | All four are equivalent UTF-8 string representations                   |
| `integer`          | `pa.int8()`, `pa.int16()`, `pa.int32()`, `pa.int64()`, `pa.uint8()`, `pa.uint16()`, `pa.uint32()`, `pa.uint64()` | Any width, signed or unsigned                                     |
| `number`           | `pa.float32()`, `pa.float64()`, `pa.decimal128(p, s)`, `pa.decimal256(p, s)`                               | Float and decimal both accepted; float16 is not accepted               |
| `boolean`          | `pa.bool_()`                                                                                                | Exact match only                                                       |
| `date`             | `pa.date32()`, `pa.date64()`                                                                                | Both day-based encodings accepted                                      |
| `timestamp`        | `pa.timestamp(unit, tz)` for any unit (`s`, `ms`, `us`, `ns`) and any timezone                             | Narrowed by `logicalTypeOptions` — see section 7.2                     |
| `time`             | `pa.time32("s")`, `pa.time32("ms")`, `pa.time64("us")`, `pa.time64("ns")`                                  | All four time sub-second units accepted                                |
| `array`            | `pa.list_(T)`, `pa.large_list(T)`                                                                           | Element type validated recursively via `items.logicalType`             |
| `object`           | `pa.struct([...])`                                                                                          | Field types validated recursively via nested `properties`              |

### 7.2 Timestamp and `logicalTypeOptions`

ODCS allows `logicalType: timestamp` to be further narrowed by a `logicalTypeOptions` block. DQX interprets three forms:

**Form 1: No `logicalTypeOptions`** — any `pa.timestamp()` accepted regardless of unit or timezone.

```yaml
- name: created_at
  logicalType: timestamp
  required: true
  # No logicalTypeOptions → accepts pa.timestamp('us', tz=None),
  #   pa.timestamp('ms', tz='UTC'), pa.timestamp('s', tz='America/New_York'), etc.
```

**Form 2: `logicalTypeOptions` with timezone enforcement** — DQX enforces that the column's timezone matches `defaultTimezone`.

```yaml
- name: created_at_utc
  logicalType: timestamp
  required: true
  logicalTypeOptions:
    timezone: true
    defaultTimezone: "UTC"
  # DQX enforces tz == "UTC"; pa.timestamp('us', tz='America/New_York') raises SchemaValidationError
```

**Form 3: `physicalType: TIMESTAMP WITH TIME ZONE`** — stored as metadata only; no additional validation beyond `logicalType: timestamp`.

```yaml
- name: created_at
  logicalType: timestamp
  physicalType: TIMESTAMP WITH TIME ZONE
  required: true
  # physicalType stored as metadata; DQX validates logicalType only
```

### 7.3 The `required` Field

The `required` field on an ODCS property controls null validation at runtime.

- `required: true` — the column must not contain null values. DQX validates this at runtime when `to_checks()` executes. If nulls are found, `SchemaValidationError` is raised.
- `required: false` (default when omitted) — the column is nullable. Null values are allowed and do not trigger any error.

**Note:** The ODCS `required` field is the **inverse** of the legacy DQX contract `nullable` field. In the old DQX contract format, `nullable: false` means no nulls allowed. In ODCS, `required: true` means no nulls allowed. When migrating contracts from the old format to ODCS, flip the boolean: `nullable: false` → `required: true`, `nullable: true` → `required: false`.

### 7.4 `physicalType`

The `physicalType` field is stored as metadata on the `Contract` and its properties. DQX does not use `physicalType` for any validation — `logicalType` is the sole validation basis. `physicalType` communicates the underlying storage format to readers and downstream tooling without affecting DQX behavior.

Representative examples of `physicalType` values:

- `VARCHAR(18)` — bounded string column in a SQL warehouse
- `DOUBLE` — 64-bit IEEE 754 float in Spark
- `DECIMAL(18,2)` — fixed-point decimal in BigQuery
- `TIMESTAMP WITH TIME ZONE` — timezone-aware timestamp in PostgreSQL

### 7.5 Nested Types

**`array`:** When a property has `logicalType: array`, DQX validates the element type recursively using the `items.logicalType` field. For example, an array with `items.logicalType: integer` must be backed by `pa.list_(pa.int64())` or `pa.large_list(pa.int64())` (or any integer sub-type). Nullability (`required`) is enforced only at the top-level property — nested element nullability is not validated.

**`object`:** When a property has `logicalType: object`, DQX validates each field declared in `properties[]` against the column's `pa.struct([...])` fields using each field's `logicalType`. Struct fields that are present in the schema but absent from `properties[]` are ignored. Fields declared in `properties[]` but absent from the struct raise `SchemaValidationError`. Nested field nullability is not validated.

#### YAML Example: Nested Object Type

```yaml
- name: address
  logicalType: object
  required: true
  properties:
    - name: street
      logicalType: string
      required: true     # required on nested field stored for documentation; not enforced at runtime
    - name: city
      logicalType: string
      required: true
    - name: zip_code
      logicalType: string
      required: false
    - name: country_code
      logicalType: string
      required: true
```

### 7.6 `SchemaValidationError` Format

DQX raises `SchemaValidationError` for two conditions, with messages in the following format:

**Type mismatch:**

```text
SchemaValidationError: Property 'order_id' type mismatch:
  expected logicalType 'integer' (int8–int64, uint8–uint64), got pa.string()
```

**Required column contains nulls:**

```text
SchemaValidationError: Property 'email' is required (required: true)
  but contains null values
```

Both messages include the property name, the declared `logicalType` with its accepted PyArrow types, and the actual PyArrow type observed. This allows engineers to locate the source of the mismatch without manual schema inspection.

---

## SLA

### 8.1 Overview

`slaProperties` is a list of key/value SLA definitions in ODCS format. DQX processes only entries with `property: latency` (synonym: `ly`). All other properties — `retention`, `availability`, `generalAvailability`, `endOfSupport`, `endOfLife`, `frequency`, `timeOfAvailability`, `errorRate` — are parsed and stored in `Contract.sla_metadata` as a dictionary, but generate no assertions. They are available to downstream tooling through `Contract.sla_metadata` without affecting check execution.

### 8.2 Latency Resolution

When DQX encounters a `property: latency` entry in `slaProperties`, it resolves two values: `max_age_hours` (computed from `value` and `unit`) and `timestamp_column` (resolved from the schema). The resolution algorithm proceeds as follows:

1. **Compute `max_age_hours`:** Multiply `value` by the unit conversion factor from the table in section 8.3. For example, `value: 24, unit: h` → `max_age_hours = 24.0`.

2. **Resolve `timestamp_column` from `element:`:** If the latency entry carries an `element:` field in `"table.column"` notation (e.g., `element: "orders_tbl.order_date"`), the timestamp column is the part after the last `.` — `order_date` in this example.

3. **Resolve `timestamp_column` from schema partitioning:** If `element` is absent, DQX searches the schema `properties[]` for the first property that has both `partitioned: true` and `partitionKeyPosition: 1`. The name of that property becomes the timestamp column.

4. **Emit `ContractWarning` and skip:** If neither `element` nor a matching partitioned property is found, DQX emits a `ContractWarning` and does not generate a freshness check for this latency entry. Processing of other rules continues normally.

### 8.3 Unit Conversion Table

| `unit` values                    | Conversion to hours | Example                                         |
|----------------------------------|---------------------|-------------------------------------------------|
| `h`, `hr`, `hour`, `hours`       | × 1                 | `value: 24, unit: h` → 24.0 hours              |
| `d`, `day`, `days`               | × 24                | `value: 2, unit: d` → 48.0 hours               |
| `y`, `yr`, `year`, `years`       | × 8760              | `value: 1, unit: y` → 8760.0 hours             |

### 8.4 Auto-Generated Freshness Check

DQX resolves `SlaLatency.max_age_hours` and `SlaLatency.timestamp_column` at parse time and stores them on the `Contract` instance. These resolved values define the freshness check DQX **intends** to construct once `FreshnessCheck.to_dqx()` / `MetricProvider.freshness()` are implemented. This construction is a planned feature — see Section 8.5 for the current `NotImplementedError` status, which means latency-bearing contracts are not yet runnable via `to_checks()`.

```yaml
# For: property: latency, value: 24, unit: h, element: orders_tbl.order_date
# DQX internally generates:
#   check name: "SLA: Freshness check"
#   max_age_hours: 24.0
#   timestamp_column: order_date
#   severity: P0
```

The generated check name is always `"SLA: Freshness check"`. The severity is always `"P0"` — latency SLA violations are treated as critical failures.

### 8.5 Current Status

`to_checks()` raises `NotImplementedError` for the SLA-generated freshness check because `FreshnessCheck.to_dqx()` is not yet implemented. The contract fully parses and validates `slaProperties` — `SlaLatency.max_age_hours` and `SlaLatency.timestamp_column` are resolved correctly and are accessible on the `Contract` instance — but execution is deferred pending `MetricProvider.freshness()` implementation.

Teams that need freshness validation today must implement it **outside the contract pipeline**. A contract that includes `slaProperties` with `property: latency` is not yet executable via `contract.to_checks()` — calling it on such a contract raises `NotImplementedError`. There is no supported end-to-end freshness path until `MetricProvider.freshness()` is implemented.

### 8.6 Other SLA Properties

| ODCS `property`       | DQX treatment                                           |
|-----------------------|---------------------------------------------------------|
| `latency` (synonym: `ly`) | Parsed → freshness check (raises `NotImplementedError` in `to_checks()`) |
| `retention`           | Stored in `Contract.sla_metadata`                       |
| `availability`        | Stored in `Contract.sla_metadata`                       |
| `generalAvailability` | Stored in `Contract.sla_metadata`                       |
| `endOfSupport`        | Stored in `Contract.sla_metadata`                       |
| `endOfLife`           | Stored in `Contract.sla_metadata`                       |
| `frequency`           | Stored in `Contract.sla_metadata`                       |
| `timeOfAvailability`  | Stored in `Contract.sla_metadata`                       |
| `errorRate`           | Stored in `Contract.sla_metadata`                       |

### 8.7 Complete SLA Example

The block below shows a `slaProperties` section with a `latency` entry and three additional properties, with inline comments showing DQX treatment of each entry.

```yaml
slaProperties:
  # Parsed successfully; execution is not supported yet because freshness still
  # raises NotImplementedError in contract.to_checks() — MetricProvider.freshness()
  # is not yet implemented:
  # → max_age_hours = 24.0 (24 × 1 hour)
  # → timestamp_column = order_date (resolved from element: "orders_tbl.order_date")
  # → intended generated check: "SLA: Freshness check", severity P0
  - property: latency
    value: 24
    unit: h
    element: orders_tbl.order_date

  # Stored in Contract.sla_metadata["retention"] = {"value": 3, "unit": "y"}; not executed
  - property: retention
    value: 3
    unit: y
    description: "Order data retained for 3 years per regulatory requirement"

  # Stored in Contract.sla_metadata["availability"] = {"value": 99.9, "unit": "percent"}; not executed
  - property: availability
    value: 99.9
    unit: percent
    description: "Pipeline must be available 99.9% of the time"

  # Stored in Contract.sla_metadata["frequency"] = {"value": 1, "unit": "d"}; not executed
  - property: frequency
    value: 1
    unit: d
    description: "Data refreshed once per day"
```
