# Check Types Reference

> Part of the [Data Contracts Technical Specification](index.md).

---

## Overview

Data contract checks validate the quality of a single dataset. Each check computes a metric — a row count, a null percentage, a column mean — and compares it against a declared acceptance condition. Checks that exceed their acceptance condition produce failures at the declared severity level, giving data teams a structured signal to act on.

Data contracts support two categories of checks:

1. **Table-Level Checks**: Validate table-wide properties (row counts, freshness, duplicates, partitions)
2. **Column-Level Checks**: Validate individual column values (nullability, uniqueness, ranges, patterns, distributions)

**Naming Conventions:**

- Check names should be descriptive business statements (e.g., "Order ID is unique")
- Severity levels: `P0` (critical), `P1` (important), `P2` (nice-to-have), `P3` (informational)
- All checks support optional `tags` parameter for categorization
- Check type names in YAML contracts use `snake_case` (e.g., `num_rows`, `duplicates`, `nulls`). These `snake_case` names are the specification surface; the contract parser normalizes them to the internal implementation identifiers (e.g., `NumRows`, `NullCount`) used inside DQX.

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

**Note:** Cross-dataset validation (referential integrity, schema consistency, aggregate reconciliation) should be handled at the orchestration/workflow level rather than in individual data contracts. This keeps contracts focused on single-dataset quality while allowing orchestration tools to manage relationships between datasets.

---

## Check Types Summary

### Table-Level Checks

Validate table-wide properties and aggregates. Specified in the top-level `checks` array.

| Check Type | Description | Return |
|------------|-------------|--------|
| [`num_rows`](#num_rows) | Total row count validation | count |
| [`duplicates`](#duplicates) | Count of duplicate rows | count or pct |
| [`freshness`](#freshness) | Data recency validation | age_hours |
| [`completeness`](#completeness) | Partition gap detection | gap_count |

**Total: 4 table-level checks**

---

### Column-Level Checks

Validate individual column values and properties. Specified within the `checks` array of a column definition.

#### Statistical Checks

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

#### Value Checks

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

## Validators

Every check returns a single float metric. Validators compare that metric against a declared acceptance condition. At most one validator may be specified per check. The optional `tolerance` parameter applies to all four and defaults to `0` for integer checks and `1e-6` for floating-point checks.

### min

Sets a lower bound. The check passes if `actual ≥ (min - tolerance)`.

```yaml
- name: "At least 1000 rows daily"
  type: num_rows
  min: 1000
  severity: P0
```

### max

Sets an upper bound. The check passes if `actual ≤ (max + tolerance)`.

```yaml
- name: "Low null percentage"
  type: nulls
  return: pct
  max: 0.05
  severity: P1
```

### between

Inclusive range — shorthand for specifying both `min` and `max`. The check passes if `actual ≥ (min - tolerance) AND actual ≤ (max + tolerance)`. Cannot be combined with `min` or `max`; doing so raises a `ContractValidationError`.

```yaml
- name: "Stable row count"
  type: num_rows
  between: [1000, 100000]
  severity: P1
```

### equals

Exact match. The check passes if `|actual - expected| ≤ tolerance`.

```yaml
- name: "Total allocated budget is exactly $1M"
  type: sum
  equals: 1000000.0
  tolerance: 0.01
  severity: P0
```

---

## Table-Level Checks

Table-level checks are specified in the top-level `checks` array (sibling to `columns`). They validate table-wide properties and aggregates.

---

### Volume Checks

#### `num_rows`

The `num_rows` check validates that the total row count falls within specified bounds.

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `duplicates`

The `duplicates` check validates that the number of duplicate rows (based on specified columns) is within bounds.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[string]` | Yes | None | Columns to check for duplicates (composite key) |
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

### Freshness Checks

#### `freshness`

The `freshness` check validates that data is not stale — the most recent timestamp falls within an acceptable age.

**Returns:** `age_hours` — the age of the most recent (or oldest, depending on `aggregation`) record in hours. Validated implicitly: the check passes when `age_hours <= max_age_hours`.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_age_hours` | `float` | Yes | None | Maximum allowed age in hours |
| `timestamp_column` | `string` | Yes | None | Column containing timestamps |
| `aggregation` | `string` | No | "max" | Aggregation method ("max" or "min") |

> Validators: not applicable — uses `max_age_hours` instead of standard validators.

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

#### `completeness`

The `completeness` check validates that partitioned data has no missing partitions within a time range (gap detection). This check only applies to tables with `metadata.partitioned_by` defined.

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

> Validators: not applicable — uses `max_gap_count` instead of standard validators.

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

## Column-Level Checks

Column-level checks are specified within the `checks` array of a column definition. They validate individual column values and statistical properties.

---

### Value Checks

Value checks validate individual values within a column (rather than computing aggregates over the column). Each check defines its own return semantics. For `nulls` and `duplicates`, the return value represents null or duplicate values found. For `whitelist`, `blacklist`, `pattern`, and `length`, the return value represents conforming (valid) rows. Use the `return` parameter to specify the return type (`count` or `pct`). When `return: pct` is used, the percentage denominator is always the total row count including nulls; the numerator differs by check family: for `nulls` and `duplicates` the numerator is the count of null or duplicate values respectively, while for `whitelist`, `blacklist`, `pattern`, and `length` the numerator is the count of conforming (valid) rows.

---

#### `nulls`

The `nulls` check validates null values in a column, returning the count or percentage of null values based on the `return` parameter.

**Return Parameter:** Use `return: count` (default) to return absolute null count, or `return: pct` to return null percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `whitelist`

The `whitelist` check validates that non-null values match a set of allowed values, returning the count or percentage of rows that conform.

**Return Parameter:** Use `return: count` (default) to return the count of matching rows, or `return: pct` to return the percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `values` | `list` | Yes | None | List of allowed values |
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |
| `case_sensitive` | `bool` | No | `true` | Whether comparison is case-sensitive (strings only) |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `blacklist`

The `blacklist` check validates that non-null values do not match any forbidden value, returning the count or percentage of rows that pass (rows not in the blacklist).

**Return Parameter:** Use `return: count` (default) to return the count of passing rows, or `return: pct` to return the percentage (0-1 scale). Both return values represent conforming rows — use `min` to set a lower bound (e.g., at least 95% must pass).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `values` | `list` | Yes | None | List of forbidden values |
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |
| `case_sensitive` | `bool` | No | `true` | Whether comparison is case-sensitive (strings only) |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

**Example 3: Percentage Passing**

```yaml
columns:
  - name: username
    type: string
    nullable: false
    description: "User username"
    checks:
      - name: "Almost all usernames are not reserved"
        type: blacklist
        values: ["admin", "root", "system"]
        return: pct
        min: 0.99  # At least 99% must pass
        severity: P1
```

---

#### `duplicates`

The `duplicates` check (column-level) validates that the count of duplicate values in a column is within specified bounds. This is the column-level version of the table-level `duplicates` check, which validates duplicates across multiple columns.

**Semantics:** Duplicate count is computed as `COUNT(*) - COUNT(DISTINCT col)` — the total number of non-first occurrences. If value "A" appears 3 times, it contributes 2 to the count. **NULL handling:** `COUNT(*)` includes all rows while `COUNT(DISTINCT col)` excludes NULLs, so NULL values are counted as duplicates. For example, `[A, A, NULL]` yields a duplicate count of 2 (one extra "A" plus one NULL). This is the intended behavior: NULLs are treated as duplicate occurrences. If you need to exclude NULLs, use `COUNT(col) - COUNT(DISTINCT col)` in a manual check instead.

**Return Parameter:** Use `return: count` (default) to return absolute duplicate count, or `return: pct` to return duplicate percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `return` | `string` | No | `count` | Return type: "count" (absolute) or "pct" (percentage 0-1) |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

### String and Collection Checks

String and collection checks are part of Value Checks. See the Value Checks section above for context.

#### `pattern`

The `pattern` check validates that string values match a pattern, supporting either explicit regex or predefined format shortcuts, and returns the count or percentage of conforming values.

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

\* Exactly ONE of `pattern` or `format` must be specified. Specifying `flags` together with `format` is invalid and raises a `ContractValidationError`.

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `length`

The `length` check validates that the length of string, list, or map values falls within specified bounds, returning the count or percentage of rows that meet the criteria. For strings, length is the character count. For lists and maps, length is the element count.

**Return Parameter:** Use `return: count` (default) to return count of rows within length bounds, or `return: pct` to return percentage (0-1 scale).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_length` | `int` | No* | None | Minimum allowed length |
| `max_length` | `int` | No* | None | Maximum allowed length |
| `return` | `string` | No | `count` | Return type: "count" or "pct" |

\* At least one of `min_length` or `max_length` required

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

**Example 5: List Column — Maximum Element Count**

```yaml
columns:
  - name: tags
    type:
      kind: list
      value_type: string
    nullable: true
    description: "Product tags"
    checks:
      - name: "Tags list is not too long"
        type: length
        max_length: 20
        severity: P1
```

**Example 6: Map Column — Element Count Range**

```yaml
columns:
  - name: properties
    type:
      kind: map
      key_type: string
      value_type: string
    nullable: true
    description: "Custom properties"
    checks:
      - name: "Properties map has reasonable size"
        type: length
        min_length: 1
        max_length: 50
        severity: P2
```

---

### Statistical Checks

Statistical checks compute aggregate metrics over the entire column and return a single numeric value.

#### `cardinality`

The `cardinality` check counts distinct non-null values in a column and validates that the count falls within specified bounds.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min` | `number` | No | None | Minimum allowed distinct value count |
| `max` | `number` | No | None | Maximum allowed distinct value count |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected distinct value count |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `min`

The `min` check validates that the minimum value in a column meets specified criteria — confirming that no value falls below an acceptable floor.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min` | `number` | No | None | Minimum allowed value for the column minimum |
| `max` | `number` | No | None | Maximum allowed value for the column minimum |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected value for the column minimum |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `max`

The `max` check validates that the maximum value in a column meets specified criteria — confirming that no value exceeds an acceptable ceiling.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min` | `number` | No | None | Minimum allowed value for the column maximum |
| `max` | `number` | No | None | Maximum allowed value for the column maximum |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected value for the column maximum |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `mean`

The `mean` check validates that the arithmetic mean of numeric values in a column falls within specified bounds.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min` | `number` | No | None | Minimum allowed mean value |
| `max` | `number` | No | None | Maximum allowed mean value |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected mean value |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

Use `min`, `max`, `between`, or `equals` to set bounds. Most use cases require range validation to detect drift in either direction.

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

#### `sum`

The `sum` check validates that the total of all non-null values in a column meets specified criteria.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min` | `number` | No | None | Minimum allowed sum |
| `max` | `number` | No | None | Maximum allowed sum |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected sum |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `count`

The `count` check validates that the number of non-null values in a column falls within specified bounds.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min` | `number` | No | None | Minimum allowed non-null count |
| `max` | `number` | No | None | Maximum allowed non-null count |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected non-null count |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `variance`

The `variance` check validates that the statistical variance of numeric values in a column falls within specified bounds, providing a measure of data spread.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min` | `number` | No | None | Minimum allowed variance |
| `max` | `number` | No | None | Maximum allowed variance |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected variance |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators).

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

#### `percentile`

The `percentile` check validates that a specific percentile value in a column falls within specified bounds.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `percentile` | `float` | Yes | None | Percentile to compute (0-100) |
| `min` | `number` | No | None | Minimum allowed percentile value |
| `max` | `number` | No | None | Maximum allowed percentile value |
| `between` | `[number, number]` | No | None | Inclusive range (shorthand for min + max) |
| `equals` | `number` | No | None | Exact expected percentile value |
| `tolerance` | `number` | No | `0` (int) / `1e-6` (float) | Acceptable variance |

> Validators: [`min`](#min), [`max`](#max), [`between`](#between), [`equals`](#equals) — see [Validators](#validators). The `percentile` parameter shares the same name as the check type but operates at a different YAML scope — `type` selects the check, `percentile` specifies the value to compute. They do not conflict.

Use `min`, `max`, `between`, or `equals` to set bounds. Most use cases require range validation to detect drift in either direction.

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

## Check Composition Patterns

Checks can be combined to create comprehensive validation strategies. Here are common patterns:

### Pattern 1: Layered Validation (Progressive Severity)

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

### Pattern 2: Range Validation with Context

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
        between: [50.0, 500.0]
        severity: P2
```

### Pattern 3: Multi-Level Validation

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
        min_length: 10
        max_length: 15
        severity: P1
```

### Pattern 4: Table-Level Reconciliation

```yaml
# Table-level checks
checks:
  # Volume check
  - name: "Daily volume within bounds"
    type: num_rows
    between: [1000, 100000]
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

### Pattern 5: Categorical with Cardinality

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
