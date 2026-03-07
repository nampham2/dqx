# Type System

> Part of the [Data Contracts Technical Specification](index.md).

## Design Philosophy

The type system prioritizes validation flexibility over exact matching. Types accept compatible variations (e.g., `int` accepts int8 through int64), require parameters only when semantically necessary (e.g., timezone for timestamp), and default to the simplest form.

---

## Type Reference Table

| Type | YAML Format | Validates Against | Notes |
|------|-------------|-------------------|-------|
| **Primitive Types** | | | |
| `int` | `type: int` | int8, int16, int32, int64, uint8, uint16, uint32, uint64 | Any integer width, signed or unsigned |
| `float` | `type: float` | float32, float64 | float32 and float64 only; float16 is not accepted |
| `bool` | `type: bool` | bool | Boolean exact match |
| `string` | `type: string` | string, utf8 | UTF-8 text; also accepts large_string, large_utf8 |
| `bytes` | `type: bytes` | binary, large_binary | Binary data |
| **Temporal Types** | | | |
| `date` | `type: date` | date32, date64 | Any date representation |
| `time` | `type: time` | time32(s/ms), time64(us/ns) | Any time representation |
| `timestamp` (simple) | `type: timestamp` | timestamp(any unit, any tz) | Accepts any timestamp |
| `timestamp` (UTC) | `type: {kind: timestamp}` | timestamp(any unit, UTC) | Object form defaults to UTC |
| `timestamp` (explicit tz) | `type: {kind: timestamp, tz: "..."}` | timestamp(any unit, specified tz) | Validates timezone match |
| **Decimal Type** | | | |
| `decimal` | `type: decimal` | decimal128(any), decimal256(any) | Any precision/scale |
| **Complex Types** | | | |
| `list` | `type: {kind: list, value_type: T}` | list\<T\> | Recursive validation of element type; also accepts large_list |
| `struct` | `type: {kind: struct, fields: [...]}` | struct\<fields\> | Recursive validation of field structure |
| `map` | `type: {kind: map, key_type: K, value_type: V}` | map\<K, V\> | Recursive validation of key/value types |

---

## Primitive Types

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

The validator checks semantic type, not storage format.

---

## Temporal Types

Store timestamps in UTC for consistency. The type system offers three levels of timezone strictness.

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

Use simple form when timezone doesn't matter or varies; use complex form with UTC (default) for most cases; use explicit timezone when data must be in a specific timezone.

---

## Decimal Type

```yaml
# Decimal - accepts any precision/scale
- name: amount
  type: decimal
  nullable: false
  description: "Transaction amount"

# Validates: decimal128(10, 2), decimal128(18, 4), decimal256(38, 6), etc.
```

The validator confirms the column is a decimal type regardless of precision or scale. Future versions may support precision constraints for strict financial validation.

---

## Complex Types

### List Type

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

### Struct Type

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

> **Note:** The `nullable` flag on nested struct and list element fields is for documentation purposes only. Schema validation enforces nullability at the top-level column only; nested field nullability is not validated.

### Map Type

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

Complex types nest arbitrarily (e.g., list of structs with maps). All types validate recursively.

---

## Type Mismatch Errors

When a column's actual type does not match the contract, the validator raises a `SchemaValidationError` describing the column, the expected type, and the actual type. The following example shows a contract that declares `user_id` as `int` but receives a `string` column:

```text
Contract declares: type: int
Actual column type: pa.string()
Error: SchemaValidationError: Column 'user_id' type mismatch: expected int (int8-int64, uint8-uint64), got string
```

The error message names the column, states the full set of accepted physical types, and identifies the actual type found. This lets engineers locate the source of the mismatch without inspecting the schema manually.

---

## Compatibility Reference

### Integer Type Compatibility

Contract type `int` validates against:
- `pa.int8()` — 8-bit signed integer (-128 to 127)
- `pa.int16()` — 16-bit signed integer (-32,768 to 32,767)
- `pa.int32()` — 32-bit signed integer (-2^31 to 2^31-1)
- `pa.int64()` — 64-bit signed integer (-2^63 to 2^63-1)
- `pa.uint8()` — 8-bit unsigned integer (0 to 255)
- `pa.uint16()` — 16-bit unsigned integer (0 to 65,535)
- `pa.uint32()` — 32-bit unsigned integer (0 to 2^32-1)
- `pa.uint64()` — 64-bit unsigned integer (0 to 2^64-1)

### Float Type Compatibility

Contract type `float` validates against:
- `pa.float32()` — 32-bit single precision (IEEE 754)
- `pa.float64()` — 64-bit double precision (IEEE 754)

Contract type `float` does **not** validate against `pa.float16()`.

### Date Type Compatibility

Contract type `date` validates against:
- `pa.date32()` — 32-bit signed integer, days since UNIX epoch
- `pa.date64()` — 64-bit signed integer, milliseconds since UNIX epoch

### Time Type Compatibility

Contract type `time` validates against:
- `pa.time32('s')` — 32-bit signed integer, seconds since midnight
- `pa.time32('ms')` — 32-bit signed integer, milliseconds since midnight
- `pa.time64('us')` — 64-bit signed integer, microseconds since midnight
- `pa.time64('ns')` — 64-bit signed integer, nanoseconds since midnight

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
- `pa.decimal128(precision, scale)` — Any precision/scale combination
- `pa.decimal256(precision, scale)` — Any precision/scale combination

### String Type Compatibility

Contract type `string` validates against:
- `pa.string()` — UTF-8 encoded variable-length string
- `pa.utf8()` — alias for string
- `pa.large_string()` — large UTF-8 string (64-bit offsets, common in DuckDB)
- `pa.large_utf8()` — alias for large_string

### List Type Compatibility

Contract type `list` validates against:
- `pa.list_(value_type)` — standard list with 32-bit offsets
- `pa.large_list(value_type)` — large list with 64-bit offsets
