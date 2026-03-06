# Type System

> Part of the [Data Contracts Technical Specification](index.md).

## Design Philosophy

The type system prioritizes validation flexibility over exact matching. Types accept compatible variations (e.g., `int` accepts int8 through int64), require parameters only when semantically necessary (e.g., timezone for timestamp), and default to the simplest form.

## Type Reference Table

| Type | YAML Format | Validates Against | Notes |
|------|-------------|-------------------|-------|
| **Primitive Types** | | | |
| `int` | `type: int` | int8, int16, int32, int64, uint8, uint16, uint32, uint64 | Any integer width, signed or unsigned |
| `float` | `type: float` | float32, float64 | Any float precision |
| `bool` | `type: bool` | bool | Boolean exact match |
| `string` | `type: string` | string, utf8 | UTF-8 text |
| `bytes` | `type: bytes` | binary, large_binary | Binary data |
| **Temporal Types** | | | |
| `date` | `type: date` | date32, date64 | Any date representation |
| `time` | `type: time` | time32(s/ms), time64(us/ns) | Any time representation |
| `timestamp` | `type: timestamp` or `type: {kind: timestamp}` or `type: {kind: timestamp, tz: "America/New_York"}` | timestamp(any unit) | Simple form is flexible; object form validates timezone (defaults to UTC) |
| **Decimal Type** | | | |
| `decimal` | `type: decimal` | decimal128(any), decimal256(any) | Any precision/scale |
| **Complex Types** | | | |
| `list` | `type: {kind: list, value_type: T}` | list\<T\> | Recursive validation of element type |
| `struct` | `type: {kind: struct, fields: [...]}` | struct\<fields\> | Recursive validation of field structure |
| `map` | `type: {kind: map, key_type: K, value_type: V}` | map\<K, V\> | Recursive validation of key/value types |

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

For validation, exact widths and precisions are implementation details. We verify the semantic type, not storage format.

## Temporal Types

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

Most timestamps should be stored in UTC for consistency. Use simple form when timezone doesn't matter or varies; use complex form with UTC (default) for most cases; use explicit timezone when data must be in a specific timezone.

## Decimal Type

```yaml
# Decimal - accepts any precision/scale
- name: amount
  type: decimal
  nullable: false
  description: "Transaction amount"

# Validates: decimal128(10, 2), decimal128(18, 4), decimal256(38, 6), etc.
```

For initial validation, we verify it's a decimal type. Precision/scale parameters can be added later if needed for strict financial validation.

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

Complex types can be nested arbitrarily (e.g., list of structs with maps). All types validate recursively.

---
