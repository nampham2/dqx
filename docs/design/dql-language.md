# DQL: A Language for Data Quality

## Goals

DQL (Data Quality Language) is designed with three goals:

1. **Simple, dedicated syntax** — A language built specifically for data quality checks, where intent is clear and boilerplate is minimal
2. **DQX engine integration** — Runs directly on the DQX runtime, leveraging its metric computation, profiling, and validation capabilities
3. **AI-native** — Structured and unambiguous enough for AI agents to read, write, and modify autonomously when investigating and resolving data quality issues

## Solution

DQL expresses data quality checks in a syntax built for the task. The interpreter validates and executes checks directly against your database.

```dql
suite "Order Validation" {
    check "Completeness" on orders {
        assert null_count(customer_id) == 0  severity P0
        assert null_count(email) / num_rows() < 5%
    }
}
```

Run:

```bash
dql run suite.dql --connection databricks://... --date 2024-12-25
```

## Structure

A DQL file contains one suite. A suite contains checks, profiles, constants, and macros.

```
suite
├── metadata (name, threshold)
├── constants
├── macros
├── checks
│   └── assertions
└── profiles
    └── rules
```

### Suite

Every DQL file begins with a suite declaration:

```dql
suite "E-Commerce Data Quality" {
    availability_threshold 80%

    # checks and profiles go here
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Identifies the suite |
| `availability_threshold` | No | Minimum data availability (default: 90%) |

### Check

A check groups related assertions. Each check targets one or more datasets.

```dql
check "Price Validation" on orders {
    assert average(price) > 0
    assert maximum(price) < 10000
}
```

Target multiple datasets:

```dql
check "Cross-Dataset" on orders, returns {
    assert num_rows(dataset returns) / num_rows(dataset orders) < 15%
}
```

**Cross-dataset semantics:**
- Each dataset resolves independently for the target date
- Division by zero (empty dataset) returns `None`, triggering assertion failure
- No implicit join—metrics compute per-dataset, then combine

### Assertion

An assertion validates one metric against one condition. Write assertions as declarative statements:

```dql
assert average(price) > 0
```

Add a name, severity, tolerance, or tags:

```dql
assert average(price) > 0
    name "Average price is positive"
    severity P0
    tags [pricing, critical]
```

With tolerance for floating-point comparison:

```dql
assert average(price) / average(price, lag 1) == 1.0 tolerance 0.05
    name "Ratio near 1.0"
```

ASCII tolerance alternatives: `tolerance 0.05` or `+/- 0.05` (Unicode `±` also supported).

| Element | Required | Description |
|---------|----------|-------------|
| `name` | No | Descriptive label (recommended) |
| `metric` | Yes | Expression to evaluate |
| `condition` | Yes | Comparison or keyword |
| `severity` | No | P0, P1, P2, or P3 (default: P1) |
| `tolerance` | No | Margin for `==` comparisons (expands to range check) |
| `tags` | No | Labels for profile targeting |

### Assertion Naming Convention

Names serve as unique identifiers for programmatic reference. Use hierarchical names for machine-friendly identification:

```dql
check "Completeness" on orders {
    assert null_count(email) / num_rows() < 5%
        name "orders.completeness.email_null"

    assert null_count(phone) / num_rows() < 10%
        name "orders.completeness.phone_null"
}
```

**Convention:** `dataset.check.assertion` — enables algorithms to reference, add, or remove assertions by name.

### Assertion Annotations

Annotations provide metadata for algorithmic optimization:

```dql
check "Volume" on orders {
    @experimental                    # Algorithm-proposed, not yet validated
    assert day_over_day(sum(tax)) < 0.5
        name "orders.volume.tax_stability"

    @cost(false_positive=1, false_negative=100)
    assert num_rows() >= 1000
        name "orders.volume.min_rows"
        severity P0
}
```

| Annotation | Description |
|------------|-------------|
| `@experimental` | Marks assertion as algorithm-proposed, pending validation |
| `@required` | Assertion cannot be removed or disabled by algorithms |
| `@cost(false_positive=N, false_negative=M)` | Cost of false positive (N) and false negative (M) for RL reward |

**Cost semantics:**
- `false_positive` — Cost when assertion fails but data is actually fine (alert fatigue)
- `false_negative` — Cost when assertion passes but data has issues (missed problem)
- Higher `fn` relative to `fp` makes the algorithm prefer stricter thresholds

**Safety constraints:**

Prevent algorithms from gaming the system by removing critical checks:

```dql
check "Critical" on orders {
    @required                        # Cannot be removed by RL agent
    @cost(false_positive=1, false_negative=1000)
    assert null_count(customer_id) == 0
        name "orders.critical.customer_id"
        severity P0
}
```

| Constraint | Behavior |
|------------|----------|
| `@required` | Critical assertion that must always run |
| `severity P0` | Implied required unless explicitly `@experimental` |
| Non-tunable threshold | Threshold cannot be modified (no `tunable` keyword) |

**Constraint enforcement:** Tunable constants can only be adjusted within their declared bounds. Non-tunable thresholds and `@required` assertions represent fixed business logic that algorithms cannot modify.

**Design principle:** Algorithms can only tune what humans explicitly marked as tunable. Critical business logic stays protected.

### Conditions

Conditions specify how to validate a metric:

| Syntax | Meaning |
|--------|---------|
| `> N` | Greater than N |
| `>= N` | Greater than or equal to N |
| `< N` | Less than N |
| `<= N` | Less than or equal to N |
| `== N` | Equal to N |
| `!= N` | Not equal to N |
| `between A and B` | In range [A, B] |
| `is positive` | Greater than zero |
| `is negative` | Less than zero |
| `is None` | Value is None |
| `is not None` | Value is not None |

## Metrics

Metrics compute values from data. DQL provides built-in metrics and supports arithmetic combinations.

### Built-in Metrics

**Aggregate metrics** compute over all rows:

```dql
num_rows()              # Row count
average(price)          # Mean value
sum(revenue)            # Total
minimum(quantity)       # Smallest value
maximum(temperature)    # Largest value
variance(score)         # Statistical variance
```

**Completeness metrics** measure data presence:

```dql
null_count(email)           # Count of None values
unique_count(customer_id)   # Count of distinct values
duplicate_count([id, date]) # Count of duplicate combinations
```

**Value metrics** check specific values:

```dql
count_values(status, "pending")  # Rows where status == "pending"
first(timestamp)                 # First value by row order
first(timestamp, order_by price) # First value ordered by price
```

`first()` returns the first value in storage order unless `order_by` specifies a column.

**Warning:** Without `order_by`, results are non-deterministic for distributed storage (Parquet, Delta Lake). Row order depends on file layout and query execution. Always use `order_by` for reproducible results.

**Custom SQL** executes arbitrary expressions:

```dql
sql("SUM(amount) / COUNT(*)")
```

**`sql()` limitations:**

- **No validation** — The interpreter cannot verify column names, syntax, or types
- **No interpolation** — Constants and macro parameters cannot be inserted into SQL strings (prevents SQL injection)
- **Dialect-specific** — SQL syntax varies across databases; DQL does not translate

```dql
# Preferred: interpreter validates
assert sum(amount) / num_rows() > 10

# Escape hatch: no validation
assert sql("SUM(amount) / COUNT(*)") > 10

# ERROR: interpolation not allowed in sql() — parser rejects this
macro bad_sql(col) {
    assert sql("SUM({col})") > 0  # Parse error: interpolation in sql() not permitted
}
```

Use built-in metrics when possible; reserve `sql()` for expressions DQL cannot represent.

### Parameters

Metrics accept optional parameters:

```dql
average(price, lag 1)           # Yesterday's average
average(price, dataset orders)  # Specific dataset
```

| Parameter | Description |
|-----------|-------------|
| `lag` | Calendar days offset (1 = yesterday, 7 = one week ago) |
| `dataset` | Restrict metric to named dataset |

**Lag semantics:** The `lag` parameter offsets by calendar days, not partition offsets. For a weekly-partitioned table queried on Monday with `lag 1`, the interpreter computes the metric for Sunday's data (which may fall in the previous week's partition).

### Extended Metrics

Extended metrics wrap base metrics for time-series analysis:

```dql
day_over_day(average(price))     # Absolute % change vs yesterday
week_over_week(sum(revenue))     # Absolute % change vs last week
stddev(average(price), n 7)      # Standard deviation over 7 days
```

Day-over-day computes `abs((today - yesterday) / yesterday)`. A result of `0.1` means 10% change.

### Arithmetic

Combine metrics with arithmetic operators:

```dql
null_count(email) / num_rows()   # Null percentage
sum(revenue) - sum(cost)         # Profit
average(price) * 1.1             # 10% markup
```

Math functions:

```dql
abs(day_over_day(price) - 1.0)   # Distance from no change
sqrt(variance(score))            # Standard deviation
```

Supported functions: `abs`, `sqrt`, `log`, `exp`.

Utility functions:

```dql
coalesce(average(price), 0)      # Default if None
coalesce(sum(cost), sum(revenue), 0)  # First non-None
```

Supported: `coalesce(expr, expr, ...)`.

## Constants

Constants define reusable values. Declare them at suite level:

```dql
suite "Orders" {
    const NULL_THRESHOLD = 5%
    const MIN_ORDERS = 1000

    check "Completeness" on orders {
        assert null_count(email) / num_rows() < NULL_THRESHOLD
        assert num_rows() >= MIN_ORDERS
    }
}
```

### Tunable Constants

Mark constants as tunable to enable algorithmic optimization (e.g., RL-based threshold tuning):

```dql
const NULL_THRESHOLD = 5% tunable [0%, 20%]      # Algorithm can adjust within bounds
const MIN_ORDERS = 1000 tunable [100, 10000]     # Integer bounds
const VARIANCE_LIMIT = 0.5 tunable [0.1, 1.0]    # Decimal bounds
```

| Element | Required | Description |
|---------|----------|-------------|
| `tunable` | No | Marks constant for algorithmic adjustment |
| `[min, max]` | Yes (if tunable) | Valid range for optimization |

**Semantics:**
- Algorithms can modify tunable constants within bounds
- Non-tunable constants are fixed and cannot be changed programmatically
- Bounds are inclusive: `[0%, 20%]` allows values from 0% to 20%

#### Python Tunable API

Tunables are implemented as an extensible type hierarchy in `dqx/tunables.py`:

```python
from dqx.tunables import TunableFloat, TunablePercent, TunableInt, TunableChoice

# Define tunables with type-specific validation
NULL_THRESHOLD = TunablePercent("NULL_THRESHOLD", value=0.05, bounds=(0.0, 0.20))
MIN_ORDERS = TunableInt("MIN_ORDERS", value=1000, bounds=(100, 10000))
TOLERANCE = TunableFloat("TOLERANCE", value=0.001, bounds=(0.0001, 0.01))
AGG_METHOD = TunableChoice(
    "AGG_METHOD", value="mean", choices=("mean", "median", "max")
)

# Register with suite at construction
suite = VerificationSuite(
    checks=[completeness_check],
    db=db,
    name="Orders",
    tunables=[NULL_THRESHOLD, MIN_ORDERS, TOLERANCE, AGG_METHOD],
)


# Use in assertions
@check(name="Completeness")
def completeness_check(mp: MetricProvider, ctx: Context):
    ctx.assert_that(mp.null_count("email") / mp.num_rows()).where(
        name="Email null rate"
    ).is_lt(
        NULL_THRESHOLD.value
    )  # .value gets current value
```

**Tunable Types:**

| Type | Description | Example |
|------|-------------|---------|
| `TunableFloat` | Bounded float | `TunableFloat("X", value=0.5, bounds=(0.0, 1.0))` |
| `TunablePercent` | Percentage (0-1 internally) | `TunablePercent("X", value=0.05, bounds=(0.0, 0.20))` |
| `TunableInt` | Bounded integer | `TunableInt("X", value=100, bounds=(10, 1000))` |
| `TunableChoice` | Categorical | `TunableChoice("X", value="a", choices=("a", "b", "c"))` |

**RL Agent API:**

```python
# Get all tunable parameters for action space
params = suite.get_tunable_params()
# [
#   {"name": "NULL_THRESHOLD", "type": "percent", "value": 0.05, "bounds": (0.0, 0.20)},
#   {"name": "MIN_ORDERS", "type": "int", "value": 1000, "bounds": (100, 10000)},
# ]

# Modify with validation and history tracking
suite.set_param("NULL_THRESHOLD", 0.03, agent="rl_optimizer", reason="Episode 42")

# View change history
history = suite.get_param_history("NULL_THRESHOLD")
# [TunableChange(timestamp=..., old_value=0.05, new_value=0.03, agent="rl_optimizer", ...)]
```

### Percentage Semantics

Percentage literals convert to decimals: `5%` becomes `0.05`.

```dql
assert null_count(email) / num_rows() < 5%   # 5% = 0.05
assert day_over_day(num_rows()) < 0.5        # 0.5 = 50% change (raw decimal)
```

**Rule:** Use `%` suffix for human-readable percentages. Use raw decimals for ratios. The interpreter treats `5%` and `0.05` identically—the distinction is for readability.

### None Handling

None propagates through arithmetic:

```dql
sum(revenue) - sum(cost)   # None if either is None
```

None in a comparison fails the assertion:

```dql
assert average(price) > 0   # Fails if average(price) is None
```

**Division by zero** returns None:

```dql
sum(a) / sum(b)   # None if sum(b) == 0
0 / 0             # None (not NaN)
```

Use `coalesce` to provide defaults:

```dql
assert coalesce(average(price), 0) >= 0
```

## Macros

Macros generate repeated patterns. Define a macro with parameters:

```dql
macro null_check(column, threshold) {
    assert null_count({column}) / num_rows() < {threshold}
        name "{column} null rate"
}
```

Apply with `use`:

```dql
check "Completeness" on orders {
    use null_check(email, 5%)
    use null_check(phone, 10%)
    use null_check(address, 15%)
}
```

The interpreter expands macros before execution.

### Variadic Macros

Accept multiple arguments with `...`:

```dql
macro completeness(columns...) {
    for col in columns {
        assert null_count({col}) == 0
            name "{col} not null"
    }
}

check "Completeness" on orders {
    use completeness(customer_id, email, amount)
}
```

### Macro Semantics

**Scoping:** Macros use lexical scoping. Parameters shadow outer constants.

```dql
const THRESHOLD = 5%

macro check_rate(THRESHOLD) {           # Parameter shadows constant
    assert error_rate() < {THRESHOLD}   # Uses parameter, not constant
}
```

**Recursion:** Macros cannot call themselves. The interpreter rejects recursive definitions.

**Hygiene:** Macro-generated assertion logic is duplicated, but names are not auto-suffixed. If you call `use null_check(email, 5%)` twice, both produce `name "email null rate"`. The interpreter warns on duplicate names. Use unique parameters or omit names to let the interpreter generate them:

```dql
macro null_check(column, threshold) {
    assert null_count({column}) / num_rows() < {threshold}
    # No explicit name — interpreter generates unique ID
}
```

**Order:** Macros must be defined before use. Forward references are errors.

**Nesting:** Macros can call other macros:

```dql
macro base_check(col) {
    assert null_count({col}) == 0
}

macro extended_check(col) {
    use base_check({col})
    assert unique_count({col}) == num_rows()
}
```

## Profiles

Profiles modify assertion behavior during specific periods. Define profiles at suite level.

```dql
profile "Black Friday" {
    type holiday
    from 2024-11-29
    to   2024-12-02

    disable check "Volume"
    scale tag "seasonal" by 3.0x
}
```

### Profile Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Profile identifier |
| `type` | Yes | Profile type: `holiday` or `recurring` |
| `from` | Yes | Start date (ISO format or expression) |
| `to` | Yes | End date (ISO format or expression) |

### Profile Types

**`holiday`** — Fixed dates. ISO dates apply to that specific year only. Date expressions re-evaluate each year.

**`recurring`** — Re-evaluates `from`/`to` expressions relative to the execution date (e.g., monthly patterns).

### Dynamic Dates

Use date expressions for recurring events:

```dql
profile "Thanksgiving Week" {
    type holiday
    from nth_weekday(november, thursday, 4)      # 4th Thursday
    to   nth_weekday(november, thursday, 4) + 3  # Through Sunday
}

profile "Christmas" {
    type holiday
    from december(20)    # December 20 of execution year
    to   january(5)      # January 5 of following year
}

profile "Month End" {
    type recurring
    from last_day_of_month() - 2
    to   last_day_of_month()
}
```

**Year handling:** When `to` references a month earlier than `from` (e.g., December → January), the interpreter infers year rollover. Explicit year can be specified: `january(5, year + 1)`.

Date functions:

| Function | Description |
|----------|-------------|
| `nth_weekday(month, day, n)` | Nth occurrence of weekday in month |
| `last_day_of_month()` | Last day of current month |
| `month(day)` | Specific day in month (e.g., `december(25)`) |
| `month(day, year)` | Explicit year: `year`, `year + 1`, `year - 1` |
| Date arithmetic | `+ N` or `- N` for day offsets |

### Rules

Rules select assertions and apply modifications:

**Disable** skips matched assertions:

```dql
disable check "Volume"
disable assertion "Order count" in "Volume"
```

**Scale** multiplies the computed metric value before comparison:

```dql
scale tag "seasonal" by 2.0x
scale check "Revenue" by 1.5x
```

**Scale semantics:** The multiplier applies to the metric value, not the threshold. Use this when you expect the metric to be *lower* than normal due to reduced activity.

```dql
# Normal: assert num_rows() >= 1000
# Holiday traffic drops to 500 rows (50% of normal)
# Without profile: 500 >= 1000 → FAIL (false positive!)
#
# With scale 2.0x: 500 × 2.0 = 1000 >= 1000 → PASS
# Interpretation: "500 rows during a 0.5× traffic period is like 1000 rows normally"
```

**Mental model:** Scale answers: "What would this metric be under normal conditions?" A scale of 2.0x compensates for a period with 50% expected traffic—multiply the actual value to normalize it for comparison against normal-period thresholds.

For `between A and B`, the scaled metric compares against both bounds unchanged.

**Downgrade** changes severity:

```dql
downgrade tag "non-critical" to P3
```

### Rule Ordering

Rules apply in definition order. Multipliers compound; severity uses last match.

```dql
profile "Holiday" {
    type holiday
    from 2024-12-20
    to   2025-01-05

    scale tag "volume" by 1.5x
    scale check "Orders" by 2.0x
    # Assertion with tag "volume" in "Orders": multiplier = 3.0x
}
```

### Profile Overlap

When multiple profiles match the same date, rules from all matching profiles apply in profile definition order. Multipliers compound across profiles.

```dql
profile "Holiday Season" { from 2024-11-15 to 2025-01-05 scale tag "volume" by 1.5x }
profile "Black Friday" { from 2024-11-29 to 2024-12-02 scale tag "volume" by 2.0x }

# On 2024-11-30: Both profiles active
# Combined multiplier for "volume" tag: 1.5 × 2.0 = 3.0x
```

**Warning:** Overlapping profiles with compounding multipliers can produce unexpected results. The interpreter logs active profiles and combined multipliers for debugging.

## Imports

Split large configurations across files with imports:

```dql
# common/null_checks.dql
export macro null_check(column, threshold) {
    assert null_count({column}) / num_rows() < {threshold}
}

export const STANDARD_NULL_THRESHOLD = 5%
```

Import in another file:

```dql
import "common/null_checks.dql"

suite "Orders" {
    check "Completeness" on orders {
        use null_check(email, STANDARD_NULL_THRESHOLD)
    }
}
```

Selective imports:

```dql
import { null_check } from "common/null_checks.dql"
```

Aliased imports:

```dql
import "common/null_checks.dql" as checks

suite "Orders" {
    check "Test" on orders {
        use checks.null_check(email, 5%)
    }
}
```

### Path Resolution

Import paths resolve in order:

1. **Relative to importing file** — `./utils.dql` or `../common/checks.dql`
2. **Relative to project root** — `common/null_checks.dql` (no leading `./`)
3. **DQL_PATH directories** — Colon-separated list of search directories

```bash
export DQL_PATH="/shared/dql-libs:/team/common"
dql run suite.dql
```

### Import Cycles

Circular imports are detected and rejected:

```dql
# a.dql
import "b.dql"   # Error: circular import detected: a.dql → b.dql → a.dql

# b.dql
import "a.dql"
```

The interpreter builds a dependency graph before execution and fails fast if cycles exist.

## Complete Example

```dql
suite "E-Commerce Data Quality" {
    availability_threshold 80%

    const MAX_NULL_RATE = 5%
    const MIN_ORDERS = 1000

    macro null_rate(column) {
        assert null_count({column}) / num_rows() < MAX_NULL_RATE
            name "{column} null rate below threshold"
    }

    check "Completeness" on orders {
        assert null_count(customer_id) == 0
            name "No null customer IDs"
            severity P0

        use null_rate(email)
        use null_rate(phone)
    }

    check "Volume" on orders {
        assert num_rows() >= MIN_ORDERS
            name "At least minimum orders"
            tags [volume]

        assert day_over_day(num_rows()) between 0.5 and 2.0
            name "Day-over-day stable"
            tags [volume, trend]
    }

    check "Revenue" on orders {
        assert sum(amount) + sum(tax) is positive
            name "Total revenue positive"
            severity P0

        assert sum(amount) / num_rows() between 10 and 500
            name "Average order value in range"
    }

    check "Cross-Dataset" on orders, returns {
        assert num_rows(dataset returns) / num_rows(dataset orders) < 15%
            name "Return rate below 15%"
    }

    check "Stability" on orders {
        assert abs(day_over_day(average(price))) < 0.1
            name "Price change within 10%"
            tags [stability]

        assert sqrt(variance(amount)) / average(amount) < 0.5
            name "Coefficient of variation acceptable"
            tags [stability]
    }

    profile "Black Friday" {
        type holiday
        from 2024-11-29
        to   2024-12-02

        scale tag "volume" by 3.0x
    }

    profile "Christmas" {
        type holiday
        from 2024-12-20
        to   2025-01-05

        disable check "Volume"
        downgrade tag "trend" to P3
    }
}
```

## History File

DQL tracks changes in a separate history file to support algorithmic optimization and auditing. The history file is a JSONL (JSON Lines) file alongside the DQL source.

**File convention:**
- `suite.dql` — The specification (human + algorithm authored)
- `suite.dql.history` — Change log (append-only)

### History Format

```jsonl
{"ts": "2024-12-01T10:00:00Z", "action": "set_param", "param": "NULL_THRESHOLD", "old": 0.10, "new": 0.05, "agent": "rl_optimizer", "episode": 42}
{"ts": "2024-12-15T14:30:00Z", "action": "set_param", "param": "MIN_ORDERS", "old": 1000, "new": 800, "agent": "autotuner", "reason": "seasonal adjustment"}
{"ts": "2024-12-20T09:00:00Z", "action": "set_param", "param": "TOLERANCE", "old": 0.001, "new": 0.002, "agent": "human", "reason": "reduce false positives"}
```

### History Actions

| Action | Description | Fields |
|--------|-------------|--------|
| `set_param` | Tunable constant changed | `param`, `old`, `new`, `agent`, `reason` |

### Agent Field

The `agent` field identifies who made the change:
- `human` — Manual edit by engineer
- `rl_optimizer` — Reinforcement learning algorithm
- `autotuner` — Threshold optimization algorithm
- Custom agent names for other automation

### Usage

```bash
# View history
dql history suite.dql

# Rollback to previous state
dql rollback suite.dql --to 2024-12-15

# Export history for analysis
dql history suite.dql --format csv > changes.csv
```

The history file can be git-tracked separately or added to `.gitignore` depending on team preference.

## Runtime

The interpreter executes DQL files directly against your database. Metric expressions pass to sympy for parsing, enabling full compatibility with DQX's Python runtime.

```
┌─────────────────────────────────────────────────────────────┐
│                      DQL Source                             │
│                     (suite.dql)                             │
└──────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Interpreter   │
                    │   (via sympy)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   DQX Runtime   │
                    │   (database)    │
                    └─────────────────┘
```

### Installation

```bash
pip install dqx[dql]
```

### Commands

**Run a suite**:

```bash
dql run suite.dql --connection databricks://host/warehouse --date 2024-12-25
```

**Validate without executing**:

```bash
dql check suite.dql
```

**Watch for changes** (re-runs on file modification):

```bash
dql run suite.dql --watch
```

**Output formats**:

```bash
dql run suite.dql --output json      # Machine-readable
dql run suite.dql --output table     # Human-readable (default)
dql run suite.dql --output summary   # Pass/fail counts only
```

### Configuration File

Store connection settings in `dqx.toml` to avoid repeating arguments:

```toml
[connection]
type = "databricks"
host = "workspace.cloud.databricks.com"
http_path = "/sql/1.0/warehouses/abc123"

[defaults]
date = "today"
output = "table"
```

Then run:

```bash
dql run suite.dql
```

### Python API

Embed the interpreter in Python code:

```python
from dqx.dql import Interpreter
from datetime import date

interp = Interpreter(db=my_db, target_date=date.today())
results = interp.run_file("suite.dql")

for r in results:
    print(f"{r.check}/{r.assertion}: {r.status}")
```

Or run from a string:

```python
dql_source = """
suite "Quick Check" {
    check "Basic" on orders {
        assert num_rows() > 0
    }
}
"""
results = interp.run_string(dql_source)
```

### RL Agent Integration

DQL provides a programmatic API for reinforcement learning agents to optimize data quality checks:

1. Read tunable parameters and their bounds
2. Modify thresholds within bounds
3. Run checks and observe outcomes
4. Compute rewards from results and costs
5. Log changes to history

**Action Space:**

The RL agent operates on a continuous, bounded action space derived from tunable constants:

| Component | Type | Description |
|-----------|------|-------------|
| **Threshold adjustments** | Continuous, bounded | One dimension per `tunable` constant |

```python
# Continuous action space derived from tunable constants
action_space = {
    "NULL_THRESHOLD": (0.0, 0.20),  # from tunable [0%, 20%]
    "MIN_ORDERS": (100, 10000),  # from tunable [100, 10000]
    "DOD_LIMIT": (0.1, 1.0),  # from tunable [0.1, 1.0]
}

# Example action from policy
action = {
    "NULL_THRESHOLD": 0.03,  # Lower threshold (stricter)
    "MIN_ORDERS": 1500,  # Raise minimum
    "DOD_LIMIT": 0.4,  # Slightly tighter
}
```

For threshold optimization, use bounded continuous control algorithms (PPO, SAC).

**Example DQL with tunable thresholds and costs:**

```dql
suite "Orders" {
    const NULL_THRESHOLD = 5% tunable [0%, 20%]
    const MIN_ORDERS = 1000 tunable [100, 10000]
    const DOD_LIMIT = 0.5 tunable [0.1, 1.0]

    check "Completeness" on orders {
        @cost(false_positive=1, false_negative=50)
        assert null_count(email) / num_rows() < NULL_THRESHOLD
            name "orders.completeness.email_null"

        @cost(false_positive=1, false_negative=100)
        assert null_count(customer_id) == 0
            name "orders.completeness.customer_id_null"
            severity P0
    }

    check "Volume" on orders {
        @cost(false_positive=5, false_negative=200)
        assert num_rows() >= MIN_ORDERS
            name "orders.volume.min_rows"

        @experimental
        @cost(false_positive=2, false_negative=50)
        assert day_over_day(num_rows()) < DOD_LIMIT
            name "orders.volume.dod_stability"
    }
}
```

**RL Agent Implementation:**

```python
from dqx.dql import Suite, History
from datetime import date, timedelta
import numpy as np


class DQThresholdAgent:
    """RL agent that optimizes DQL thresholds."""

    def __init__(self, suite_path: str, db):
        self.suite = Suite.load(suite_path)
        self.history = History(suite_path + ".history")
        self.db = db
        self.episode = 0

    def get_state(self) -> np.ndarray:
        """Extract current thresholds as state vector."""
        params = self.suite.get_tunable_params()
        return np.array([p.normalized_value for p in params])

    def get_action_space(self) -> list:
        """Get tunable parameters with bounds."""
        return [
            {"name": p.name, "bounds": (p.min_bound, p.max_bound)}
            for p in self.suite.get_tunable_params()
        ]

    def apply_action(self, actions: dict[str, float]):
        """Apply threshold changes from RL policy."""
        for name, new_value in actions.items():
            old = self.suite.get_param(name)
            self.suite.set_param(name, new_value)
            self.history.log(
                {
                    "action": "set_param",
                    "param": name,
                    "old": old,
                    "new": new_value,
                    "agent": "rl_optimizer",
                    "episode": self.episode,
                }
            )

    def run_and_observe(self, target_date: date) -> dict:
        """Execute checks and return structured results."""
        results = self.suite.run(db=self.db, date=target_date)
        return {
            "assertions": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "value": r.metric_value,
                }
                for r in results.assertions
            ]
        }

    def compute_reward(self, results: dict, ground_truth: dict = None) -> float:
        """Compute reward from results.

        Simple reward: +1 for correct predictions, -1 for incorrect.
        Can be extended to use @cost annotations when available.
        """
        reward = 0.0
        for a in results["assertions"]:
            if ground_truth and a["name"] in ground_truth:
                actual_issue = ground_truth[a["name"]]
                if a["passed"] == (not actual_issue):
                    reward += 1.0  # Correct prediction
                else:
                    reward -= 1.0  # Incorrect prediction
        return reward

    def save(self):
        """Persist changes to DQL file and history."""
        self.suite.save()
        self.history.flush()
```

**Training Loop:**

```python
agent = DQThresholdAgent("suite.dql", db_connection)

for episode in range(1000):
    agent.episode = episode
    state = agent.get_state()

    # RL policy selects threshold adjustments
    actions = policy.select_action(state)
    agent.apply_action(actions)

    # Run on historical date for training
    train_date = date.today() - timedelta(days=np.random.randint(1, 90))
    results = agent.run_and_observe(train_date)

    # Reward signal (with optional ground truth labels)
    reward = agent.compute_reward(results, labeled_data.get(train_date))

    # Policy gradient update
    next_state = agent.get_state()
    policy.update(state, actions, reward, next_state)

    if episode % 100 == 0:
        agent.save()
```

**Key API Methods:**

| Method | Description |
|--------|-------------|
| `Suite.load(path)` | Parse DQL file into manipulable object |
| `suite.get_tunable_params()` | List parameters with bounds for action space |
| `suite.set_param(name, value)` | Modify threshold (validates bounds) |
| `suite.get_param(name)` | Get current value of a parameter |
| `suite.get_param_history(name)` | Get change history for a parameter |
| `suite.run(db, date)` | Execute checks, return structured results |
| `suite.save()` | Persist changes back to `.dql` file |
| `History.log(entry)` | Append to `.dql.history` file |

**Profile-Aware Training:**

Profiles apply multipliers to metrics during certain periods (holidays, promotions). The RL agent must account for this when training on historical data:

```python
def run_and_observe(self, target_date: date) -> dict:
    """Execute checks with profile context."""
    results = self.suite.run(db=self.db, date=target_date)

    # Get active profiles for this date
    active_profiles = self.suite.get_active_profiles(target_date)

    return {
        "assertions": [...],
        "profiles": [
            {"name": p.name, "multipliers": p.get_multipliers()}
            for p in active_profiles
        ],
        "is_special_period": len(active_profiles) > 0,
    }
```

**Two training strategies:**

| Strategy | Description | When to use |
|----------|-------------|-------------|
| **Profile-aware** | Include profile context in state, learn separate thresholds | When profiles are stable and well-defined |
| **Profile-excluded** | Train only on non-profile dates | When you want single robust threshold |

```python
# Strategy 1: Profile-aware state
def get_state(self) -> np.ndarray:
    params = [p.normalized_value for p in self.suite.get_tunable_params()]

    # Add profile indicators to state
    profile_active = [
        1.0 if self.suite.is_profile_active(p.name, self.current_date) else 0.0
        for p in self.suite.get_profiles()
    ]

    return np.array(params + profile_active)


# Strategy 2: Exclude profile periods from training
def sample_training_date(self) -> date:
    while True:
        d = date.today() - timedelta(days=np.random.randint(1, 90))
        if not self.suite.get_active_profiles(d):
            return d  # Only train on "normal" periods
```

**Profile multipliers in reward computation:**

The interpreter applies multipliers before comparison, so the RL agent sees *post-multiplier* results. This is important:

```dql
check "Volume" on orders {
    assert num_rows() >= 1000
        tags [volume]
}

profile "Holiday" {
    type holiday
    from 2024-12-20
    to   2024-12-31
    scale tag "volume" by 2.0x
}
```

```python
# On holiday: actual rows = 600
# Multiplier 2.0x applied: 600 * 2.0 = 1200
# Comparison: 1200 >= 1000 → PASS

# The agent sees:
{
    "name": "orders.volume.min_rows",
    "passed": True,
    "raw_value": 600,  # Before multiplier
    "scaled_value": 1200,  # After multiplier (used for comparison)
    "multiplier": 2.0,
    "threshold": 1000,
}
```

**Implication:** Thresholds are defined for "normal" periods. Profiles handle anomalous periods. The RL agent should optimize thresholds for normal conditions—profiles automatically adjust for special periods.

**Human-in-the-Loop Integration:**

The system supports multiple touchpoints for human oversight:

| Touchpoint | Purpose | Frequency |
|------------|---------|-----------|
| **Review queue** | Approve/reject algorithm proposals | Before deployment |
| **Ground truth labeling** | Improve reward signal accuracy | Periodic batch |
| **Shadow mode** | Test changes without affecting production | Before promotion |
| **Rollback** | Revert bad changes | On-demand |

**1. Ground Truth Labeling (False Positive Annotation):**

When an assertion fails, humans label whether it was a real issue or false alarm:

```bash
# View recent failures needing review
dql alerts suite.dql --unlabeled --days 7

# DATE        ASSERTION                        VALUE    THRESHOLD  STATUS
# 2024-12-20  orders.volume.min_rows           950      >= 1000    FAIL
# 2024-12-21  orders.completeness.email_null   6.2%     < 5%       FAIL

# Human investigates and labels
dql label suite.dql \
    --date 2024-12-20 \
    --assertion "orders.volume.min_rows" \
    --false-positive \
    --reason "Planned vendor maintenance"
```

| Alert Status | `ground_truth` | Meaning | RL Impact |
|--------------|----------------|---------|-----------|
| FAIL | `true` | True positive (real issue) | No penalty |
| FAIL | `false` | **False positive** (false alarm) | `-cost_fp` penalty |
| PASS | `true` | False negative (missed issue) | `-cost_fn` penalty |
| PASS | `false` | True negative (correct) | No penalty |

**2. Review Queue for Proposed Changes:**

```bash
# CLI for review workflow
dql review suite.dql --pending              # List pending changes
dql review suite.dql --approve <entry_id>   # Approve change
dql review suite.dql --reject <entry_id> --reason "too aggressive"
```

**3. Staged Deployment:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ RL Proposes │ ──▶ │   Pending   │ ──▶ │   Shadow    │ ──▶ │ Production  │
│   Change    │     │   Review    │     │    Mode     │     │    Live     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                     Human reviews        Runs in parallel,
                     and approves         alerts logged but
                                          not actioned
```

```bash
# Shadow mode: run proposed thresholds alongside production
dql run suite.dql --shadow suite_proposed.dql --date 2024-12-25

# Promote after validation
dql promote suite.dql --assertion "orders.volume.dod_stability"

# Rollback if needed
dql rollback suite.dql --to 2024-12-15
```

**4. Annotation for Human Review:**

```dql
check "Critical" on orders {
    @required
    @human_review                    # Any change requires approval
    assert null_count(customer_id) == 0
        name "orders.critical.customer_id"
        severity P0
}
```

| Annotation | Behavior |
|------------|----------|
| `@human_review` | Threshold changes go to review queue |
| `@auto_approve` | Algorithm changes apply immediately (low-risk) |

**Multi-Checkpoint Joint Training:**

DQX checks data at multiple points in the system. Joint RL training optimizes across all checkpoints by learning correlations from historical data — no explicit DAG required:

```python
class JointTrainingAgent:
    """RL agent that learns correlations across checkpoints."""

    def __init__(self, suites: list[str], db):
        self.suites = {s: Suite.load(s) for s in suites}

    def learn_correlations(self, days: int = 90):
        """Discover correlations from historical co-failures."""
        history = self.db.query(
            """
            SELECT date, assertion_name, passed FROM dql_results
            WHERE date > current_date - interval '{days} days'
        """
        )
        failure_matrix = self.pivot_failures(history)
        self.correlation_matrix = np.corrcoef(failure_matrix.T)

    def get_state(self) -> np.ndarray:
        """Combined state from all checkpoints."""
        params = []
        for suite in self.suites.values():
            params.extend([p.normalized_value for p in suite.get_tunable_params()])
        return np.array(params)

    def compute_reward(self, results: dict) -> float:
        """Reward considers learned correlations."""
        reward = 0.0
        for a in results["assertions"]:
            if not a["passed"] and a.get("is_fp"):
                reward -= a["cost"]["false_positive"]
            # Bonus for catching correlated failures early
            for corr_name, strength in self.get_correlated(a["name"]):
                if self.also_failed(results, corr_name):
                    reward += 5 * strength
        return reward
```

**Key insight:** Correlations are learned from failure patterns, not declared. The system discovers relationships automatically.

**Handling Partial Observations:**

In real systems, when one component fails, downstream checks often don't run:

```
Day 1: [ingestion ✓] → [transform ✓] → [serving ✓]    # Full run
Day 2: [ingestion ✗] → [transform ?] → [serving ?]    # Stopped early
Day 3: [ingestion ✓] → [transform ✗] → [serving ?]    # Partial run
```

**Solution 1: Run Status as Signal**

```python
def run_and_observe(self, target_date: date) -> dict:
    results = {"assertions": [], "run_status": {}}
    for name, suite in self.suites.items():
        try:
            result = suite.run(db=self.db, date=target_date)
            results["assertions"].extend(result.assertions)
            results["run_status"][name] = "completed"
        except UpstreamFailure:
            results["run_status"][name] = "skipped"
    return results


def get_state(self) -> np.ndarray:
    params = self.get_threshold_params()
    run_mask = [1.0 if s == "completed" else 0.0 for s in self.run_status.values()]
    return np.concatenate([params, run_mask])
```

**Solution 2: Per-Checkpoint Training**

```python
class PerCheckpointAgents:
    """Each checkpoint trains independently on days it ran."""

    def train_episode(self, target_date: date):
        for name, agent in self.agents.items():
            if self.did_run(name, target_date):
                agent.train_step(target_date)
```

**Solution 3: Reward Shaping for Early Stops**

```python
def compute_reward(self, results: dict) -> float:
    reward = 0.0
    for name, status in results["run_status"].items():
        if status == "completed":
            reward += self.assertion_rewards(results, name)
        elif status == "skipped":
            reward += 5  # Bonus: upstream caught issue early
    return reward
```

| Approach | When to use |
|----------|-------------|
| **Run status as signal** | Simple, works with existing data |
| **Per-checkpoint agents** | Loosely coupled checkpoints |
| **Reward shaping** | Always — "skipped" = early detection |

**Key insight:** "Skipped because upstream failed" is valuable signal — catching issues early is the desired behavior.

### Architecture

The interpreter walks the AST and calls DQX APIs directly:

```python
import sympy as sp


class Interpreter:
    """Executes DQL AST against DQX runtime."""

    # Whitelisted sympy functions
    MATH_FUNCTIONS = {
        "abs": sp.Abs,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "min": sp.Min,
        "max": sp.Max,
    }

    def __init__(self, db: Database, target_date: date):
        self.db = db
        self.target_date = target_date
        self.provider = MetricProvider(db)
        self.constants: dict[str, Any] = {}
        self.macros: dict[str, MacroNode] = {}

    def eval_metric_expr(self, expr_str: str) -> sp.Expr:
        """Parse metric expression using sympy."""
        namespace = {**self.MATH_FUNCTIONS}

        # Add metric functions that call provider methods
        for name in ["num_rows", "average", "sum", "null_count", ...]:
            namespace[name] = self._metric_wrapper(name)

        return sp.sympify(expr_str, locals=namespace, evaluate=False)

    def run(self, ast: SuiteNode, datasources: list) -> list[Result]:
        suite = VerificationSuite(ast.name, db=self.db)
        suite.data_av_threshold = ast.threshold

        # Register constants and macros
        for const in ast.constants:
            self.constants[const.name] = self.eval_expr(const.value)
        for macro in ast.macros:
            self.macros[macro.name] = macro

        # Build checks
        for check_node in ast.checks:
            check = self.build_check(check_node)
            suite.add_check(check, datasets=check_node.datasets)

        # Add profiles
        for profile_node in ast.profiles:
            suite.add_profile(self.build_profile(profile_node))

        # Execute
        key = ResultKey(self.target_date, tags={})
        suite.run(datasources, key)
        return suite.collect_results()
```

The interpreter owns the sympy namespace. When it parses `abs(day_over_day(average(price)))`:
1. `abs` resolves to `sp.Abs` from `MATH_FUNCTIONS`
2. `day_over_day`, `average` resolve to provider method wrappers
3. `sympify()` builds the expression tree

### Runtime Errors

The interpreter reports errors with source location:

```
error: assertion failed
  --> suite.dql:15:9
   |
15 |     assert null_count(customer_id) == 0  severity P0
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = metric value: 42
   = expected: == 0
   = dataset: orders
   = date: 2024-12-25
```

## Error Messages

The interpreter reports errors with file location and context:

```
error[E001]: unknown metric 'avg'
  --> suite.dql:12:16
   |
12 |     assert avg(price) > 0
   |            ^^^ did you mean 'average'?

error[E002]: duplicate assertion name
  --> suite.dql:18:9
   |
14 |         name "Order count"
   |              -------------- first defined here
18 |         name "Order count"
   |              ^^^^^^^^^^^^^^ duplicate
```

Warning for unnamed assertions:

```
warning[W001]: assertion has no name
  --> suite.dql:22:5
   |
22 |     assert num_rows() > 0
   |     ^^^^^^^^^^^^^^^^^^^^^ consider adding a descriptive name
```

## Grammar Reference

```ebnf
(* === Top-level === *)
suite       = "suite" STRING "{" suite_body "}"
suite_body  = (metadata | const | macro | check | profile | import)*

metadata    = "availability_threshold" PERCENT

(* === Constants and Macros === *)
const       = ["export"] "const" IDENT "=" expr [tunable]
tunable     = "tunable" "[" expr "," expr "]"
macro       = ["export"] "macro" IDENT "(" params ")" "{" macro_body "}"
params      = IDENT ("," IDENT)* ["..."]
macro_body  = (assertion | use | for_loop)+
for_loop    = "for" IDENT "in" IDENT "{" (assertion | use)+ "}"

(* === Checks and Assertions === *)
check       = "check" STRING "on" datasets "{" (annotation | assertion | use)+ "}"
datasets    = ident ("," ident)*
use         = "use" qualified_ident "(" [args] ")"

annotation  = "@" IDENT ["(" ann_args ")"]
ann_args    = IDENT "=" expr ("," IDENT "=" expr)*

assertion   = [annotation*] "assert" expr condition modifiers*
condition   = comparison | "between" expr "and" expr | "is" keyword
comparison  = ("<" | "<=" | ">" | ">=" | "==" | "!=") expr
keyword     = "positive" | "negative" | "None" | "not" "None"
modifiers   = name | tolerance | severity | tags
name        = "name" STRING
tolerance   = ("tolerance" | "+/-" | "±") NUMBER
severity    = "severity" SEVERITY
tags        = "tags" "[" IDENT ("," IDENT)* "]"

(* === Expressions === *)
expr        = ["-"] term (("+"|"-") term)*
term        = factor (("*"|"/") factor)*
factor      = NUMBER | PERCENT | call | "(" expr ")" | ident | "None"
call        = qualified_ident "(" [args] ")"
args        = arg ("," arg)*
arg         = expr | IDENT expr | "[" ident ("," ident)* "]"
qualified_ident = IDENT ("." IDENT)*

(* === Profiles === *)
profile     = "profile" STRING "{" profile_body "}"
profile_body= "type" IDENT "from" date_expr "to" date_expr rule*
date_expr   = DATE | date_func | date_expr ("+" | "-") NUMBER
date_func   = IDENT "(" [args] ")"
rule        = disable | scale | downgrade
disable     = "disable" ("check" STRING | "assertion" STRING "in" STRING)
scale       = "scale" selector "by" NUMBER "x"
downgrade   = "downgrade" selector "to" SEVERITY
selector    = "check" STRING | "tag" STRING

(* === Imports === *)
import      = "import" STRING ["as" IDENT]
            | "import" "{" IDENT ("," IDENT)* "}" "from" STRING

(* === Tokens === *)
STRING      = '"' (ESC | [^"\\])* '"'
ESC         = '\\' ["\\nrt]           (* \" \\ \n \r \t *)
NUMBER      = [0-9]+ ('.' [0-9]+)?
PERCENT     = NUMBER '%'
DATE        = [0-9]{4} '-' [0-9]{2} '-' [0-9]{2}
SEVERITY    = 'P0' | 'P1' | 'P2' | 'P3'
IDENT       = [a-zA-Z_] [a-zA-Z0-9_]*
ident       = IDENT | '`' [^`]+ '`'           (* backticks escape reserved words *)
COMMENT     = '#' [^\n]*                  (* Python-style comments *)
```

### Reserved Words

The following are reserved: `suite`, `check`, `assert`, `on`, `from`, `to`, `by`, `in`, `and`, `is`, `between`, `profile`, `type`, `macro`, `const`, `use`, `for`, `import`, `export`, `as`, `name`, `severity`, `tags`, `tolerance`, `scale`, `disable`, `downgrade`.

Use backticks to escape column or dataset names that conflict:

```dql
check "Test" on `from` {              # 'from' as dataset name
    assert count(`to`) > 0            # 'to' as column name
}
```

## Design Decisions

### Interpreter-Only Architecture

DQL runs directly via the interpreter. No compilation step. Metric expressions pass to sympy for parsing, enabling full compatibility with DQX's Python runtime.

Benefits:
- **Faster iteration** — No build step between edit and run
- **Better errors** — Messages point to DQL source, not generated code
- **Full sympy support** — All math functions (`abs`, `sqrt`, `log`, `exp`, `min`, `max`) work

### String-Based Expressions

DQL metric expressions are strings that pass directly to `sympy.sympify()`. This design:
- Matches DQX's existing `MetricExpressionParser`
- Enables arbitrary arithmetic without grammar changes
- Supports all whitelisted sympy functions

```dql
assert abs(day_over_day(average(price))) < 0.1   # sympy handles this
assert sqrt(variance(score)) < 10                 # no special grammar needed
```

### Validation Before Execution

The interpreter validates before running:

- Unknown metric functions
- Invalid severity levels
- Duplicate names
- Type mismatches

Errors surface immediately with source location.

### Macro Expansion

Macros expand before execution. The interpreter sees only concrete assertions. This simplifies debugging: no macro indirection at runtime.

## Implementation Status

### Implemented ✅

The following features from this design are **already implemented** in the DQX codebase:

#### Conditions

| Condition | Description | Location |
|-----------|-------------|----------|
| `!= N` | Not equal to N | ✅ `is_neq()` in `api.py` AssertionReady |
| `is None` | Value is None | ✅ `is_none()` in `api.py` AssertionReady |
| `is not None` | Value is not None | ✅ `is_not_none()` in `api.py` AssertionReady |

#### Utility Functions

| Function | Description | Location |
|----------|-------------|----------|
| `coalesce(expr, ...)` | Return first non-None value | ✅ `Coalesce` class in `functions.py` |

#### Metric Parameters

| Parameter | Description | Location |
|-----------|-------------|----------|
| `order_by` for `first()` | Sort before taking first value | ✅ `first()` in `provider.py` |

#### Annotations

| Annotation | Description | Location |
|------------|-------------|----------|
| `@experimental` | Mark algorithm-proposed assertions | ✅ `experimental` param in `api.py` AssertionDraft.where() |
| `@required` | Prevent removal by algorithms | ✅ `required` param in `api.py` AssertionDraft.where() |
| `@cost(fp, fn)` | False positive/negative costs for RL reward | ✅ `cost={"fp": N, "fn": M}` param in `api.py` AssertionDraft.where() |

#### Tunable Constants

| Feature | Description | Location |
|---------|-------------|----------|
| `tunable` constants | Bounded parameters for RL optimization | ✅ `TunableFloat`, `TunablePercent`, `TunableInt`, `TunableChoice` in `tunables.py` |
| `suite.get_tunable_params()` | List tunable params with bounds | ✅ `VerificationSuite.get_tunable_params()` in `api.py` |
| `suite.set_param(name, value)` | Modify threshold within bounds | ✅ `VerificationSuite.set_param()` in `api.py` |
| `suite.get_param_history(name)` | Get change history for tunable | ✅ `VerificationSuite.get_param_history()` in `api.py` |

### Not Yet Implemented ❌

The following features are specified in this design but **require implementation**:

#### DQL Core (Parser & Interpreter)

| Feature | Description | Priority |
|---------|-------------|----------|
| Lexer/Tokenizer | Tokenize DQL source into tokens | High |
| Parser | Parse tokens into AST nodes | High |
| AST Nodes | Suite, Check, Assertion, Profile, Macro, etc. | High |
| Interpreter | Execute AST against DQX runtime | High |
| Macro expansion | Expand macros before execution | Medium |
| Import system | `import`, `export`, path resolution | Medium |
| CLI (`dql run`, `dql check`) | Command-line interface | Medium |

#### RL Agent Integration

| Feature | Description | Priority |
|---------|-------------|----------|
| `Suite.load(path)` | Parse DQL into manipulable Suite object | High |
| `suite.save()` | Persist changes back to `.dql` file | Medium |

#### History & Auditing

| Feature | Description | Priority |
|---------|-------------|----------|
| `.dql.history` file | JSONL change log alongside DQL source | Medium |
| `History.log(entry)` | Append action records | Medium |
| `dql history` command | View change history | Low |
| `dql rollback --to DATE` | Revert to previous state | Low |
| `dql review` command | Approve/reject pending changes | Low |

#### Human-in-the-Loop

| Feature | Description | Priority |
|---------|-------------|----------|
| `@human_review` annotation | Changes require approval | Low |
| `@auto_approve` annotation | Low-risk changes apply immediately | Low |
| `dql label` command | Label false positives for reward tuning | Low |
| Shadow mode | Test proposed changes without production impact | Low |
| `dql promote` command | Promote experimental to production | Low |

## Future Extensions

### Planned

| Feature | Description | Example |
|---------|-------------|---------|
| **Row-level checks** | Assert condition on every row | `assert each row: price > 0` |
| **Named shortcuts** | Concise syntax for common patterns | `assert unique(customer_id)` |
| **Freshness checks** | Data recency validation | `assert freshness(updated_at) < 1 hour` |
| **Timeliness checks** | SLA compliance validation | `assert timeliness(partition_date) by 06:00 UTC` |
| **Schema validation** | Assert table structure | `assert column email exists` |

### Considered

| Feature | Description | Example |
|---------|-------------|---------|
| **Statistical bounds** | Anomaly detection via z-score | `assert average(price) within 3 stddev` |
| **Referential integrity** | Foreign key validation | `assert values(order_id) in values(id, dataset orders)` |
| **Distribution checks** | Categorical distribution matching | `assert distribution(status) matches {...}` |
| **Percentile metrics** | P50, P95, P99 calculations | `assert percentile(latency, 0.99) < 500` |
| **LSP server** | IDE autocomplete and diagnostics | — |

### Named Shortcuts (Planned)

Syntactic sugar for common data quality patterns:

```dql
assert unique(customer_id)              # duplicate_count([customer_id]) == 0
assert not_null(email)                  # null_count(email) == 0
assert not_null(email, phone, address)  # Multiple columns
assert positive(price)                  # minimum(price) > 0
assert non_negative(quantity)           # minimum(quantity) >= 0
assert in_set(status, ["A", "B", "C"])  # All values in allowed set
assert not_empty()                      # num_rows() > 0
```

### Freshness & Timeliness (Planned)

```dql
# Freshness: How old is the newest data?
assert freshness(updated_at) < 2 hours
    name "Data refreshed recently"

# Timeliness: Did data arrive on schedule?
assert timeliness(partition_date) by 06:00 UTC
    name "Daily load completed by SLA"

# Data lag: Time since partition date
assert data_lag(partition_date) < 18 hours
    name "Data available within 18 hours"
```

### Row-Level Checks (Planned)

```dql
# All rows must satisfy condition
assert each row: price > 0
    name "All prices positive"

assert each row: status in ["pending", "shipped", "delivered"]
    name "Valid status values"

# With percentage tolerance (inspired by Great Expectations "mostly" parameter)
assert 95% of rows: email matches "^[^@]+@[^@]+$"
    name "Most emails valid format"
```
