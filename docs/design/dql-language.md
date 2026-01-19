# DQL: A Language for Data Quality

## Goals

DQL (Data Quality Language) is designed with three goals:

1. **Simple, dedicated syntax** — A language built specifically for data quality checks, where intent is clear and boilerplate is minimal
2. **DQX engine integration** — Runs directly on the DQX runtime, leveraging its metric computation, profiling, and validation capabilities
3. **AI-native** — Structured and unambiguous enough for AI agents to read, write, and modify autonomously when investigating and resolving data quality issues

## Solution

DQL expresses data quality checks in a syntax built for the task. DQX validates and executes checks directly against your database.

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

A DQL file contains one suite. A suite contains checks and tunables.

```text
suite
├── metadata (name, threshold)
├── tunables
└── checks
    └── assertions
```

### Suite

Every DQL file begins with a suite declaration:

```dql
suite "E-Commerce Data Quality" {
    availability_threshold 80%

    # checks and tunables go here
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
    assert num_rows(dataset=returns) / num_rows(dataset=orders) < 15%
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
assert average(price) / average(price, lag=1) == 1.0 tolerance 0.05
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
| `sample` | No | Sample data before computing metric |

### Sampling

For large datasets, sample data before computing metrics:

```dql
assert average(price) > 0 sample 10%
    name "Price positive (10% sample)"

assert average(price) > 0 sample 10000 rows
    name "Price positive (10k rows)"
```

With seed for reproducibility:

```dql
assert average(price) > 0 sample 10% seed 42
    name "Price positive (reproducible)"
```

| Syntax | Meaning |
|--------|---------|
| `sample N%` | Random sample of N percent of rows |
| `sample N rows` | Random sample of N rows |
| `seed M` | Random seed for reproducibility |

**Database compatibility:**

| Database | Method |
|----------|--------|
| DuckDB | `USING SAMPLE N%` |
| Databricks | `TABLESAMPLE (N PERCENT)` |
| PostgreSQL | `TABLESAMPLE BERNOULLI(N)` |
| BigQuery | `WHERE RAND() < N/100` |

**Semantics:**
- Sampling applies to the entire metric expression
- Without `seed`, results are non-deterministic across runs
- Row-based sampling (`N rows`) uses reservoir sampling for exact counts

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
first(timestamp, order_by=price) # First value ordered by price
```

`first()` returns the first value in storage order unless `order_by` specifies a column.

**Warning:** Without `order_by`, results are non-deterministic for distributed storage (Parquet, Delta Lake). Row order depends on file layout and query execution. Always use `order_by` for reproducible results.

**Custom SQL** executes arbitrary expressions:

```dql
sql("SUM(amount) / COUNT(*)")
```

**`sql()` limitations:**

- **No validation** — The interpreter cannot verify column names, syntax, or types
- **No interpolation** — Constants cannot be inserted into SQL strings (prevents SQL injection)
- **Dialect-specific** — SQL syntax varies across databases; DQL does not translate

```dql
# Preferred: interpreter validates
assert sum(amount) / num_rows() > 10

# Escape hatch: no validation
assert sql("SUM(amount) / COUNT(*)") > 10

# ERROR: This example shows unsupported syntax for illustration
# DQL does not support string interpolation in sql() functions
tunable MY_COL = "amount" bounds ["amount", "revenue"]  # ERROR: String interpolation not supported
assert sql("SUM({MY_COL})") > 0  # ERROR: Interpolation in sql() not permitted
```

Use built-in metrics when possible; reserve `sql()` for expressions DQL cannot represent.

### Parameters

Metrics accept optional parameters:

```dql
average(price, lag=1)           # Yesterday's average
average(price, dataset=orders)  # Specific dataset
```

| Parameter | Description |
|-----------|-------------|
| `lag` | Calendar days offset (1 = yesterday, 7 = one week ago) |
| `dataset` | Restrict metric to named dataset |

**Lag semantics:** The `lag` parameter offsets by calendar days, not partition offsets. For a weekly-partitioned table queried on Monday with `lag 1`, VerificationSuite computes the metric for Sunday's data (which may fall in the previous week's partition).

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

## Tunables

Tunables define reusable values with explicit bounds for algorithmic optimization. All tunables require bounds specification to enable RL-based threshold tuning.

Declare tunables at suite level:

```dql
suite "Orders" {
    tunable NULL_THRESHOLD = 5% bounds [0%, 20%]
    tunable MIN_ORDERS = 1000 bounds [100, 10000]
    tunable VARIANCE_LIMIT = 0.5 bounds [0.1, 1.0]

    check "Completeness" on orders {
        assert null_count(email) / num_rows() < NULL_THRESHOLD
        assert num_rows() >= MIN_ORDERS
    }
}
```

### Syntax

```dql
tunable <NAME> = <value> bounds [<min>, <max>]
```

| Element | Required | Description |
|---------|----------|-------------|
| `tunable` | Yes | Keyword to declare a tunable parameter |
| `<NAME>` | Yes | Identifier for the tunable (uppercase convention) |
| `<value>` | Yes | Initial/default value |
| `bounds` | Yes | Keyword introducing the bounds specification |
| `[<min>, <max>]` | Yes | Valid range for optimization (inclusive) |

**Semantics:**
- Algorithms can modify tunables within their declared bounds
- Bounds are always required - no implicit or unbounded tunables
- Bounds are inclusive: `[0%, 20%]` allows values from 0% to 20%
- Initial value must be within the specified bounds

**Examples:**

```dql
# Percentage tunables
tunable MAX_NULL_RATE = 5% bounds [0%, 20%]

# Integer tunables
tunable MIN_DAILY_ORDERS = 1000 bounds [100, 10000]

# Decimal tunables
tunable VARIANCE_THRESHOLD = 0.3 bounds [0.0, 1.0]
```

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

## Profiles

## Profiles (Removed in v0.6.0)

**Note:** Profile definitions are no longer part of DQL syntax as of v0.6.0.

Profiles are now defined in:
- **YAML configuration files** (recommended for most use cases)
- **Python API** using `SeasonalProfile` class (for programmatic control)

See the [Migration Guide](../migration/dql-profiles-to-yaml.md) for converting existing DQL profiles to YAML or Python API.

**Why the change?** Profiles define runtime behavior (when/how to modify checks), which is conceptually separate from validation logic (what to check). This separation:
- Enables environment-specific configuration without modifying DQL
- Allows profile reuse across multiple suites
- Follows the same pattern as tunables
- Makes DQL simpler and more focused on validation logic

**YAML Example:**
```yaml
profiles:
  - name: "Holiday Season"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Volume"
      - action: "scale"
        target: "tag"
        name: "reconciliation"
        multiplier: 1.5
```

**Python API Example:**
```python
from datetime import date
from dqx.profiles import SeasonalProfile, check, tag

holiday = SeasonalProfile(
    name="Holiday Season",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[
        check("Volume").disable(),
        tag("reconciliation").set(metric_multiplier=1.5),
    ],
)

suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    profiles=[holiday],
)
```

For full documentation on profiles, see [Profiles Design](profiles.md).

## Complete Example

```dql
suite "E-Commerce Data Quality" {
    availability_threshold 80%

    tunable MAX_NULL_RATE = 5% bounds [0%, 20%]
    tunable MIN_ORDERS = 1000 bounds [100, 10000]

    check "Completeness" on orders {
        assert null_count(customer_id) == 0
            name "No null customer IDs"
            severity P0

        assert null_count(email) / num_rows() < MAX_NULL_RATE
            name "Email null rate below threshold"

        assert null_count(phone) / num_rows() < MAX_NULL_RATE
            name "Phone null rate below threshold"
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
        assert num_rows(dataset=returns) / num_rows(dataset=orders) < 15%
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

    # Note: Profiles are now defined in YAML configuration or via Python API
    # See "Profiles (Removed)" section below for migration details
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
| `set_param` | Tunable parameter changed | `param`, `old`, `new`, `agent`, `reason` |

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

```text
┌─────────────────────────────────────────────────────────────┐
│                      DQL Source                             │
│                     (suite.dql)                             │
└──────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ VerificationSuite    │
                    │ (parses & executes)  │
                    │    (via sympy)       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    DQX Runtime       │
                    │    (database)        │
                    └──────────────────────┘
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

Execute DQL through VerificationSuite:

```python
from dqx.api import VerificationSuite
from dqx.common import ResultKey
from datetime import date
from pathlib import Path

suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    config=Path("config.yaml"),  # Optional: load tunables + profiles
)

suite.run(datasources, ResultKey(date.today(), {}))
results = suite.collect_results()

for r in results:
    print(f"{r.check}/{r.assertion_name}: {r.status}")
```

Or use inline DQL source:

```python
dql_source = """
suite "Quick Check" {
    check "Basic" on orders {
        assert num_rows() > 0
            name "Has rows"
    }
}
"""

suite = VerificationSuite(
    dql=dql_source,
    db=db,
)

key = ResultKey(date.today(), {})
suite.run(datasources, key)
results = suite.collect_results()
```

### RL Agent Integration

DQL provides a programmatic API for reinforcement learning agents to optimize data quality checks:

1. Read tunable parameters and their bounds
2. Modify thresholds within bounds
3. Run checks and observe outcomes
4. Compute rewards from results and costs
5. Log changes to history

**Action Space:**

The RL agent operates on a continuous, bounded action space derived from tunable parameters:

| Component | Type | Description |
|-----------|------|-------------|
| **Threshold adjustments** | Continuous, bounded | One dimension per `tunable` parameter |

```python
# Continuous action space derived from tunable parameters
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
    tunable NULL_THRESHOLD = 5% bounds [0%, 20%]
    tunable MIN_ORDERS = 1000 bounds [100, 10000]
    tunable DOD_LIMIT = 0.5 bounds [0.1, 1.0]

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
    from 2024-12-20
    to   2024-12-31
    scale tag "volume" by 2.0
}
```

```python
# On holiday: actual rows = 600
# Multiplier 2.0 applied: 600 * 2.0 = 1200
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

VerificationSuite parses the DQL AST and builds Python check functions dynamically:

```python
# DQL parsing happens in VerificationSuite.__init__()
suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
)

# Internally:
# 1. Parse DQL file to AST
# 2. Extract tunables and create Tunable objects
# 3. Build check functions from Check AST nodes
# 4. Each check function evaluates metric expressions using sympy
# 5. Assertions are registered in the dependency graph
```

VerificationSuite uses sympy to parse metric expressions. When it parses `abs(day_over_day(average(price)))`:
1. `abs` resolves to `sp.Abs` from the metric namespace

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
suite_body  = (metadata | tunable | check)*

metadata    = "availability_threshold" PERCENT

(* === Tunables === *)
tunable     = "tunable" IDENT "=" expr "bounds" "[" expr "," expr "]"

(* === Checks and Assertions === *)
check       = "check" STRING "on" datasets "{" assertion+ "}"
datasets    = ident ("," ident)*

annotation  = "@" IDENT ["(" ann_args ")"]
ann_args    = IDENT "=" (NUMBER | STRING) ("," IDENT "=" (NUMBER | STRING))*

assertion   = annotation* "assert" expr condition modifiers*
condition   = comparison | "between" bound "and" bound | "is" keyword
comparison  = ("<" | "<=" | ">" | ">=" | "==" | "!=") expr
bound       = bound_term (("*"|"/") bound_term)*   (* restricted to avoid 'and' ambiguity *)
bound_term  = NUMBER | PERCENT | ident | call
keyword     = "positive" | "negative" | "None" | "not" "None"
modifiers   = name | tolerance | severity | tags | sample
name        = "name" STRING
tolerance   = ("tolerance" | "+/-" | "±") NUMBER
severity    = "severity" SEVERITY
tags        = "tags" "[" IDENT ("," IDENT)* "]"
sample      = "sample" (PERCENT | NUMBER "rows") ["seed" NUMBER]

(* === Expressions === *)
expr        = term (("+"|"-") term)*
term        = factor (("*"|"/") factor)*
factor      = "-" factor | NUMBER | PERCENT | call | "(" expr ")" | ident | "None" | STRING
call        = qualified_ident "(" [args] ")"
args        = arg ("," arg)*
arg         = named_arg | list_arg | expr
named_arg   = "lag" "=" NUMBER | "dataset" "=" ident | "order_by" "=" ident | "n" "=" NUMBER
list_arg    = "[" ident ("," ident)* "]"
qualified_ident = IDENT ("." IDENT)*

(* === Tokens === *)
STRING      = '"' (ESC | [^"\\])* '"'
ESC         = '\\' ["\\nrt]           (* \" \\ \n \r \t *)
NUMBER      = [0-9]+ ('.' [0-9]+)?
PERCENT     = NUMBER '%'
SEVERITY    = 'P0' | 'P1' | 'P2' | 'P3'
IDENT       = [a-zA-Z_] [a-zA-Z0-9_]*
ident       = IDENT | '`' [^`]+ '`'           (* backticks escape reserved words *)
COMMENT     = '#' [^\n]*                  (* Python-style comments *)
```

### Grammar Notes

**Named arguments:** Function arguments use specific keywords (`lag`, `dataset`, `order_by`, `n`) rather than arbitrary identifiers to avoid parser ambiguity.

**Between bounds:** The `between A and B` condition restricts bounds to simple expressions (numbers, percentages, identifiers, function calls, and `*`/`/` operators). Full arithmetic with `+`/`-` would conflict with the `and` keyword. Use parenthesized comparisons for complex bounds: `>= (A + B)`.

**Annotation values:** Annotation arguments accept only NUMBER or STRING values, not full expressions.

### Reserved Words

The following are reserved: `suite`, `check`, `assert`, `on`, `from`, `to`, `by`, `in`, `and`, `is`, `between`, `profile`, `type`, `tunable`, `bounds`, `name`, `severity`, `tags`, `tolerance`, `scale`, `disable`, `set`, `sample`, `seed`, `rows`, `lag`, `dataset`, `order_by`, `n`.

Use backticks to escape column or dataset names that conflict:

```dql
check "Test" on `from` {              # 'from' as dataset name
    assert count(`to`) > 0            # 'to' as column name
}
```

## Design Decisions

### Direct Execution Architecture

DQL is parsed and executed directly by VerificationSuite. No compilation step. Metric expressions pass to sympy for parsing, enabling full compatibility with DQX's Python runtime.

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

DQX validates before running:

- Unknown metric functions
- Invalid severity levels
- Duplicate names
- Type mismatches

Errors surface immediately with source location.

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

#### DQL Core (Parser & Execution)

| Feature | Description | Priority |
|---------|-------------|----------|
| Lexer/Tokenizer | Tokenize DQL source into tokens | High |
| Parser | Parse tokens into AST nodes | High |
| AST Nodes | Suite, Check, Assertion, Profile, etc. | High |
| ~~Interpreter~~ ✅ VerificationSuite execution | Execute AST against DQX runtime (implemented in VerificationSuite) | ~~High~~ ✅ Complete |
| Sampling | `sample N%` / `sample N rows` with optional `seed` | Medium |
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
| **Referential integrity** | Foreign key validation | `assert values(order_id) in values(id, dataset=orders)` |
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
