# DQL-VerificationSuite Integration Refactoring

## Overview

This document describes the refactoring to integrate DQL (Data Quality Language) parsing directly into the `VerificationSuite` class and move profile configuration out of DQL syntax into external YAML configuration and Python API parameters.

### Current State

DQX currently has two separate ways to define data quality checks:

1. **Python API** (`VerificationSuite`): Define checks using Python functions decorated with `@check`
2. **DQL Language** (`Interpreter`): Write checks in DQL syntax, which gets parsed and converted to Python checks

The `Interpreter` class acts as a separate layer that:
- Parses `.dql` files into an Abstract Syntax Tree (AST)
- Converts DQL checks into Python check functions
- Creates a `VerificationSuite` instance with those checks
- Executes the suite and returns results

Additionally, profiles (date-based behavior overrides) can be defined in two places:
- Inside DQL files using `profile` blocks
- In Python code using `SeasonalProfile` objects passed to `VerificationSuite`

### Goals

1. **Simplify the API**: Allow `VerificationSuite` to accept DQL programs directly without the `Interpreter` middleman
2. **Separate concerns**: Remove profiles from DQL syntax; treat them as runtime configuration separate from validation logic
3. **Unified configuration**: Load profiles from YAML config files (like tunables) or pass them via Python API
4. **Cleaner architecture**: DQL defines "what to validate," config/API defines "how to modify behavior"

### Non-Goals

- Maintaining backward compatibility with DQL files containing profiles (breaking change accepted)
- Keeping the `Interpreter` class as a public API
- Supporting profile definitions in DQL syntax

---

## Background

### What is DQL?

DQL (Data Quality Language) is a domain-specific language for defining data quality checks:

```dql
suite "Order Validation" {
    check "Completeness" on orders {
        assert null_count(customer_id) == 0
            name "Customer ID must not be null"
            severity P0
    }
}
```

### What are Profiles?

Profiles are date-based behavioral overrides that modify assertion execution. For example, a "Holiday Season" profile might:
- Disable volume checks during low-activity periods
- Scale expected metrics by a multiplier (e.g., 2x traffic during Black Friday)
- Override assertion severity levels

**Current DQL syntax:**
```dql
profile "Holiday Season" {
    from 2024-12-20
    to 2025-01-05

    disable check "Volume"
    scale tag "reconciliation" by 1.5
}
```

**Proposed YAML config:**
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

### What are Tunables?

Tunables are adjustable parameters in data quality checks that can be tuned by AI agents or operators:

```dql
tunable MAX_NULL_RATE = 1% bounds [0%, 5%]
```

They already support YAML configuration loading. This refactoring extends the same pattern to profiles.

### Why Separate Profiles from DQL?

1. **Separation of concerns**: DQL files define validation logic (what to check), profiles define runtime behavior (when/how to modify checks)
2. **Environment flexibility**: Different environments (dev/staging/prod) can use the same DQL with different profile configurations
3. **Easier maintenance**: Profile dates and multipliers change more frequently than validation logic
4. **Consistency**: Tunables are already loaded from YAML; profiles should follow the same pattern
5. **Reusability**: Share profiles across multiple DQL suites

---

## Proposed Architecture

### Before (Current)

```text
┌──────────┐
│ DQL File │ (contains checks + profiles + tunables)
└────┬─────┘
     │
     v
┌────────────┐
│ Interpreter│ (parses DQL, builds checks)
└────┬───────┘
     │
     v
┌──────────────────┐
│VerificationSuite │ (executes checks)
└──────────────────┘
```

### After (Proposed)

```text
┌──────────┐  ┌────────────┐
│ DQL File │  │ YAML Config│ (profiles + tunables)
│(checks)  │  │            │
└────┬─────┘  └────┬───────┘
     │             │
     └─────┬───────┘
           v
    ┌──────────────────┐
    │VerificationSuite │ (parses DQL, loads config, executes)
    └──────────────────┘
```

### API Changes

**Old way (Interpreter):**
```python
from dqx.dql import Interpreter

interp = Interpreter(db=db)
results = interp.run(Path("suite.dql"), datasources, date.today())
```

**New way (VerificationSuite with DQL):**
```python
from dqx import VerificationSuite
from pathlib import Path

suite = VerificationSuite(
    dql=Path("suite.dql"),  # NEW: DQL file path (suite name defined in DQL)
    db=db,
    config=Path("config.yaml"),  # Loads profiles + tunables
)
suite.run(datasources, key)
results = suite.collect_results()
```

**Alternative with programmatic profiles:**
```python
from dqx import VerificationSuite, SeasonalProfile, tag
from datetime import date

holiday = SeasonalProfile(
    name="Holiday Season",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[tag("reconciliation").set(metric_multiplier=1.5)],
)

suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    profiles=[holiday],  # Pass profiles programmatically
)
```

**Existing Python API (unchanged):**
```python
from dqx import VerificationSuite, check

@check(name="Completeness", datasets=["orders"])
def completeness_check(mp, ctx):
    ctx.assert_that(mp.null_count("customer_id")).where(
        name="Customer ID not null"
    ).is_eq(0)

suite = VerificationSuite(
    checks=[completeness_check],  # Still works!
    db=db,
    name="My Suite",
)
```

---

## Implementation Plan

### Phase 1: Profile Configuration Infrastructure

#### 1.1 Extend YAML Config for Profiles

**File:** `src/dqx/config.py`

Add profile loading from YAML config files with schema validation.

**YAML Schema:**
```yaml
# config.yaml
tunables:
  MAX_NULL_RATE: 0.02
  MIN_DAILY_TRANSACTIONS: 15000

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

      - action: "set_severity"
        target: "tag"
        name: "fraud"
        severity: "P0"
```

**Schema Definition:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Profile identifier |
| `type` | string | Yes | Must be "seasonal" |
| `start_date` | string | Yes | ISO 8601 date (YYYY-MM-DD) |
| `end_date` | string | Yes | ISO 8601 date (YYYY-MM-DD) |
| `rules` | array | Yes | List of rule objects |

**Rule Schema:**

| Field | Type | Required | Valid Values | Notes |
|-------|------|----------|--------------|-------|
| `action` | string | Yes | "disable", "scale", "set_severity" | |
| `target` | string | Yes | "check", "tag" | "assertion" not supported in YAML (use check instead) |
| `name` | string | Yes | - | Check or tag name to match |
| `multiplier` | float | Conditional | > 0 | Required for "scale" action |
| `severity` | string | Conditional | "P0", "P1", "P2", "P3" | Required for "set_severity" action |

**Implementation tasks:**
- Add `load_profiles_from_config(config_data: dict[str, Any]) -> list[SeasonalProfile]` function
- Parse YAML profiles section and convert to `SeasonalProfile` objects
- Create `Rule` objects from YAML rule dicts
- Validate date format (ISO 8601: YYYY-MM-DD) using `datetime.date.fromisoformat()`
- Validate required fields and action-specific parameters
- Raise `DQXError` with clear messages for invalid configuration
- Add unit tests for YAML profile parsing (valid/invalid cases)

**Example implementation:**
```python
def load_profiles_from_config(config_data: dict[str, Any]) -> list[SeasonalProfile]:
    """Load profiles from YAML config dictionary.

    Args:
        config_data: Parsed YAML config dictionary

    Returns:
        List of SeasonalProfile objects

    Raises:
        DQXError: If profile configuration is invalid
    """
    profiles_data = config_data.get("profiles", [])
    profiles = []

    for profile_dict in profiles_data:
        # Validate required fields
        # Parse dates
        # Build rules
        # Create SeasonalProfile
        profiles.append(profile)

    return profiles
```

#### 1.2 Update VerificationSuite Constructor

**File:** `src/dqx/api.py`

**New signature:**
```python
class VerificationSuite:
    def __init__(
        self,
        checks: Sequence[CheckProducer | DecoratedCheck] | None = None,
        dql: str | Path | None = None,  # NEW
        db: MetricDB,
        name: str,
        log_level: int | str = logging.INFO,
        data_av_threshold: float = 0.9,
        profiles: Sequence[Profile] | None = None,
        config: Path | None = None,
    ) -> None:
        """Initialize a VerificationSuite.

        Args:
            checks: Python check functions (mutually exclusive with dql)
            dql: DQL source string or Path to .dql file (mutually exclusive with checks)
            db: Metric database for storing/retrieving metrics
            name: Suite name (overridden by DQL suite name if dql provided)
            log_level: Logging level
            data_av_threshold: Minimum data availability threshold
            profiles: Profile objects to apply (merged with config profiles)
            config: Path to YAML config file (loads tunables + profiles)

        Raises:
            DQXError: If both or neither of checks/dql provided
        """
```

**Implementation logic:**
```python
def __init__(self, ...):
    # Validation: exactly one of checks or dql must be provided
    if (checks is None) == (dql is None):
        raise DQXError(
            "Exactly one of 'checks' or 'dql' must be provided. "
            "Use 'checks' for Python API or 'dql' for DQL files."
        )

    # Load config if provided (tunables + profiles)
    config_profiles = []
    if config is not None:
        logger.info(f"Loading configuration from {config}")
        config_data = load_config(config)

        # Load profiles from config (NEW)
        config_profiles = load_profiles_from_config(config_data)
        if config_profiles:
            logger.info(f"Loaded {len(config_profiles)} profile(s) from config")

    # Merge profiles: config + API parameter (both active)
    all_profiles = list(config_profiles)
    if profiles is not None:
        all_profiles.extend(profiles)
    self._profiles = all_profiles

    # Parse DQL if provided
    if dql is not None:
        from dqx.dql.parser import parse, parse_file

        if isinstance(dql, Path):
            suite_ast = parse_file(dql)
        else:
            suite_ast = parse(dql, filename=name)

        # Extract suite properties (DQL takes precedence)
        self._name = suite_ast.name if suite_ast.name else name.strip()
        self._data_av_threshold = suite_ast.availability_threshold or data_av_threshold

        # Build checks from DQL
        logger.info("Building checks from DQL...")
        self._checks = self._build_dql_checks(suite_ast)

        # Apply tunables from config AFTER DQL parsing
        # (DQL defines tunables, config provides values)
        if config is not None:
            apply_tunables_from_config(config_data, self._tunables)
    else:
        # Existing checks-based path
        self._checks = checks
        self._name = name.strip()
        self._data_av_threshold = data_av_threshold

    # ... rest of existing initialization ...
```

**Implementation tasks:**
- Add `dql` parameter to constructor signature
- Make `checks` parameter optional (default `None`)
- Add mutual exclusion validation for `checks` and `dql`
- Add profile loading and merging logic
- Add DQL parsing branch
- Update docstring with new parameters and examples
- Add error messages for invalid usage
- Maintain backward compatibility for Python API (checks parameter)

---

### Phase 2: Remove Profiles from DQL

#### 2.1 Update DQL Grammar

**File:** `src/dqx/dql/grammar.lark`

**Remove these grammar rules:**
```lark
// REMOVE: Profile definitions
profile: "profile" STRING "{" profile_body "}"
profile_body: ("from" date_expr)? ("to" date_expr)? profile_rule*

// REMOVE: Profile rules
profile_rule: disable_rule | scale_rule | set_severity_rule
disable_rule: "disable" ("check" | "assertion") STRING ("in" STRING)?
scale_rule: "scale" ("check" | "tag") STRING "by" NUMBER
set_severity_rule: "set" "severity" ("check" | "tag") STRING "to" SEVERITY

// REMOVE: Date expressions (only used by profiles)
date_expr: DATE ("+" NUMBER | "-" NUMBER)?
```

**Update suite statement:**
```lark
suite: "suite" STRING "{" suite_body "}"
suite_body: suite_statement*
suite_statement: tunable | check  // REMOVED: | profile
```

**Implementation tasks:**
- Remove all profile-related grammar rules
- Remove `profile` from `suite_statement` alternatives
- Remove `date_expr` rule and `DATE` token if not used elsewhere
- Update grammar comments if any reference profiles
- Verify grammar still parses correctly with tests

#### 2.2 Update AST

**File:** `src/dqx/dql/ast.py`

**Remove these dataclasses:**
```python
# REMOVE these classes entirely:

@dataclass(frozen=True)
class Profile:
    """A profile that modifies assertions during a date range."""
    name: str
    from_date: DateExpr
    to_date: DateExpr
    rules: tuple[Rule, ...]
    loc: SourceLocation | None = None

@dataclass(frozen=True)
class DisableRule:
    """Disable a check or assertion."""
    target_type: str  # "check" or "assertion"
    target_name: str
    in_check: str | None = None
    loc: SourceLocation | None = None

@dataclass(frozen=True)
class ScaleRule:
    """Scale a check or tag by a multiplier."""
    selector_type: str  # "check" or "tag"
    selector_name: str
    multiplier: float
    loc: SourceLocation | None = None

@dataclass(frozen=True)
class SetSeverityRule:
    """Set severity for a selector."""
    selector_type: str  # "check" or "tag"
    selector_name: str
    severity: Severity
    loc: SourceLocation | None = None

@dataclass(frozen=True)
class DateExpr:
    """A date expression with optional day offset."""
    value: date
    offset: int = 0
    loc: SourceLocation | None = None

# REMOVE type alias:
Rule = DisableRule | ScaleRule | SetSeverityRule
```

**Update Suite dataclass:**
```python
@dataclass(frozen=True)
class Suite:
    """Top-level suite containing checks and tunables."""
    name: str
    checks: tuple[Check, ...]
    tunables: tuple[Tunable, ...]
    # REMOVED: profiles: tuple[Profile, ...] = ()
    availability_threshold: float | None = None
    loc: SourceLocation | None = None
```

**Implementation tasks:**
- Delete `Profile`, `DisableRule`, `ScaleRule`, `SetSeverityRule`, `DateExpr` classes
- Delete `Rule` type alias
- Remove `profiles` field from `Suite` dataclass
- Update module docstring if it mentions profiles
- Run type checker to catch any remaining references

#### 2.3 Update Parser

**File:** `src/dqx/dql/parser.py`

**Remove these transformer methods:**
```python
# REMOVE these methods entirely:

def profile(self, tree: Any) -> Profile:
    """Transform profile rule."""
    ...

def profile_rule(self, tree: Any) -> Rule:
    """Transform a profile rule (disable/scale/set_severity)."""
    ...

def disable_rule(self, tree: Any) -> DisableRule:
    """Transform disable rule."""
    ...

def scale_rule(self, tree: Any) -> ScaleRule:
    """Transform scale rule."""
    ...

def set_severity_rule(self, tree: Any) -> SetSeverityRule:
    """Transform set_severity rule."""
    ...

def date_expr(self, tree: Any) -> DateExpr:
    """Transform date expression."""
    ...
```

**Update suite transformer:**
```python
def suite(self, tree: Any) -> Suite:
    """Transform suite declaration."""
    name = None
    checks = []
    tunables = []
    # REMOVED: profiles = []
    availability_threshold = None

    for item in tree.children:
        if isinstance(item, Token) and item.type == "STRING":
            name = self._unquote_string(item.value)
        elif hasattr(item, "data"):
            if item.data == "availability_threshold":
                availability_threshold = self._extract_threshold(item)
        elif isinstance(item, Check):
            checks.append(item)
        elif isinstance(item, Tunable):
            tunables.append(item)
        # REMOVED: elif isinstance(item, Profile):
        #             profiles.append(item)

    return Suite(
        name=name,
        checks=tuple(checks),
        tunables=tuple(tunables),
        # REMOVED: profiles=tuple(profiles),
        availability_threshold=availability_threshold,
        loc=self._source_location(tree),
    )
```

**Remove imports:**
```python
from dqx.dql.ast import (
    Annotation,
    Assertion,
    Check,
    Collection,
    # REMOVED: DateExpr,
    # REMOVED: DisableRule,
    Expr,
    # REMOVED: Profile,
    # REMOVED: Rule,
    # REMOVED: ScaleRule,
    # REMOVED: SetSeverityRule,
    Severity,
    SourceLocation,
    Suite,
    Tunable,
)
```

**Implementation tasks:**
- Delete all profile-related transformer methods
- Update `suite()` transformer to not collect profiles
- Remove profile-related imports from `ast` module
- Remove any helper methods only used by profile transformers
- Verify parser tests still pass

#### 2.4 Update Interpreter

**File:** `src/dqx/dql/interpreter.py`

The `Interpreter` class becomes much simpler since it no longer handles profiles.

**Remove these methods:**
```python
# REMOVE these methods entirely:

def _activate_profiles(self, profiles_ast: tuple[Profile, ...], execution_date: date) -> None:
    """Determine which profiles are active for execution date."""
    ...

def _resolve_date_expr(self, date_expr: DateExpr, execution_date: date) -> date:
    """Resolve DQL date expression to concrete date."""
    ...

def _apply_profile_scaling(self, metric_value: Any, statement_ast: Assertion | Collection) -> Any:
    """Apply scale multipliers from active profiles."""
    ...

def _is_disabled(self, statement_ast: Assertion | Collection) -> bool:
    """Check if assertion or collection is disabled by any active profile."""
    ...

def _resolve_severity(self, statement_ast: Assertion | Collection) -> Severity:
    """Resolve severity with profile overrides."""
    ...

def _rule_matches_statement(self, rule: Rule, statement_ast: Assertion | Collection) -> bool:
    """Check if a profile rule applies to an assertion or collection."""
    ...
```

**Remove instance variables:**
```python
class Interpreter:
    def __init__(self, db: MetricDB):
        self.db = db
        self.tunables: dict[str, float] = {}
        # REMOVED: self.active_profiles: list[Profile] = []
        # REMOVED: self.current_check_name: str | None = None
        self.datasources: dict[str, SqlDataSource] = {}
```

**Simplify `_execute()` method:**
```python
def _execute(
    self,
    suite_ast: Suite,
    datasources: Mapping[str, SqlDataSource],
    execution_date: date,
    tags: dict[str, str] | None,
) -> SuiteResults:
    """Execute a parsed suite AST."""
    self.datasources = dict(datasources)

    # Validate datasources match what DQL expects
    self._validate_datasources(suite_ast, datasources)

    # REMOVED: self._activate_profiles(suite_ast.profiles, execution_date)

    # Build DQX VerificationSuite from AST
    suite = self._build_suite(suite_ast)

    # Execute the suite
    tags_dict = tags if tags is not None else {}
    key = ResultKey(execution_date, tags_dict)
    suite.run(list(datasources.values()), key)

    # Collect and return results
    return self._collect_results(suite, suite_ast.name, execution_date)
```

**Simplify `_setup_assertion_ready()` method:**
```python
def _setup_assertion_ready(
    self,
    statement: Assertion | Collection,
    mp: MetricProvider,
    ctx: Context
) -> Any | None:
    """Common setup for assertions and collections.

    Returns AssertionReady object.
    """
    # REMOVED: if self._is_disabled(statement): return None

    # Evaluate metric expression
    metric_value = self._eval_metric_expr(statement.expr, mp)

    # REMOVED: metric_value = self._apply_profile_scaling(metric_value, statement)
    # REMOVED: severity = self._resolve_severity(statement)

    # Always use original severity (no profile override)
    severity = statement.severity

    # Validate name is present
    if statement.name is None:  # pragma: no cover
        raise DQLError("Statement must have a name", loc=statement.loc)

    # Extract cost annotation
    cost_annotation = self._get_cost_annotation(statement)
    cost_dict = {k: float(v) for k, v in cost_annotation.items()} if cost_annotation else None

    # Build and return assertion ready object
    return ctx.assert_that(metric_value).where(
        name=statement.name,
        severity=severity.value,
        tags=set(statement.tags),
        experimental=self._has_annotation(statement, "experimental"),
        required=self._has_annotation(statement, "required"),
        cost=cost_dict,
    )
```

**Update imports:**
```python
from dqx.dql.ast import (
    Assertion,
    Check,
    Collection,
    # REMOVED: DateExpr,
    # REMOVED: DisableRule,
    Expr,
    # REMOVED: Profile,
    # REMOVED: Rule,
    # REMOVED: ScaleRule,
    # REMOVED: SetSeverityRule,
    Severity,
    Suite,
    Tunable,
)
```

**Remove `# === Profile Logic ===` section:**
The entire profile logic section (lines ~694-778) can be deleted.

**Implementation tasks:**
- Delete all profile-related methods
- Remove profile-related instance variables from `__init__()`
- Remove profile activation from `_execute()`
- Simplify `_setup_assertion_ready()` to always use original severity
- Remove profile-related imports
- Update docstrings that mention profiles
- Verify interpreter tests still pass

---

### Phase 3: Integrate DQL into VerificationSuite

#### 3.1 Add DQL Parsing Helper Methods

**File:** `src/dqx/api.py`

Add these private methods to `VerificationSuite` class. These methods move logic from `Interpreter` into `VerificationSuite`.

```python
def _build_dql_checks(self, suite_ast: Suite) -> list[CheckProducer]:
    """Convert DQL Suite AST to Python check functions.

    Args:
        suite_ast: Parsed DQL Suite AST

    Returns:
        List of check functions compatible with VerificationSuite
    """
    # Build tunables for substitution in expressions
    tunables_dict = self._build_tunables_from_ast(suite_ast.tunables)

    # Build each check
    checks = []
    for check_ast in suite_ast.checks:
        check_func = self._build_check_from_ast(check_ast, tunables_dict)
        checks.append(check_func)

    return checks

def _build_tunables_from_ast(
    self,
    tunables_ast: tuple[Tunable, ...]
) -> dict[str, float]:
    """Convert DQL Tunable AST nodes to value dictionary.

    Creates Tunable objects (auto-discovered from graph later) and
    returns dict mapping tunable name to numeric value for expression
    substitution during parsing.

    Args:
        tunables_ast: Tuple of Tunable AST nodes from DQL

    Returns:
        Dict mapping tunable name to numeric value
    """
    from dqx.tunables import TunableFloat, TunableInt, TunablePercent

    tunables_dict = {}
    for t in tunables_ast:
        # Evaluate bounds and value expressions
        min_val = self._eval_simple_expr(t.bounds[0], tunables_dict)
        max_val = self._eval_simple_expr(t.bounds[1], tunables_dict)
        value = self._eval_simple_expr(t.value, tunables_dict)

        # Store for expression substitution
        tunables_dict[t.name] = value

        # Create Tunable object (will be auto-discovered from graph)
        # Determine type based on values
        if 0 <= value <= 1 and 0 <= min_val <= 1 and 0 <= max_val <= 1:
            TunablePercent(name=t.name, value=value, bounds=(min_val, max_val))
        elif isinstance(value, int) and isinstance(min_val, int) and isinstance(max_val, int):
            TunableInt(name=t.name, value=value, bounds=(min_val, max_val))
        else:
            TunableFloat(name=t.name, value=value, bounds=(min_val, max_val))

    return tunables_dict

def _build_check_from_ast(
    self,
    check_ast: Check,
    tunables: dict[str, float]
) -> CheckProducer:
    """Convert DQL Check AST to Python check function.

    Args:
        check_ast: Check AST node from DQL
        tunables: Dict of tunable values for substitution

    Returns:
        Check function compatible with VerificationSuite
    """
    check_name = check_ast.name
    assertions = check_ast.assertions

    # Capture tunables in closure for use in dynamic_check
    tunables_copy = dict(tunables)

    @check(name=check_name, datasets=list(check_ast.datasets))
    def dynamic_check(mp: MetricProvider, ctx: Context) -> None:
        """Generated check function from DQL."""
        for assertion_ast in assertions:
            self._build_statement(assertion_ast, mp, ctx, tunables_copy)

    return dynamic_check

def _build_statement(
    self,
    statement: Assertion | Collection,
    mp: MetricProvider,
    ctx: Context,
    tunables: dict[str, float],
) -> None:
    """Convert DQL Assertion or Collection to ctx.assert_that() call.

    Args:
        statement: Assertion or Collection AST node
        mp: MetricProvider for evaluating metrics
        ctx: Context for creating assertions
        tunables: Dict of tunable values for substitution
    """
    from dqx.dql.ast import Assertion, Collection

    # Dispatch based on type
    if isinstance(statement, Collection):
        self._build_collection(statement, mp, ctx, tunables)
    else:
        self._build_assertion(statement, mp, ctx, tunables)

def _build_assertion(
    self,
    assertion_ast: Assertion,
    mp: MetricProvider,
    ctx: Context,
    tunables: dict[str, float],
) -> None:
    """Convert DQL Assertion to ctx.assert_that() call."""
    ready = self._setup_assertion_ready(assertion_ast, mp, ctx, tunables)
    if ready is None:
        return

    # Apply condition
    self._apply_condition(ready, assertion_ast, tunables)

def _build_collection(
    self,
    collection_ast: Collection,
    mp: MetricProvider,
    ctx: Context,
    tunables: dict[str, float],
) -> None:
    """Convert DQL Collection to ctx.assert_that(...).noop() call."""
    ready = self._setup_assertion_ready(collection_ast, mp, ctx, tunables)
    if ready is None:
        return

    # Call noop() instead of applying a condition
    ready.noop()

def _setup_assertion_ready(
    self,
    statement: Assertion | Collection,
    mp: MetricProvider,
    ctx: Context,
    tunables: dict[str, float],
) -> AssertionReady | None:
    """Common setup for assertions and collections.

    Returns AssertionReady object.

    Args:
        statement: Assertion or Collection AST node
        mp: MetricProvider for evaluating metric expressions
        ctx: Context for creating assertions
        tunables: Dict of tunable values for substitution

    Returns:
        AssertionReady object, or None if statement should be skipped
    """
    from dqx.dql.errors import DQLError

    # Evaluate metric expression
    metric_value = self._eval_metric_expr(statement.expr, mp, tunables)

    # Use original severity (no profile override in DQL)
    severity = statement.severity

    # Validate name is present
    if statement.name is None:  # pragma: no cover
        raise DQLError("Statement must have a name", loc=statement.loc)

    # Extract cost annotation
    cost_annotation = self._get_cost_annotation(statement)
    cost_dict = {k: float(v) for k, v in cost_annotation.items()} if cost_annotation else None

    # Build and return assertion ready object
    return ctx.assert_that(metric_value).where(
        name=statement.name,
        severity=severity.value,
        tags=set(statement.tags),
        experimental=self._has_annotation(statement, "experimental"),
        required=self._has_annotation(statement, "required"),
        cost=cost_dict,
    )
```

**Implementation tasks:**
- Add these methods to `VerificationSuite` class
- Import necessary DQL types at method scope (avoid circular imports)
- Add proper type hints and docstrings
- Handle edge cases (empty checks, missing names, etc.)

#### 3.2 Move Expression Evaluation Methods

**File:** `src/dqx/api.py`

Move expression evaluation methods from `Interpreter` to `VerificationSuite`:

```python
def _eval_metric_expr(
    self,
    expr: Expr,
    mp: MetricProvider,
    tunables: dict[str, float],
) -> Any:
    """Parse metric expression using sympy.

    Args:
        expr: Expression AST node containing text to parse
        mp: MetricProvider for building namespace
        tunables: Dict of tunable values for substitution

    Returns:
        SymPy expression ready for evaluation
    """
    import sympy as sp
    from dqx.dql.errors import DQLError

    # Substitute tunable values in expression text
    expr_text = self._substitute_tunables(expr.text, tunables)

    # Handle stddev extension with named params specially
    if "stddev(" in expr_text and (", n=" in expr_text or ", offset=" in expr_text):
        return self._handle_stddev_extension(expr_text, mp, tunables)

    # Build namespace with metric functions
    namespace = self._build_metric_namespace(mp)

    # Parse with sympy
    try:
        return sp.sympify(expr_text, locals=namespace, evaluate=False)
    except (sp.SympifyError, TypeError, ValueError) as e:  # pragma: no cover
        raise DQLError(
            f"Failed to parse metric expression: {expr.text}\n{e}",
            loc=expr.loc,
        ) from e

def _eval_simple_expr(self, expr: Expr, tunables: dict[str, float]) -> float:
    """Evaluate simple numeric expressions for tunable bounds and values.

    Args:
        expr: Expression AST node
        tunables: Dict of tunable values for substitution

    Returns:
        Numeric value (int or float)
    """
    from dqx.dql.errors import DQLError

    # Substitute tunables first
    text = self._substitute_tunables(expr.text.strip(), tunables)

    # Handle percentages (already converted by parser, but keep for safety)
    if text.endswith("%"):  # pragma: no cover
        return float(text[:-1]) / 100

    # Handle numeric literals - preserve int vs float
    try:
        if "." not in text:
            return int(text)
        return float(text)
    except ValueError:  # pragma: no cover
        raise DQLError(f"Cannot evaluate expression: {text}", loc=expr.loc) from None

def _substitute_tunables(self, expr_text: str, tunables: dict[str, float]) -> str:
    """Replace tunable names with their values in expression.

    Uses word-boundary regex to avoid corrupting identifiers.

    Args:
        expr_text: Expression text with tunable references
        tunables: Dict mapping tunable names to values

    Returns:
        Expression text with tunables substituted
    """
    import re

    result = expr_text
    # Sort by length descending to handle longest matches first
    for name in sorted(tunables.keys(), key=len, reverse=True):
        value = tunables[name]
        # Use word boundaries to match only complete identifiers
        pattern = r"\b" + re.escape(name) + r"\b"
        result = re.sub(pattern, str(value), result)
    return result

def _build_metric_namespace(self, mp: MetricProvider) -> dict[str, Any]:
    """Build sympy namespace with all metric and math functions.

    Creates a namespace dict for sympy.sympify() that includes:
    - Math functions (abs, sqrt, log, etc.)
    - Base metrics (num_rows, average, sum, etc.)
    - Extension functions (day_over_day, week_over_week)
    - Utility functions (coalesce)

    Args:
        mp: MetricProvider for accessing metric functions

    Returns:
        Dict mapping function names to callables
    """
    import sympy as sp
    from dqx.functions import Coalesce

    def _to_str(arg: Any) -> str:
        """Convert sympy Symbol to string."""
        return str(arg) if isinstance(arg, sp.Symbol) else arg

    def _convert_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
        """Convert sympy types in kwargs to Python primitives."""
        result: dict[str, Any] = {}
        for key, value in kw.items():
            if isinstance(value, sp.Basic):
                if value.is_Integer:
                    result[key] = int(value)  # type: ignore[arg-type]
                elif value.is_Float or value.is_Rational:  # pragma: no cover
                    result[key] = float(value)  # type: ignore[arg-type]
                elif isinstance(value, sp.Symbol):
                    result[key] = str(value)
                else:  # pragma: no cover
                    try:
                        result[key] = float(value)  # type: ignore[arg-type]
                    except (TypeError, AttributeError):
                        result[key] = str(value)
            else:
                result[key] = value  # pragma: no cover
        return result

    def _convert_list_arg(cols: Any) -> list[str]:
        """Convert list of Symbols to list of strings."""
        if isinstance(cols, list):
            return [_to_str(item) for item in cols]
        elif isinstance(cols, tuple):  # pragma: no cover
            return [_to_str(item) for item in cols]
        else:  # pragma: no cover
            return [_to_str(cols)]

    def _convert_value(val: Any) -> int | str | bool:
        """Convert value argument to proper Python type."""
        if isinstance(val, sp.Basic):
            if val.is_Integer:
                return int(val)  # type: ignore[arg-type]
            elif val.is_Float or val.is_Rational:  # pragma: no cover
                return int(float(val))  # type: ignore[arg-type]
            elif isinstance(val, sp.Symbol):  # pragma: no cover
                return str(val)
            else:  # pragma: no cover
                try:
                    return int(val)  # type: ignore[arg-type]
                except (TypeError, AttributeError):
                    return str(val)
        elif isinstance(val, float):
            return int(val)  # pragma: no cover
        elif isinstance(val, (int, str, bool)):
            return val
        else:  # pragma: no cover
            return str(val)

    namespace = {
        # Math functions
        "abs": sp.Abs,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "min": sp.Min,
        "max": sp.Max,
        # Base metrics
        "num_rows": lambda **kw: mp.num_rows(**_convert_kwargs(kw)),
        "null_count": lambda col, **kw: mp.null_count(_to_str(col), **_convert_kwargs(kw)),
        "average": lambda col, **kw: mp.average(_to_str(col), **_convert_kwargs(kw)),
        "sum": lambda col, **kw: mp.sum(_to_str(col), **_convert_kwargs(kw)),
        "minimum": lambda col, **kw: mp.minimum(_to_str(col), **_convert_kwargs(kw)),
        "maximum": lambda col, **kw: mp.maximum(_to_str(col), **_convert_kwargs(kw)),
        "variance": lambda col, **kw: mp.variance(_to_str(col), **_convert_kwargs(kw)),
        "unique_count": lambda col, **kw: mp.unique_count(_to_str(col), **_convert_kwargs(kw)),
        "duplicate_count": lambda cols, **kw: mp.duplicate_count(_convert_list_arg(cols), **_convert_kwargs(kw)),
        "count_values": lambda col, val, **kw: mp.count_values(
            _to_str(col), _convert_value(val), **_convert_kwargs(kw)
        ),
        "first": lambda col, **kw: mp.first(_to_str(col), **_convert_kwargs(kw)),
        # Utility functions
        "coalesce": lambda *args: Coalesce(*args),
        # Extension functions
        "day_over_day": lambda metric, **kw: mp.ext.day_over_day(metric, **_convert_kwargs(kw)),
        "week_over_week": lambda metric, **kw: mp.ext.week_over_week(metric, **_convert_kwargs(kw)),
        # Raw SQL escape hatch
        "custom_sql": lambda expr: mp.custom_sql(_to_str(expr)),
    }
    return namespace

def _handle_stddev_extension(
    self,
    expr_text: str,
    mp: MetricProvider,
    tunables: dict[str, float],
) -> Any:
    """Handle stddev extension function with named parameters.

    Parses stddev calls with optional offset and required n parameters.

    Args:
        expr_text: Expression string containing stddev call
        mp: MetricProvider for accessing extension functions
        tunables: Dict of tunable values for substitution

    Returns:
        Result of mp.ext.stddev() call
    """
    import re
    import sympy as sp

    # Find the matching closing paren for stddev(
    stddev_start = expr_text.find("stddev(")
    if stddev_start == -1:  # pragma: no cover
        namespace = self._build_metric_namespace(mp)
        return sp.sympify(expr_text, locals=namespace, evaluate=False)

    # Start after "stddev("
    pos = stddev_start + 7
    paren_count = 1
    inner_start = pos

    # Find the matching closing paren
    while pos < len(expr_text) and paren_count > 0:
        if expr_text[pos] == "(":
            paren_count += 1
        elif expr_text[pos] == ")":
            paren_count -= 1
        pos += 1

    if paren_count != 0:  # pragma: no cover
        namespace = self._build_metric_namespace(mp)
        return sp.sympify(expr_text, locals=namespace, evaluate=False)

    # Extract everything inside stddev(...)
    inner_content = expr_text[inner_start : pos - 1]

    # Split inner content by commas at top level
    parts = []
    current_part = []
    paren_depth = 0

    for char in inner_content:
        if char == "(":
            paren_depth += 1
            current_part.append(char)
        elif char == ")":
            paren_depth -= 1
            current_part.append(char)
        elif char == "," and paren_depth == 0:
            parts.append("".join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)

    if current_part:
        parts.append("".join(current_part).strip())

    # First part is the inner expression
    inner_expr_text = parts[0] if parts else ""

    # Extract offset and n parameters
    offset = 0
    n = None

    for part in parts[1:]:
        offset_match = re.search(r"offset\s*=\s*(\d+)", part)
        n_match = re.search(r"n\s*=\s*(\d+)", part)

        if offset_match:
            offset = int(offset_match.group(1))
        if n_match:
            n = int(n_match.group(1))

    if len(parts) > 1 and n is None:  # pragma: no cover
        raise ValueError(f"stddev requires 'n' parameter: {expr_text}")

    # Parse the inner expression
    namespace = self._build_metric_namespace(mp)
    inner_metric = sp.sympify(inner_expr_text, locals=namespace, evaluate=False)

    # Call mp.ext.stddev with parsed parameters
    if n is not None:
        result = mp.ext.stddev(inner_metric, offset=offset, n=n)
    else:  # pragma: no cover
        namespace = self._build_metric_namespace(mp)
        return sp.sympify(expr_text, locals=namespace, evaluate=False)

    return result

def _apply_condition(
    self,
    ready: AssertionReady,
    assertion_ast: Assertion,
    tunables: dict[str, float],
) -> None:
    """Apply the assertion condition to AssertionReady.

    Args:
        ready: AssertionReady object to apply condition to
        assertion_ast: Assertion AST node with condition info
        tunables: Dict of tunable values for substitution
    """
    from dqx.dql.errors import DQLError

    cond = assertion_ast.condition

    # Validate threshold present for conditions that require it
    requires_threshold = cond in (">", ">=", "<", "<=", "==", "!=", "between")
    if requires_threshold and assertion_ast.threshold is None:  # pragma: no cover
        raise DQLError(f"Condition '{cond}' requires a threshold", assertion_ast.loc)

    if cond == ">":
        threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)  # type: ignore[arg-type]
        ready.is_gt(threshold)
    elif cond == ">=":
        threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)  # type: ignore[arg-type]
        ready.is_geq(threshold)
    elif cond == "<":
        threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)  # type: ignore[arg-type]
        ready.is_lt(threshold)
    elif cond == "<=":
        threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)  # type: ignore[arg-type]
        ready.is_leq(threshold)
    elif cond == "==":
        threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)  # type: ignore[arg-type]
        if assertion_ast.tolerance:
            ready.is_eq(threshold, tol=assertion_ast.tolerance)
        else:
            ready.is_eq(threshold)
    elif cond == "!=":
        threshold = self._eval_simple_expr(assertion_ast.threshold, tunables)  # type: ignore[arg-type]
        ready.is_neq(threshold)
    elif cond == "between":
        if assertion_ast.threshold_upper is None:  # pragma: no cover
            raise DQLError("Condition 'between' requires upper threshold", assertion_ast.loc)
        lower = self._eval_simple_expr(assertion_ast.threshold, tunables)  # type: ignore[arg-type]
        upper = self._eval_simple_expr(assertion_ast.threshold_upper, tunables)
        ready.is_between(lower, upper)
    elif cond == "is":
        if assertion_ast.keyword == "positive":
            ready.is_positive()
        elif assertion_ast.keyword == "negative":
            ready.is_negative()
        else:  # pragma: no cover
            raise DQLError(f"Unknown keyword: {assertion_ast.keyword}", assertion_ast.loc)
    else:  # pragma: no cover
        raise DQLError(f"Unknown condition: {cond}", assertion_ast.loc)

def _has_annotation(self, statement: Assertion | Collection, name: str) -> bool:
    """Check if assertion or collection has a specific annotation.

    Args:
        statement: Assertion or Collection AST node
        name: Annotation name to check for

    Returns:
        True if annotation is present
    """
    return any(ann.name == name for ann in statement.annotations)

def _get_cost_annotation(self, statement: Assertion | Collection) -> dict[str, int] | None:
    """Extract cost annotation args if present.

    Args:
        statement: Assertion or Collection AST node

    Returns:
        Dict with 'fp' and 'fn' keys, or None if no cost annotation
    """
    for ann in statement.annotations:
        if ann.name == "cost":
            return {
                "fp": int(ann.args.get("false_positive", 1)),
                "fn": int(ann.args.get("false_negative", 1)),
            }
    return None
```

**Implementation tasks:**
- Add these methods to `VerificationSuite` class
- Import DQL types at method scope to avoid circular imports
- Add comprehensive docstrings
- Add type hints for all parameters and returns
- Handle edge cases and error conditions
- Maintain existing logic from `Interpreter`

#### 3.3 Update Interpreter (Optional Deprecation)

**File:** `src/dqx/dql/interpreter.py`

**Option A: Simplify Interpreter to delegate to VerificationSuite** (Recommended)

```python
class Interpreter:
    """Execute DQL files against DQX runtime.

    Note: This class is a convenience wrapper around VerificationSuite.
    For new code, use VerificationSuite(dql=...) directly.
    """

    def __init__(self, db: MetricDB):
        """Initialize interpreter with metric storage."""
        self.db = db

    def run(
        self,
        source: str | Path,
        datasources: Mapping[str, SqlDataSource],
        date: date,
        tags: dict[str, str] | None = None,
        *,
        filename: str | None = None,
    ) -> SuiteResults:
        """Parse and execute DQL from file or string.

        Delegates to VerificationSuite for execution.
        """
        # Create VerificationSuite with DQL
        suite = VerificationSuite(
            dql=source,
            db=self.db,
        )

        # Execute
        tags_dict = tags if tags is not None else {}
        key = ResultKey(date, tags_dict)
        suite.run(list(datasources.values()), key)

        # Collect and convert results
        return self._collect_results(suite, suite._name, date)

    def _collect_results(
        self,
        suite: VerificationSuite,
        suite_name: str,
        execution_date: date
    ) -> SuiteResults:
        """Convert VerificationSuite results to SuiteResults format."""
        from returns.result import Failure, Success

        dqx_results = suite.collect_results()

        assertions = []
        for result in dqx_results:
            metric_value = None
            reason = None

            match result.metric:
                case Success(value):
                    metric_value = value
                case Failure(failures):  # pragma: no cover
                    reason = "; ".join(str(f) for f in failures)

            assertions.append(
                AssertionResult(
                    check_name=result.check,
                    assertion_name=result.assertion,
                    passed=(result.status == "PASSED"),
                    metric_value=metric_value,
                    threshold=None,
                    condition=result.expression or "unknown",
                    severity=result.severity,
                    reason=reason,
                )
            )

        return SuiteResults(
            suite_name=suite_name,
            assertions=tuple(assertions),
            execution_date=execution_date,
        )
```

**Option B: Remove Interpreter entirely**
- Delete `interpreter.py`
- Update imports in `dql/__init__.py`
- Update tests to use `VerificationSuite` directly

**Recommendation:** Keep simplified Interpreter (Option A) for backward compatibility in existing code, but document that `VerificationSuite(dql=...)` is preferred.

---

### Phase 4: Update Tests

#### 4.1 Update DQL Test Files

**Files:** `tests/dql/*.dql`

Remove all `profile` blocks from DQL test files.

**Example: `tests/dql/banking_transactions.dql`**

Remove this block:
```dql
# Profiles for special periods
profile "Holiday Season" {
    from 2024-12-20
    to 2025-01-05

    disable check "Volume"
    scale tag "reconciliation" by 1.5
}
```

**Create corresponding YAML config:** `tests/dql/banking_transactions_config.yaml`

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

**Implementation tasks:**
- Review all `.dql` files in `tests/dql/`
- Remove `profile` blocks
- Create `.yaml` config files for tests that need profiles
- Update test code to load configs

#### 4.2 Update Parser Tests

**File:** `tests/dql/test_parser.py`

Remove all profile-related test cases:

```python
class TestProfiles:  # REMOVE THIS ENTIRE CLASS
    def test_profile_with_date_range(self): ...
    def test_profile_with_disable_rule(self): ...
    def test_profile_with_scale_rule(self): ...
    # ... etc
```

**Implementation tasks:**
- Delete `TestProfiles` class
- Remove any test cases that parse profiles
- Verify remaining tests pass

#### 4.3 Update Interpreter Tests

**File:** `tests/dql/test_interpreter.py`

Remove profile-related test cases:

```python
class TestProfileDates:  # REMOVE THIS ENTIRE CLASS
    def test_profile_activation_in_range(self): ...
    def test_profile_activation_outside_range(self): ...
    # ... etc

class TestProfiles:  # REMOVE THIS ENTIRE CLASS
    def test_profile_disable_check(self): ...
    def test_profile_scale_tag(self): ...
    # ... etc
```

Update existing tests to use `VerificationSuite` directly:

```python
def test_dql_execution(self, db, datasource):
    """Test DQL execution through VerificationSuite."""
    dql = '''
        suite "Test Suite" {
            check "Test Check" on ds {
                assert num_rows() > 0
                    name "Has rows"
            }
        }
    '''

    suite = VerificationSuite(
        dql=dql,
        db=db,

    )
    suite.run([datasource], ResultKey(date.today(), {}))
    results = suite.collect_results()

    assert len(results) == 1
    assert results[0].status == "PASSED"
```

**Implementation tasks:**
- Delete profile-related test classes
- Update tests to use `VerificationSuite(dql=...)`
- Add tests for YAML profile loading
- Maintain 100% coverage

#### 4.4 Update Collection Tests

**File:** `tests/dql/test_collect.py`

Update profile tests to use YAML config:

```python
class TestCollectWithProfiles:  # UPDATE THIS CLASS
    def test_collect_with_profiles_from_yaml(self, db, datasource):
        """Test collection with profiles loaded from YAML."""
        dql = Path("tests/dql/test_suite.dql")
        config = Path("tests/dql/test_config.yaml")

        suite = VerificationSuite(
            dql=dql,
            db=db,

            config=config,
        )
        suite.run([datasource], ResultKey(date(2024, 12, 25), {}))
        results = suite.collect_results()

        # Verify profile was applied
        assert ...

    def test_collect_with_profiles_from_api(self, db, datasource):
        """Test collection with profiles from Python API."""
        dql = Path("tests/dql/test_suite.dql")

        profile = SeasonalProfile(
            name="Test Profile",
            start_date=date(2024, 12, 20),
            end_date=date(2025, 1, 5),
            rules=[check("Volume").disable()],
        )

        suite = VerificationSuite(
            dql=dql,
            db=db,

            profiles=[profile],
        )
        suite.run([datasource], ResultKey(date(2024, 12, 25), {}))
        results = suite.collect_results()

        # Verify profile was applied
        assert ...
```

**Implementation tasks:**
- Remove tests for DQL profile syntax
- Add tests for YAML profile loading
- Add tests for API profile passing
- Add tests for profile merging (config + API)

#### 4.5 Update E2E Tests

**File:** `tests/e2e/test_api_e2e.py`

Add end-to-end tests for DQL integration:

```python
def test_dql_suite_with_profiles(db, datasource):
    """Test complete DQL suite execution with profiles."""
    dql = Path("tests/dql/commerce_suite.dql")

    profile = SeasonalProfile(
        name="Holiday Season",
        start_date=date(2024, 12, 20),
        end_date=date(2025, 1, 5),
        rules=[tag("xmas").set(metric_multiplier=2.0)],
    )

    suite = VerificationSuite(
        dql=dql,
        db=db,

        profiles=[profile],
    )

    suite.run([datasource], ResultKey(date(2024, 12, 25), {}))
    results = suite.collect_results()

    assert len(results) > 0
    # Verify profile was applied correctly
```

**Implementation tasks:**
- Add tests for DQL + YAML config
- Add tests for DQL + API profiles
- Test cross-dataset checks in DQL
- Test tunables + profiles interaction

#### 4.6 Update Config Tests

**File:** `tests/test_config.py` (or create if doesn't exist)

Add tests for profile YAML loading:

```python
def test_load_profiles_from_yaml():
    """Test loading profiles from YAML config."""
    yaml_content = """
    profiles:
      - name: "Holiday Season"
        type: "seasonal"
        start_date: "2024-12-20"
        end_date: "2025-01-05"
        rules:
          - action: "disable"
            target: "check"
            name: "Volume"
    """

    config_data = yaml.safe_load(yaml_content)
    profiles = load_profiles_from_config(config_data)

    assert len(profiles) == 1
    assert profiles[0].name == "Holiday Season"
    assert len(profiles[0].rules) == 1

def test_invalid_profile_config():
    """Test error handling for invalid profile config."""
    yaml_content = """
    profiles:
      - name: "Bad Profile"
        type: "seasonal"
        # Missing start_date and end_date
    """

    config_data = yaml.safe_load(yaml_content)
    with pytest.raises(DQXError, match="start_date"):
        load_profiles_from_config(config_data)
```

**Implementation tasks:**
- Test valid profile YAML parsing
- Test invalid configs (missing fields, bad dates, etc.)
- Test all rule action types
- Test profile merging logic

---

### Phase 5: Documentation & Cleanup

#### 5.1 Update API Documentation

**File:** `src/dqx/api.py`

Update `VerificationSuite.__init__()` docstring:

```python
def __init__(
    self,
    checks: Sequence[CheckProducer | DecoratedCheck] | None = None,
    dql: str | Path | None = None,
    db: MetricDB,
    name: str,
    log_level: int | str = logging.INFO,
    data_av_threshold: float = 0.9,
    profiles: Sequence[Profile] | None = None,
    config: Path | None = None,
) -> None:
    """Initialize a VerificationSuite.

    The suite can be initialized in two ways:

    1. **Python API**: Pass check functions via `checks` parameter
    2. **DQL**: Pass DQL source or file path via `dql` parameter

    Exactly one of `checks` or `dql` must be provided.

    Args:
        checks: Python check functions decorated with @check.
                Mutually exclusive with `dql`.
        dql: DQL source string or Path to .dql file.
             Mutually exclusive with `checks`.
             If Path, must point to a valid .dql file.
             If str, treated as DQL source code.
        db: Metric database for storing and retrieving metrics.
        name: Suite name. If dql is provided and defines a suite name,
              the DQL name takes precedence.
        log_level: Logging level (default: logging.INFO).
        data_av_threshold: Minimum data availability threshold (0.0-1.0).
                          Can be overridden by DQL availability_threshold.
        profiles: Profile objects to apply during suite execution.
                 Profiles from config file (if provided) are loaded first,
                 then profiles from this parameter are appended.
                 All profiles are active; there is no override behavior.
        config: Path to YAML configuration file.
               Config can specify:
               - tunables: Initial values for tunable parameters
               - profiles: Profile definitions (merged with profiles param)

    Raises:
        DQXError: If both or neither of checks/dql provided.
        DQXError: If DQL parsing fails.
        DQXError: If config file is invalid.
        DQXError: If suite validation fails.

    Examples:
        Python API (existing usage):

        >>> @check(name="Completeness", datasets=["orders"])
        >>> def check_completeness(mp, ctx):
        >>>     ctx.assert_that(mp.null_count("id")).where(
        >>>         name="ID not null"
        >>>     ).is_eq(0)
        >>>
        >>> suite = VerificationSuite(
        >>>     checks=[check_completeness],
        >>>     db=db,
        >>>     name="My Suite",
        >>> )

        DQL with file:

        >>> suite = VerificationSuite(
        >>>     dql=Path("suites/orders.dql"),
        >>>     db=db,
        >>>     config=Path("config/prod.yaml"),
        >>> )

        DQL with inline source:

        >>> dql_source = '''
        >>> suite "Orders" {
        >>>     check "Completeness" on orders {
        >>>         assert null_count(id) == 0
        >>>             name "ID not null"
        >>>     }
        >>> }
        >>> '''
        >>> suite = VerificationSuite(
        >>>     dql=dql_source,
        >>>     db=db,
        >>>
        >>> )

        DQL with programmatic profiles:

        >>> holiday = SeasonalProfile(
        >>>
        >>>     start_date=date(2024, 12, 20),
        >>>     end_date=date(2025, 1, 5),
        >>>     rules=[check("Volume").disable()],
        >>> )
        >>>
        >>> suite = VerificationSuite(
        >>>     dql=Path("suites/orders.dql"),
        >>>     db=db,
        >>>     profiles=[holiday],
        >>> )
    """
```

#### 5.2 Update Config Module Documentation

**File:** `src/dqx/config.py`

Add module-level docstring explaining profile loading:

```python
"""Configuration file loading for tunables and profiles.

DQX supports loading configuration from YAML files. Configuration files
can specify:

1. **Tunables**: Initial values for tunable parameters defined in DQL or Python
2. **Profiles**: Date-based behavioral overrides for assertions

## YAML Format

```yaml
# Tunables: Override initial values
tunables:
  MAX_NULL_RATE: 0.02
  MIN_DAILY_TRANSACTIONS: 15000
  THRESHOLD: 0.8

# Profiles: Define behavioral overrides
profiles:
  - name: "Holiday Season"
    type: "seasonal"
    start_date: "2024-12-20"  # ISO 8601 format
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Volume"

      - action: "scale"
        target: "tag"
        name: "reconciliation"
        multiplier: 1.5

      - action: "set_severity"
        target: "tag"
        name: "fraud"
        severity: "P0"
```

## Usage

```python
from dqx import VerificationSuite
from pathlib import Path

suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    config=Path("config/prod.yaml"),  # Loads tunables + profiles
)
```

## Profile Actions

- **disable**: Skip matching assertions
- **scale**: Multiply metric value by multiplier before comparison
- **set_severity**: Override assertion severity level

## Profile Targets

- **check**: Match all assertions in a check by check name
- **tag**: Match all assertions with a specific tag

## Profile Precedence

Profiles from config file and profiles from Python API are both active.
There is no override behavior - all matching rules are applied.
If multiple rules match, they are applied in order:
- Disabled assertions are skipped (any disable rule wins)
- Multipliers are compounded (multiply together)
- Last severity override wins
"""
```

#### 5.3 Update DQL Documentation

**File:** `docs/design/dql-language.md`

Update structure section to remove profiles:

```markdown
## Structure

A DQL file contains one suite. A suite contains checks and tunables.

```text
suite
├── metadata (name, threshold)
├── tunables
└── checks
    └── assertions
```

Note: Profiles are no longer part of DQL syntax. Use YAML configuration
or Python API to define profiles.

### Migration from DQL Profiles

If your DQL files contain `profile` blocks, migrate them to YAML config:

**Before (DQL):**
```dql
suite "Orders" {
    profile "Holiday Season" {
        from 2024-12-20
        to 2025-01-05

        disable check "Volume"
        scale tag "reconciliation" by 1.5
    }
}
```

**After (YAML config):**
```yaml
# config.yaml
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

**After (Python API):**
```python
from dqx import VerificationSuite, SeasonalProfile, check, tag
from datetime import date

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
```

#### 5.4 Create Migration Guide

**File:** `docs/migration/dql-profiles-to-yaml.md` (create new)

```markdown
# Migrating DQL Profiles to YAML Config

This guide helps you migrate from DQL profile syntax (removed in v0.6.0)
to YAML configuration or Python API.

## Why the Change?

Profiles define runtime behavior (when/how to modify assertions), which is
conceptually separate from validation logic (what to validate). Moving
profiles out of DQL provides:

- **Separation of concerns**: DQL files define validation logic only
- **Environment flexibility**: Use same DQL with different configs per environment
- **Easier maintenance**: Update profile dates/multipliers without touching DQL
- **Consistency**: Profiles follow same pattern as tunables (YAML config)

## Migration Steps

### Step 1: Identify DQL Files with Profiles

Search for `profile` keyword in your `.dql` files:

```bash
grep -r "profile \"" *.dql
```

### Step 2: Extract Profile Definitions

For each profile block in DQL, extract:
- Profile name
- Date range (from/to)
- Rules (disable/scale/set_severity)

### Step 3: Convert to YAML

Use this mapping:

| DQL Syntax | YAML Syntax |
|------------|-------------|
| `profile "Name" { from DATE to DATE ... }` | `name: "Name"<br>start_date: "DATE"<br>end_date: "DATE"` |
| `disable check "CheckName"` | `action: "disable"<br>target: "check"<br>name: "CheckName"` |
| `disable assertion "Name" in "CheckName"` | Use check-level disable instead |
| `scale check "CheckName" by MULTIPLIER` | `action: "scale"<br>target: "check"<br>name: "CheckName"<br>multiplier: MULTIPLIER` |
| `scale tag "TagName" by MULTIPLIER` | `action: "scale"<br>target: "tag"<br>name: "TagName"<br>multiplier: MULTIPLIER` |
| `set severity check "CheckName" to P0` | `action: "set_severity"<br>target: "check"<br>name: "CheckName"<br>severity: "P0"` |
| `set severity tag "TagName" to P0` | `action: "set_severity"<br>target: "tag"<br>name: "TagName"<br>severity: "P0"` |

### Step 4: Create Config File

Create `config.yaml` next to your DQL file:

```yaml
profiles:
  - name: "Your Profile Name"
    type: "seasonal"
    start_date: "YYYY-MM-DD"
    end_date: "YYYY-MM-DD"
    rules:
      # Add converted rules here
```

### Step 5: Remove Profile Blocks from DQL

Delete all `profile` blocks from your `.dql` files.

### Step 6: Update VerificationSuite Creation

**Before:**
```python
from dqx.dql import Interpreter

interp = Interpreter(db=db)
results = interp.run(Path("suite.dql"), datasources, date.today())
```

**After:**
```python
from dqx import VerificationSuite

suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    config=Path("config.yaml"),  # Add this
)
suite.run(datasources, key)
results = suite.collect_results()
```

## Complete Example

### Before (DQL with profiles)

```dql
suite "Banking Transactions" {
    tunable MAX_NULL_RATE = 1% bounds [0%, 5%]

    check "Volume" on transactions {
        assert num_rows() >= 10000
            name "Min daily transactions"
            tags [volume]
    }

    check "Reconciliation" on transactions, settlements {
        assert abs(sum(amount, dataset=transactions) - sum(amount, dataset=settlements)) < 1000
            name "Amount balance"
            tags [reconciliation, critical]
    }

    profile "Holiday Season" {
        from 2024-12-20
        to 2025-01-05

        disable check "Volume"
        scale tag "reconciliation" by 1.5
        set severity tag "critical" to P0
    }
}
```

### After (DQL without profiles)

**`banking.dql`:**
```dql
suite "Banking Transactions" {
    tunable MAX_NULL_RATE = 1% bounds [0%, 5%]

    check "Volume" on transactions {
        assert num_rows() >= 10000
            name "Min daily transactions"
            tags [volume]
    }

    check "Reconciliation" on transactions, settlements {
        assert abs(sum(amount, dataset=transactions) - sum(amount, dataset=settlements)) < 1000
            name "Amount balance"
            tags [reconciliation, critical]
    }
}
```

**`banking_config.yaml`:**
```yaml
tunables:
  MAX_NULL_RATE: 0.01  # Override if needed

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

      - action: "set_severity"
        target: "tag"
        name: "critical"
        severity: "P0"
```

**Python code:**
```python
from dqx import VerificationSuite
from pathlib import Path

suite = VerificationSuite(
    dql=Path("banking.dql"),
    db=db,
    config=Path("banking_config.yaml"),
)

suite.run(datasources, key)
results = suite.collect_results()
```

## Alternative: Python API

If you prefer programmatic configuration:

```python
from dqx import VerificationSuite, SeasonalProfile, check, tag
from datetime import date

holiday = SeasonalProfile(
    name="Holiday Season",
    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[
        check("Volume").disable(),
        tag("reconciliation").set(metric_multiplier=1.5),
        tag("critical").set(severity="P0"),
    ],
)

suite = VerificationSuite(
    dql=Path("banking.dql"),
    db=db,
    profiles=[holiday],
)
```

## Troubleshooting

### Error: "Exactly one of 'checks' or 'dql' must be provided"

Make sure you're using the `dql` parameter, not `checks`:

```python
# Wrong:
suite = VerificationSuite(checks=Path("suite.dql"), db=db, name="Suite")

# Correct:
suite = VerificationSuite(dql=Path("suite.dql"), db=db)
```

**Note**: The `name` parameter cannot be specified when using `dql`. The suite name is defined in the DQL file itself.

### Error: "Profile configuration invalid"

Check your YAML syntax:
- Dates must be in ISO 8601 format: `"YYYY-MM-DD"`
- All required fields must be present
- Rule actions must be: "disable", "scale", or "set_severity"
- Targets must be: "check" or "tag"

### Profiles Not Applied

Verify:
1. Profile date range includes your execution date
2. Profile target names match check/tag names exactly (case-sensitive)
3. Config file is passed to `VerificationSuite` constructor

### Need Help?

Open an issue on [GitHub](https://github.com/nampham2/dqx/issues).
```

#### 5.5 Update Changelog

**File:** `docs/changelog.md`

Add entry for this breaking change:

```markdown
## v0.6.0 (YYYY-MM-DD)

### Breaking Changes

#### DQL Profiles Removed

Profile definitions are no longer supported in DQL syntax. Profiles must now
be defined in YAML configuration files or passed programmatically via the
Python API.

##### Migration required:
- Extract `profile` blocks from `.dql` files
- Convert to YAML format in `config.yaml`
- Or use Python `SeasonalProfile` API
- See [Migration Guide](migration/dql-profiles-to-yaml.md) for details

##### Rationale:
- Separates validation logic (DQL) from runtime behavior (profiles)
- Enables environment-specific configuration without modifying DQL
- Follows same pattern as tunables (YAML config)
- Simplifies DQL grammar and parser

**Before:**
```dql
suite "Orders" {
    profile "Holiday" { from 2024-12-20 to 2025-01-05
        disable check "Volume"
    }
}
```

**After (YAML):**
```yaml
profiles:
  - name: "Holiday"
    type: "seasonal"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "disable"
        target: "check"
        name: "Volume"
```

**After (Python API):**
```python
holiday = SeasonalProfile(

    start_date=date(2024, 12, 20),
    end_date=date(2025, 1, 5),
    rules=[check("Volume").disable()],
)

suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,

    profiles=[holiday],
)
```

### Features

**DQL Integration into VerificationSuite**

`VerificationSuite` can now accept DQL programs directly via the `dql` parameter:

```python
suite = VerificationSuite(
    dql=Path("suite.dql"),  # NEW
    db=db,
    config=Path("config.yaml"),  # Loads tunables + profiles
)
```

This simplifies the API by removing the need for a separate `Interpreter` class.

**Profile Loading from YAML Config**

Profiles can now be loaded from YAML configuration files:

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

Load via `config` parameter:

```python
suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    config=Path("config.yaml"),  # Loads profiles + tunables
)
```

Or pass programmatically:

```python
suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    profiles=[holiday_profile],  # Python API
)
```

Profiles from config and API are both active (merged, not overridden).

### Deprecations

**`Interpreter` class** is now a thin wrapper around `VerificationSuite`.
For new code, use `VerificationSuite(dql=...)` directly.

The `Interpreter` class will be removed in v1.0.0.
```

#### 5.6 Run Quality Checks

**Run full test suite:**
```bash
uv run pytest
```

**Check coverage:**
```bash
uv run pytest --cov=src/dqx --cov-report=term-missing
```

Ensure 100% coverage is maintained.

**Run pre-commit hooks:**
```bash
uv run pre-commit run --all-files
```

Ensure all checks pass:
- ruff format
- ruff check
- mypy
- commitizen check

**Build documentation:**
```bash
uv run mkdocs build
```

Ensure docs build without errors.

---

## Testing Strategy

### Unit Tests

1. **Config Module** (`tests/test_config.py`):
   - Test YAML profile parsing (valid/invalid)
   - Test all rule action types
   - Test missing required fields
   - Test invalid date formats
   - Test profile merging logic

2. **VerificationSuite** (`tests/test_api.py`):
   - Test `dql` parameter acceptance
   - Test mutual exclusion of `checks` and `dql`
   - Test DQL parsing and check building
   - Test tunable extraction from DQL
   - Test profile loading from config
   - Test profile merging (config + API)

3. **DQL Parser** (`tests/dql/test_parser.py`):
   - Verify profiles no longer parse
   - Test suite parsing without profiles
   - Test error messages for profile syntax

### Integration Tests

1. **DQL Execution** (`tests/dql/test_interpreter.py`):
   - Test full DQL suite execution
   - Test tunables in DQL expressions
   - Test cross-dataset checks
   - Test all metric functions

2. **Profile Application** (`tests/test_profiles.py`):
   - Test disable rules work correctly
   - Test scale rules multiply metrics
   - Test severity overrides
   - Test multiple profiles active
   - Test profile date ranges

### End-to-End Tests

1. **Complete Workflow** (`tests/e2e/test_api_e2e.py`):
   - Test DQL + YAML config + datasources
   - Test DQL + API profiles
   - Test tunables + profiles interaction
   - Test result collection
   - Test error scenarios

---

## Rollout Plan

### Phase 0: Preparation
- Create feature branch
- Set up testing infrastructure
- Review plan with team

### Phase 1: Foundation (Week 1)
- Implement YAML profile loading in `config.py`
- Add unit tests for profile parsing
- Create example YAML config files

### Phase 2: Integration (Week 1)
- Add `dql` parameter to `VerificationSuite`
- Update constructor logic
- Add profile merging
- Write integration tests

### Phase 3: DQL Cleanup (Week 2)
- Remove profiles from grammar
- Remove profile AST nodes
- Update parser
- Simplify interpreter
- Update all affected tests

### Phase 4: Suite Integration (Week 2)
- Move expression evaluation to `VerificationSuite`
- Add DQL check building methods
- Test DQL execution through suite
- Ensure 100% coverage maintained

### Phase 5: Testing & Docs (Week 3)
- Update all test files
- Remove profile blocks from test `.dql` files
- Create YAML configs for tests
- Write migration guide
- Update API documentation
- Update changelog

### Phase 6: Review & Release (Week 3)
- Code review
- Final testing pass
- Update version to 0.6.0
- Merge to main
- Create release tag

---

## Success Criteria

- [ ] `VerificationSuite` accepts `dql` parameter (Path or string)
- [ ] Profiles load from YAML config via `config` parameter
- [ ] Profiles can be passed via `profiles` parameter (Python API)
- [ ] Config profiles and API profiles are both active (merged)
- [ ] DQL grammar has no profile syntax
- [ ] All profile-related AST nodes removed
- [ ] All profile-related parser methods removed
- [ ] All profile logic removed from `Interpreter`
- [ ] All tests updated to remove DQL profile usage
- [ ] New tests added for YAML profile loading
- [ ] New tests added for DQL parameter
- [ ] 100% test coverage maintained
- [ ] All pre-commit hooks pass
- [ ] Documentation updated (API, migration guide, changelog)
- [ ] Example files updated

---

## FAQ

### Q: Why remove profiles from DQL?

A: Profiles define **runtime behavior** (when/how to modify checks), which
is conceptually separate from **validation logic** (what to check). Keeping
them in DQL mixed these concerns. Moving profiles to config/API:

- Enables environment-specific behavior without modifying DQL
- Follows same pattern as tunables (YAML config)
- Makes DQL simpler and more focused
- Allows profile reuse across multiple DQL suites

### Q: Can I still use profiles?

A: Yes! Profiles are more powerful than before:

- Load from YAML config (shareable across suites)
- Pass programmatically via Python API (full flexibility)
- Mix both approaches (config + API profiles both active)

### Q: What if I have many DQL files with profiles?

A: See the [Migration Guide](migration/dql-profiles-to-yaml.md) for
step-by-step instructions. A semi-automated script could be created to
extract profiles from DQL and generate YAML configs.

### Q: Will the `Interpreter` class be removed?

A: It's simplified to a thin wrapper for backward compatibility, but
`VerificationSuite(dql=...)` is now the preferred API. The `Interpreter`
class may be fully deprecated in v1.0.0.

### Q: How do profiles work with tunables?

A: Tunables and profiles are independent:

- **Tunables**: Adjustable threshold parameters
- **Profiles**: Date-based behavioral overrides

Both can be defined in YAML config:

```yaml
tunables:
  MAX_NULL_RATE: 0.02

profiles:
  - name: "Holiday"
    start_date: "2024-12-20"
    end_date: "2025-01-05"
    rules:
      - action: "scale"
        target: "tag"
        name: "volume"
        multiplier: 2.0
```

### Q: What happens if profiles overlap?

A: All matching profiles are active. Rules are applied in order:

- **Disable**: Any disable rule wins (assertion skipped)
- **Scale**: Multipliers compound (multiply together)
- **Severity**: Last matching rule wins

### Q: Can I use assertion-level disable in YAML?

A: No, YAML only supports check-level and tag-level targeting.
Use check-level disable or add tags to specific assertions.

Python API still supports assertion-level:

```python
from dqx.profiles import assertion

rule = assertion("Check Name", "Assertion Name").disable()
```

---

## References

- [DQL Language Design](../design/dql-language.md)
- [Profiles Documentation](../design/profiles.md)
- [API Reference](../api-reference.md)
- [Configuration Guide](../user-guide.md#configuration)

---

## Appendix: File Checklist

### Modified Files

**Core:**
- `src/dqx/api.py` - Add DQL support, DQL parsing methods
- `src/dqx/config.py` - Add profile YAML loading
- `src/dqx/dql/grammar.lark` - Remove profile syntax
- `src/dqx/dql/ast.py` - Remove profile AST nodes
- `src/dqx/dql/parser.py` - Remove profile parsing
- `src/dqx/dql/interpreter.py` - Remove profile logic, simplify

**Tests:**
- `tests/test_config.py` - Add profile YAML tests
- `tests/test_api.py` - Add DQL parameter tests
- `tests/dql/test_parser.py` - Remove profile parsing tests
- `tests/dql/test_interpreter.py` - Remove profile logic tests, update for VerificationSuite
- `tests/dql/test_collect.py` - Update profile tests for YAML
- `tests/e2e/test_api_e2e.py` - Add DQL integration E2E tests
- `tests/test_profiles.py` - Add profile loading tests

**Test Data:**
- `tests/dql/*.dql` - Remove profile blocks
- `tests/dql/*_config.yaml` - Add YAML configs (create new)

**Documentation:**
- `docs/api-reference.md` - Update VerificationSuite docs
- `docs/design/dql-language.md` - Remove profile section
- `docs/user-guide.md` - Add DQL + config examples
- `docs/changelog.md` - Add v0.6.0 entry
- `docs/migration/dql-profiles-to-yaml.md` - Create migration guide (new)

### New Files

- `docs/plans/dql-verification-suite-integration.md` - This document
- `docs/migration/dql-profiles-to-yaml.md` - Migration guide
- `tests/dql/test_suite_config.yaml` - Example YAML configs for tests
- `tests/test_config.py` - Config loading tests (if doesn't exist)

### Deleted Files

None (files modified, not deleted)

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Profile Config | 3 days | None |
| Phase 2: Remove DQL Profiles | 2 days | Phase 1 |
| Phase 3: Suite Integration | 4 days | Phase 2 |
| Phase 4: Update Tests | 3 days | Phase 3 |
| Phase 5: Documentation | 2 days | Phase 4 |
| **Total** | **2-3 weeks** | Sequential |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking change affects users | High | Provide detailed migration guide, examples, and clear error messages |
| Complex refactoring introduces bugs | High | Maintain 100% test coverage, comprehensive test suite, code review |
| Performance regression | Medium | Benchmark before/after, DQL parsing is not performance-critical |
| Missed edge cases in DQL parsing | Medium | Port all existing Interpreter logic, test extensively |
| YAML schema changes needed | Low | Design extensible schema upfront, validate thoroughly |

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**Authors:** DQX Team
**Status:** Draft
