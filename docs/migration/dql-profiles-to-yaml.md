# Migrating DQL Profiles to YAML Configuration

This guide helps you migrate from DQL profile syntax (removed in v0.6.0) to YAML configuration or Python API.

## Why the Change?

Profiles define runtime behavior (when/how to modify assertions), which is conceptually separate from validation logic (what to validate). Moving profiles out of DQL provides:

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
| `profile "Name" { from DATE to DATE ... }` | `name: "Name"`<br>`start_date: "DATE"`<br>`end_date: "DATE"` |
| `disable check "CheckName"` | `action: "disable"`<br>`target: "check"`<br>`name: "CheckName"` |
| `disable assertion "Name" in "CheckName"` | Use check-level disable or add tags |
| `scale check "CheckName" by MULTIPLIER` | `action: "scale"`<br>`target: "check"`<br>`name: "CheckName"`<br>`multiplier: MULTIPLIER` |
| `scale tag "TagName" by MULTIPLIER` | `action: "scale"`<br>`target: "tag"`<br>`name: "TagName"`<br>`multiplier: MULTIPLIER` |
| `set severity check "CheckName" to P0` | `action: "set_severity"`<br>`target: "check"`<br>`name: "CheckName"`<br>`severity: "P0"` |
| `set severity tag "TagName" to P0` | `action: "set_severity"`<br>`target: "tag"`<br>`name: "TagName"`<br>`severity: "P0"` |

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
from dqx.api import VerificationSuite
from dqx.common import ResultKey
from datetime import date

suite = VerificationSuite(
    dql=Path("suite.dql"),
    db=db,
    config=Path("config.yaml"),  # Add this
)
key = ResultKey(date.today(), {})
suite.run(datasources, key)
results = suite.collect_results()
```

## Complete Example

### Before (DQL with profiles)

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

    profile "Holiday Season" {
        from 2024-12-20
        to 2025-01-05

        disable check "Volume"
        scale tag "reconciliation" by 1.5
        set severity tag "critical" to P0
    }
}
```

**Python code:**
```python
from dqx.dql import Interpreter

interp = Interpreter(db=db)
results = interp.run(
    Path("banking.dql"),
    datasources={"transactions": tx_ds, "settlements": settle_ds},
    date=date(2024, 12, 25),
)
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
from dqx.api import VerificationSuite
from dqx.common import ResultKey
from pathlib import Path

suite = VerificationSuite(
    dql=Path("banking.dql"),
    db=db,
    config=Path("banking_config.yaml"),
)

suite.run(
    datasources=[tx_ds, settle_ds],
    key=ResultKey(date(2024, 12, 25), {}),
)

results = suite.collect_results()
```

## Alternative: Python API

If you prefer programmatic configuration:

```python
from dqx.api import VerificationSuite
from dqx.profiles import SeasonalProfile, check, tag
from datetime import date
from pathlib import Path

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

### Error: "Unexpected token" or syntax error with profile

DQL no longer supports profile syntax. You need to:

1. Remove the `profile` block from your DQL file
2. Create a YAML config file with the profile definition
3. Pass the config file to VerificationSuite via the `config` parameter

### Error: "Profile configuration invalid"

Check your YAML syntax:
- Dates must be in ISO 8601 format: `"YYYY-MM-DD"`
- All required fields must be present (`name`, `type`, `start_date`, `end_date`, `rules`)
- Rule actions must be: `"disable"`, `"scale"`, or `"set_severity"`
- Targets must be: `"check"` or `"tag"`
- `type` must be `"seasonal"` (currently the only supported type)

### Profiles Not Applied

Verify:
1. Profile date range includes your execution date
2. Profile target names match check/tag names exactly (case-sensitive)
3. Config file is passed to `VerificationSuite` constructor: `config=Path("config.yaml")`
4. Config file contains valid YAML with `profiles:` section

### Result Format Changed

If you were using `Interpreter` and accessing `SuiteResults` or `AssertionResult`:

**Before:**
```python
results: SuiteResults = interp.run(...)
for assertion in results.assertions:
    print(assertion.assertion_name, assertion.passed)
```

**After:**
```python
results = suite.collect_results()
for result in results:
    print(result.assertion_name, result.status == "PASSED")
```

### Need Help?

Open an issue on [GitHub](https://github.com/nampham2/dqx/issues).

## Summary

The migration from DQL profiles to YAML configuration:

1. **Removes** profile blocks from DQL files
2. **Creates** YAML config files with profile definitions
3. **Updates** Python code to use `VerificationSuite(dql=...)` instead of `Interpreter`
4. **Provides** more flexibility for environment-specific configuration

This change makes DQL files focused on validation logic while keeping runtime behavior configuration separate and reusable.
