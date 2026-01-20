---
description: Validate release readiness without creating PR
agent: build
---

Check if your code is ready for release without starting the actual release process.

This command uses the same validation agents as `/release` to provide early feedback.

## Overview

Runs comprehensive validation using specialized DQX agents:
- **dqx-test**: Test coverage analysis (100% required)
- **dqx-quality**: Code quality checks (all hooks must pass)
- **dqx-docs**: Documentation completeness verification

Use this before running `/release` to catch issues early.

## Benefits

1. **Early Feedback**: Catch issues before starting release process
2. **No Side Effects**: Doesn't create branches or modify files
3. **Detailed Guidance**: Agents provide specific fix suggestions
4. **Same Standards**: Uses identical validation as actual release
5. **Fast Iteration**: Fix → validate → repeat until ready

## Step 1: Test Coverage Validation

Delegate to **dqx-test agent** to validate test coverage:

**Agent task:** Run full test suite and verify 100% coverage requirement

**Commands for agent:**
- `uv run pytest` - Run all tests
- `uv run pytest --cov=src/dqx --cov-report=term-missing` - Generate coverage report

**Success criteria:**
- All tests pass (~1659+ tests expected)
- Coverage is exactly 100% (5088/5088 statements expected)
- No test failures or errors

**Expected output if passing:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Test Validation (dqx-test)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All tests pass: 1659 tests
Coverage: 100% (5088/5088 statements)
Duration: ~45 seconds

✅ Ready for release
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Expected output if issues found:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Test Validation Issues (dqx-test)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Coverage: 98.5% (5042/5088 statements)
Missing: 46 statements

Uncovered lines:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. src/dqx/api.py:123-125 (3 lines)
   Code: Error handling for invalid input
   Context: Validates user input parameters

   Suggested test:
   def test_api_invalid_input():
       with pytest.raises(ValueError):
           process(None)  # Trigger error path

2. src/dqx/validator.py:456 (1 line)
   Code: Edge case for empty dataset
   Context: Handles empty pa.Table

   Suggested test:
   def test_validator_empty_dataset():
       empty = pa.Table.from_pydict({})
       result = validate(empty)
       assert result.is_valid()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action Required:
- Add tests to cover 46 uncovered statements
- OR add `# pragma: no cover` if unreachable
- Target: 100% coverage (required for release)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Step 2: Code Quality Validation

Delegate to **dqx-quality agent** to validate code quality:

**Agent task:** Run all quality checks and pre-commit hooks

**Commands for agent:**
- `uv run ruff format` - Check formatting
- `uv run ruff check --fix` - Check linting with auto-fix
- `uv run mypy src tests` - Check types
- `uv run pre-commit run --all-files` - Run all 21 hooks

**Success criteria:**
- All files properly formatted
- No linting errors
- No type errors
- All 21 pre-commit hooks pass

**Expected output if passing:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Quality Validation (dqx-quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Formatting: All files formatted correctly
Linting: No issues found
Type checking: All functions properly typed
Pre-commit hooks: All 21 passed

✅ Ready for release
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Expected output if issues found:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Quality Validation Issues (dqx-quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Linting Issues:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. src/dqx/provider.py:89:5
   Error: F841 - Local variable 'x' is assigned but never used

   Fix: Remove unused variable or prefix with underscore
   Before: x = compute_value()
   After:  _x = compute_value()  # or remove if not needed

2. src/dqx/analyzer.py:234:80
   Error: E501 - Line too long (125 > 120 characters)

   Fix: Break line into multiple lines
   Before: result = process(arg1, arg2, arg3, arg4, arg5)
   After:  result = process(
               arg1, arg2, arg3,
               arg4, arg5
           )

Type Errors:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. src/dqx/validator.py:123
   Error: Function missing return type annotation

   Fix: Add return type annotation
   Before: def validate(self):
   After:  def validate(self) -> ResultKey:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action Required:
- Run `uv run ruff check --fix` to auto-fix formatting
- Add return type to validator.py:123
- Fix line length at analyzer.py:234
- Re-run validation after fixes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Step 3: Documentation Validation

Delegate to **dqx-docs agent** to validate documentation completeness:

**Agent task:** Verify all documentation is complete and accurate

**Checks for agent:**
- All public APIs have Google-style docstrings
- README examples are current and working
- CHANGELOG is up to date with recent changes
- No TODO or FIXME comments in public API code
- Documentation consistency across files
- Version numbers in examples are not hardcoded (or match current version)

**Success criteria:**
- All public methods have complete docstrings
- Examples in README match current API
- No missing or outdated documentation

**Expected output if passing:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Documentation Validation (dqx-docs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API Documentation: All 127 public methods documented
README: Examples are current and accurate
CHANGELOG: Up to date with recent commits
Code comments: No TODO/FIXME in public APIs
Documentation: Consistent across all files
Version references: No hardcoded versions

✅ Ready for release
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Expected output if issues found:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Documentation Issues (dqx-docs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Missing Docstrings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. src/dqx/common.py:45 - validate_tags()
   Function: Validates and normalizes tag format
   Type: Public API function

   Required: Add Google-style docstring
   Template:
   """Validate and normalize tags.

   Args:
       tags: Set of tag strings to validate.

   Returns:
       frozenset[str]: Validated and normalized tags.

   Raises:
       ValueError: If any tag is invalid format.

   Example:
       >>> validate_tags({'user-facing', 'critical'})
       frozenset({'user-facing', 'critical'})
   """

2. src/dqx/display.py:78 - format_results()
   Function: Formats validation results for terminal display
   Type: Public API function
   Required: Add docstring with Args, Returns, Example

Outdated Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. README.md line 87
   Current API (incorrect):
   ctx.assert_that(x).gt(0)

   Should be (current API):
   ctx.assert_that(x).where(name="X is positive").is_gt(0)

   The API changed to require .where() before assertions

4. README.md line 124
   Hardcoded version number:
   pip install dqlib==0.5.13

   Should be (version-agnostic):
   pip install dqlib

Documentation Inconsistencies:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. docs/api.md mentions old method name
   Current: "check_value()"
   Should be: "validate_value()"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action Required:
- Add Google-style docstrings to 2 functions
- Update README examples at lines 87, 124
- Fix method name in docs/api.md
- Ensure documentation matches current API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Step 4: Comprehensive Release Readiness Report

Provide overall assessment based on all three agent validations:

### If ALL validations pass:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 Release Readiness Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent Validation Results:

✅ dqx-test
   • Tests: 1659 passed
   • Coverage: 100% (5088/5088 statements)
   • Status: Ready

✅ dqx-quality
   • Formatting: Pass
   • Linting: No issues
   • Type checking: All typed
   • Pre-commit: All 21 hooks passed
   • Status: Ready

✅ dqx-docs
   • API docs: Complete (127 methods)
   • README: Current
   • CHANGELOG: Up to date
   • Status: Ready

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Your code is ready for release!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next steps:
1. Run /release to start the release process
2. The same validation will run automatically
3. A PR will be created for manual review
4. After merge, use /release-tag to publish

All validation criteria met. No issues found.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### If ANY validation fails:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Release Readiness Issues
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent Validation Results:

❌ dqx-test
   • Coverage: 98.5% (46 statements missing)
   • See "Step 1" above for details
   • Status: Not ready

✅ dqx-quality
   • All checks passed
   • Status: Ready

❌ dqx-docs
   • Missing docstrings: 2
   • Outdated examples: 2
   • See "Step 3" above for details
   • Status: Not ready

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Summary of Issues
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests & Coverage (dqx-test):
• 46 uncovered statements across 2 files
• Add tests or use `# pragma: no cover`

Documentation (dqx-docs):
• 2 missing docstrings
• 2 outdated README examples
• Add docstrings and update examples

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 Recommended Actions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Fix test coverage:
   • Review detailed suggestions in "Step 1" above
   • Add tests for uncovered lines
   • Run: uv run pytest --cov=src/dqx --cov-report=term-missing

2. Update documentation:
   • Add Google-style docstrings to 2 functions
   • Update README examples at lines 87, 124
   • Review detailed guidance in "Step 3" above

3. Validate again:
   • After fixes: /release-validate
   • Iterate until all agents report success

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  Not ready for release - fix issues above
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Usage Examples

### Check before starting release:
```bash
# Check if ready
/release-validate

# Fix any issues reported
# Add tests, update docs, etc.

# Validate again
/release-validate

# When all pass, start release
/release
```

### Fast iteration:
```bash
# 1. Run validation
/release-validate

# 2. Fix reported issue
git add .
git commit -m "fix: add missing test coverage"

# 3. Validate again (repeat until ready)
/release-validate
```

### Before alpha release:
```bash
# Validate first
/release-validate

# If ready, create alpha
/release-alpha
```

## Important Notes

### Read-Only Operation
- This command only reads and reports
- Never modifies files or creates branches
- Safe to run multiple times
- No side effects

### Same Standards as Release
- Uses identical agents as `/release` and `/release-alpha`
- dqx-test: 100% coverage requirement
- dqx-quality: All 21 pre-commit hooks
- dqx-docs: Complete documentation

### Agent-Based Validation
- **dqx-test**: Detailed coverage analysis with suggestions
- **dqx-quality**: Specific line numbers and fix instructions
- **dqx-docs**: Documentation completeness with templates

### Fast Feedback Loop
1. Run `/release-validate`
2. Get detailed agent feedback
3. Fix specific issues
4. Run `/release-validate` again
5. Repeat until all agents pass
6. Run `/release` with confidence

### When to Use
- **Before release**: Check readiness before starting `/release`
- **During development**: Ensure ongoing release readiness
- **After fixes**: Verify issues are resolved
- **Continuous validation**: Part of development workflow

### Relation to Other Commands
- **This command** (`/release-validate`): Pre-flight check
- **`/release`**: Actual release process (includes same validation)
- **`/release-alpha`**: Alpha release (includes same validation)
- **`/release-tag`**: Tag creation after PR merge

## Tips

### Use as Pre-Commit Check
Run before committing to ensure changes maintain release readiness:
```bash
# Before commit
/release-validate

# If issues, fix them
# Then commit
```

### CI Integration
Consider running this validation in CI/CD for pull requests:
```bash
# In CI pipeline
uv run opencode /release-validate
```

### Incremental Fixes
Don't try to fix everything at once:
```bash
# Focus on one agent at a time
/release-validate  # See all issues
# Fix test coverage first
/release-validate  # Now fix docs
/release-validate  # All pass!
```

## Exit Codes

The command should set appropriate exit codes for CI integration:
- **0**: All validations pass (ready for release)
- **1**: One or more validations failed (not ready)
- **2**: Command error (unable to run validation)

This allows scripting:
```bash
if /release-validate; then
    echo "Ready for release"
else
    echo "Not ready - fix issues first"
fi
```

---

This command completes the release workflow by providing a pre-flight validation check that uses the same specialized agents and standards as the actual release process.
