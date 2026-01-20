---
description: Create stable release with PR
agent: build
---

Execute the complete stable release workflow for DQX.

## Overview

This command automates the release process with specialized agent validation:
- Interactive version selection (patch/minor/major)
- Auto-generated concise CHANGELOG (max 50 lines, validated by dqx-docs)
- Full validation (dqx-test, dqx-quality, dqx-docs)
- PR creation for manual review and merge
- Clear next steps for tag creation

## Step 1: Interactive Version Selection

Current version from pyproject.toml:
!`grep '^version = ' pyproject.toml | cut -d '"' -f2`

Recent commits since last tag:
!`git log $(git describe --tags --abbrev=0 2>/dev/null || echo HEAD~10)..HEAD --format="%s" | head -20`

**Analyze commits and suggest version bump:**

Count commit types since last release:
- BREAKING CHANGE commits → suggests **major** bump (0.5.x → 0.6.0 or 1.0.0)
- `feat:` commits → suggests **minor** bump (0.5.13 → 0.6.0)
- `fix:`, `perf:`, `refactor:` → suggests **patch** bump (0.5.13 → 0.5.14)

**Ask user to choose:**
```
Based on recent commits, I suggest a {suggested_type} version bump.

Choose version bump type:
1. patch - Bug fixes, small changes (0.5.13 → 0.5.14)
2. minor - New features, non-breaking changes (0.5.13 → 0.6.0)
3. major - Breaking changes (0.5.13 → 1.0.0)

Your choice (1/2/3):
```

Calculate next version based on user's choice:
- patch: increment patch number
- minor: increment minor, reset patch to 0
- major: increment major, reset minor and patch to 0 (only if major_version_zero is false)

**Note:** Since this project uses `major_version_zero = true`, major bumps stay at 0.x.0.

## Step 2: Generate Concise CHANGELOG

Extract commits since last tag:
!`git log $(git describe --tags --abbrev=0)..HEAD --format="%h %s %b"`

**Parse and categorize commits:**

Group by conventional commit type:
- **BREAKING CHANGE**: Any commit with "BREAKING CHANGE:" in body
- **Feat**: Commits starting with `feat:`
- **Fix**: Commits starting with `fix:`
- **Refactor**: Commits starting with `refactor:`
- **Perf**: Commits starting with `perf:`
- **Docs**: Commits starting with `docs:`
- **Build/CI/Chore**: Other commits (only if significant)

**CHANGELOG Format (max 50 lines):**

```markdown
## v{version} (YYYY-MM-DD)

### BREAKING CHANGE

- Brief description of breaking change
- Migration: One-line migration instruction

### Feat

- **scope**: brief feature description (#PR)
- **scope**: another feature (#PR)

### Fix

- **scope**: bug fix description (#PR)
- **scope**: another fix

### Refactor

- **scope**: refactoring description

### Docs

- **scope**: documentation update
```

**Rules for conciseness:**
1. Max 50 total lines for the entire entry
2. Skip trivial commits (typos, formatting, test-only changes)
3. Combine related changes: "update X, Y, and Z" instead of 3 separate lines
4. Omit commit hashes unless necessary
5. Extract PR numbers from commit messages when available
6. For breaking changes, keep migration instructions to 1-2 lines max
7. If >50 lines, prioritize: BREAKING CHANGE > Feat > Fix > Other

## Step 2.5: Validate CHANGELOG with dqx-docs

After generating the CHANGELOG entry, delegate to the **dqx-docs agent** to validate:

**Agent task:** Validate the generated CHANGELOG entry

**Validation criteria:**
- Entry is max 50 lines (strict requirement)
- Follows conventional commits format
- Uses user-facing language (not developer jargon)
- Breaking changes have clear migration paths
- Each entry is concise and meaningful
- No internal/trivial changes included
- Proper markdown formatting

**Expected agent response:**
```
✅ CHANGELOG validation passed
- Format: Correct (conventional commits)
- Length: 42 lines (within 50 line limit)
- Language: User-facing and clear
- Breaking changes: Documented with migration notes
```

**If validation fails:**
```
⚠️ CHANGELOG validation issues:

1. Length: 68 lines (exceeds 50 line limit by 18 lines)
   Suggestion: Combine similar changes, remove trivial entries

2. Language: Line 23 uses technical jargon
   Current: "Refactored SymPy expr processing in analyzer"
   Suggest: "Improved SQL query generation performance"

3. Missing migration: Breaking change at line 8 has no migration guide
   Add: "Migration: Replace old_method() with new_method(param)"
```

**Action:** Fix CHANGELOG issues and have dqx-docs re-validate before proceeding.

## Step 3: Update Version Files

1. **Update pyproject.toml:**
   ```bash
   # Change version field to new version
   ```

2. **Update uv.lock:**
   ```bash
   uv lock --no-upgrade
   ```

3. **Update CHANGELOG.md:**
   - Insert generated entry at line 1 (top of file)
   - Use today's date in format: YYYY-MM-DD
   - Ensure proper markdown formatting

4. **Verify changes:**
   - pyproject.toml has correct version
   - uv.lock is updated
   - CHANGELOG.md has new entry at top

## Step 3.5: Validate Version Consistency with dqx-docs

Delegate to **dqx-docs agent** to verify version consistency across documentation:

**Agent task:** Check version consistency in documentation files

**Files to check:**
- README.md examples
- docs/ files (if any reference version)
- Any code comments with version numbers
- Installation instructions

**Expected agent response:**
```
✅ Version consistency validated
- README.md: Installation examples don't hardcode version ✓
- Documentation: No hardcoded version references ✓
- Code comments: No outdated version mentions ✓
```

**If issues found:**
```
⚠️ Version consistency issues:

1. README.md line 87: Hardcoded version in example
   Current: pip install dqlib==0.5.13
   Update to: pip install dqlib  (or use new version)

2. docs/installation.md line 23: References old version
   Update version number to {new_version}
```

**Action:** Fix version inconsistencies before proceeding.

## Step 4: Create Release Branch

```bash
# Ensure on main with latest changes
git checkout main
git pull origin main

# Create release branch
git checkout -b release/v{version}

# Stage files
git add CHANGELOG.md pyproject.toml uv.lock

# Commit with conventional format
git commit -m "chore(release): bump version from {old_version} to {version}"
```

## Step 5: Run Full Validation Suite

Execute comprehensive validation by delegating to specialized agents.

### 5a. Test Coverage Validation

Delegate to **dqx-test agent** to validate test coverage:

**Agent task:** Run full test suite and verify 100% coverage

**Commands for agent:**
- `uv run pytest` - Run all tests
- `uv run pytest --cov=src/dqx --cov-report=term-missing` - Generate coverage report

**Success criteria:**
- All tests pass (~1659+ tests expected)
- Coverage is exactly 100% (5088/5088 statements expected)
- No test failures or errors

**Expected agent response:**
```
✅ Test validation passed
- Tests: 1659 passed, 0 failed
- Coverage: 100% (5088/5088 statements)
- Duration: ~45 seconds
```

**If validation fails:**
```
❌ Test validation failed

Coverage: 98.5% (5042/5088 statements)

Uncovered lines:
- src/dqx/api.py:123-125 (3 lines)
  Code: Error handling for invalid input
  Suggested test: Test with None parameter to trigger error path

- src/dqx/validator.py:456 (1 line)
  Code: Edge case for empty dataset
  Suggested test: Call validate() with empty pa.Table

Action required:
1. Add test for api.py:123-125 error path
2. Add test for validator.py:456 edge case
OR add `# pragma: no cover` if unreachable
```

**ABORT if dqx-test reports failure. Do not proceed to quality validation.**

### 5b. Code Quality Validation

Delegate to **dqx-quality agent** to validate code quality:

**Agent task:** Run all quality checks and pre-commit hooks

**Commands for agent:**
- `uv run ruff format` - Check formatting
- `uv run ruff check --fix` - Check linting (with auto-fix)
- `uv run mypy src tests` - Check types
- `uv run pre-commit run --all-files` - Run all 21 hooks

**Success criteria:**
- All files properly formatted
- No linting errors
- No type errors
- All 21 pre-commit hooks pass

**Expected agent response:**
```
✅ Quality validation passed
- Formatting: All files formatted correctly
- Linting: No issues found
- Type checking: All functions properly typed
- Pre-commit: All 21 hooks passed
```

**If validation fails:**
```
❌ Quality validation failed

Linting issues:
- src/dqx/provider.py:89:5
  Error: F841 - Local variable 'x' is assigned but never used
  Fix: Remove unused variable or prefix with underscore: _x

- src/dqx/analyzer.py:234:80
  Error: E501 - Line too long (125 > 120 characters)
  Fix: Break line into multiple lines or refactor expression

Type errors:
- src/dqx/validator.py:123
  Error: Function missing return type annotation
  Fix: Add return type: def validate(self) -> ResultKey:

Action required:
1. Fix linting errors (run `uv run ruff check --fix`)
2. Add missing type annotation at validator.py:123
3. Re-run validation
```

**ABORT if dqx-quality reports failure. Do not proceed to documentation validation.**

### 5c. Documentation Validation

Delegate to **dqx-docs agent** to validate documentation completeness:

**Agent task:** Verify all documentation is complete and accurate

**Checks for agent:**
- All public APIs have Google-style docstrings
- README examples are current and working
- CHANGELOG format is correct (should already be validated in Step 2.5)
- No TODO or FIXME comments in public API code
- Documentation consistency across files
- All new features are documented

**Success criteria:**
- All public methods have complete docstrings
- Examples in README match current API
- No missing documentation

**Expected agent response:**
```
✅ Documentation validation passed
- API Documentation: All 127 public methods documented
- README: Examples are current and accurate
- CHANGELOG: Properly formatted (validated earlier)
- Code comments: No TODO/FIXME in public APIs
- Documentation: Consistent across all files
```

**If validation fails:**
```
❌ Documentation validation failed

Missing docstrings:
- src/dqx/common.py:45 - validate_tags()
  Function: Validates tag format
  Required: Add Google-style docstring with Args, Returns, Example

- src/dqx/provider.py:123 - process_metrics()
  Function: Internal metric processing
  Note: Private function, should prefix with underscore: _process_metrics()

- src/dqx/display.py:78 - format_results()
  Function: Formats validation results for display
  Required: Add docstring with description and example

Outdated examples:
- README.md line 87
  Current: ctx.assert_that(x).gt(0)
  Should be: ctx.assert_that(x).where(name="X is positive").is_gt(0)

Documentation inconsistencies:
- docs/api.md mentions old method name "check_value()"
  Update to current method name: "validate_value()"

Action required:
1. Add Google-style docstrings to 3 functions
2. Update README example at line 87
3. Fix documentation inconsistency in api.md
```

**ABORT if dqx-docs reports failure.**

### 5d. Validation Summary

**All three agents must report success to proceed:**

```
✅ dqx-test: All tests pass, 100% coverage (5088/5088 statements)
✅ dqx-quality: All quality checks pass (21/21 hooks)
✅ dqx-docs: Documentation complete and accurate
```

**If ANY agent reports failure:**
```
❌ Release Validation Failed

The release branch has been created locally but not pushed.

Agent results:
- dqx-test: {status}
- dqx-quality: {status}
- dqx-docs: {status}

Action required:
1. Review detailed agent feedback above
2. Fix all reported issues
3. Delete local release branch: git branch -D release/v{version}
4. Re-run /release after fixes

RELEASE ABORTED - No PR created, no remote changes made.
```

## Step 6: Push Branch and Create PR

```bash
# Push release branch
git push -u origin release/v{version}

# Create PR with detailed description
gh pr create --title "chore(release): release version {version}" --body "<PR_BODY>"
```

**PR Body Template:**

```markdown
## Summary

Release version **v{version}** ({patch|minor|major} bump)

## Changes

{Insert concise CHANGELOG entries here - the auto-generated ones}

## Verification

✅ **All pre-commit checks passed** (validated by dqx-quality)
- Formatting (ruff format)
- Linting (ruff check)
- Type checking (mypy)
- All 21 pre-commit hooks

✅ **All tests passed:** {test_count}/{test_count} tests (validated by dqx-test)
✅ **Coverage:** 100% ({statement_count}/{statement_count} statements) (validated by dqx-test)
✅ **Documentation complete** (validated by dqx-docs)
- All public APIs documented
- README examples current
- CHANGELOG properly formatted

## Breaking Changes

{If any breaking changes, list them here with migration notes}

{If none: "No breaking changes in this release."}

## Release Process

After merging this PR:

1. **Create the release tag:**
   ```bash
   git checkout main && git pull origin main
   git tag -a v{version} -m "Release version {version}"
   git push origin v{version}
   ```

   Or use: `/release-tag`

2. **GitHub Actions will automatically:**
   - Build the package
   - Test on Python 3.11, 3.12, 3.13
   - Publish to PyPI
   - Create GitHub Release with notes
   - Send notifications

3. **Verify publication:**
   - PyPI: https://pypi.org/project/dqlib/{version}/
   - GitHub Release: https://github.com/{owner}/{repo}/releases/tag/v{version}

## Installation

After release, users can install with:
```bash
pip install dqlib=={version}
# or
pip install --upgrade dqlib
```

## Agent Validation

This release was validated by specialized DQX agents:
- **dqx-test**: Test suite and coverage validation
- **dqx-quality**: Code quality and standards enforcement
- **dqx-docs**: Documentation completeness verification

All validation criteria met before PR creation.

## Related

- Previous release: v{old_version}
- CHANGELOG: See full details in CHANGELOG.md
- Documentation: [Update if needed]

---

**Ready for review.** Please merge when approved.
```

## Step 7: Display Summary and Next Steps

```
🎉 Release Preparation Complete!

📊 Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Version:        {old_version} → v{version} ({bump_type})
Branch:         release/v{version}
CHANGELOG:      ✅ Updated ({line_count} lines)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 Agent Validation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dqx-test:       ✅ {test_count} tests passed, 100% coverage
dqx-quality:    ✅ All 21 pre-commit hooks passed
dqx-docs:       ✅ Documentation complete and accurate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 Pull Request:
{pr_url}

📋 Next Steps:

1. **Review the PR** (click link above)
   - Verify CHANGELOG entries are correct
   - Check version bump is appropriate
   - Confirm all validation passed

2. **Merge the PR** when ready
   - Use "Squash and merge" or "Merge commit"
   - PR title will become the commit message

3. **Create and push the tag:**
   After merging, run these commands:

   git checkout main
   git pull origin main
   git tag -a v{version} -m "Release version {version}"
   git push origin v{version}

   Or use: /release-tag

4. **Monitor the release:**
   - GitHub Actions: https://github.com/{owner}/{repo}/actions
   - Expected time: ~2 minutes
   - PyPI: https://pypi.org/project/dqlib/

⏱️  Estimated total time: ~5 minutes (after PR merge)

🚀 The release workflow is ready!
```

## Important Notes

### Version Numbering (PEP 440)
- Patch: 0.5.13 → 0.5.14 (bug fixes)
- Minor: 0.5.13 → 0.6.0 (new features)
- Major: 0.5.13 → 0.6.0 (breaking changes, stays at 0.x due to major_version_zero)

### Agent-Based Validation
- **dqx-test**: Enforces 100% coverage (no exceptions)
- **dqx-quality**: Enforces all code quality standards
- **dqx-docs**: Ensures documentation completeness

All three agents must report success before PR creation.

### CHANGELOG Conciseness
- Target: 30-40 lines per release
- Maximum: 50 lines (enforced by dqx-docs)
- Focus on user-facing changes
- Omit internal refactoring unless significant
- Breaking changes are always included

### Manual Review Required
- PR must be reviewed and merged manually
- No auto-merge to ensure human oversight
- Tag creation is manual after merge
- This ensures quality control

### Validation is Non-Negotiable
- 100% test coverage is required (dqx-test)
- All pre-commit hooks must pass (dqx-quality)
- Documentation must be complete (dqx-docs)
- Any failure aborts the release

### Branch Cleanup
After tag is created and release is published:
```bash
git branch -d release/v{version}
git push origin --delete release/v{version}
```

## Error Handling

### Not on main branch:
```
❌ Must be on main branch to create release

Current branch: {current_branch}

Switch to main:
    git checkout main
    git pull origin main
```

### Dirty working directory:
```
❌ Working directory has uncommitted changes

Please commit or stash changes first:
    git status
```

### dqx-test agent reports failure:
```
❌ Test Validation Failed

The dqx-test agent reported issues:
{detailed_agent_feedback}

Common issues:
- Coverage < 100%: Add tests to cover missing lines
- Failing tests: Fix implementation or test logic
- Test errors: Check fixtures, mocks, test data

Action required:
1. Review detailed feedback above
2. Fix the issues (add tests or fix code)
3. Delete release branch: git branch -D release/v{version}
4. Re-run /release after fixes

RELEASE ABORTED
```

### dqx-quality agent reports failure:
```
❌ Quality Validation Failed

The dqx-quality agent reported issues:
{detailed_agent_feedback}

Common issues:
- Type errors: Add missing type annotations
- Linting errors: Fix code style issues
- Formatting: Run `uv run ruff format`
- Pre-commit hooks: Fix specific hook failures

Action required:
1. Review detailed feedback above with line numbers
2. Fix quality issues (run suggested commands)
3. Delete release branch: git branch -D release/v{version}
4. Re-run /release after fixes

RELEASE ABORTED
```

### dqx-docs agent reports failure:
```
❌ Documentation Validation Failed

The dqx-docs agent reported issues:
{detailed_agent_feedback}

Common issues:
- Missing docstrings: Add Google-style documentation
- Outdated examples: Update README to match current API
- CHANGELOG too long: Condense to max 50 lines
- Inconsistent docs: Fix version numbers or method names

Action required:
1. Review detailed feedback with file:line references
2. Add/update documentation as specified
3. Delete release branch: git branch -D release/v{version}
4. Re-run /release after fixes

RELEASE ABORTED
```

### CHANGELOG too long (caught by dqx-docs):
```
⚠️ CHANGELOG Validation Failed

The dqx-docs agent reported:
CHANGELOG entry is {line_count} lines (max 50 allowed)

Exceeds limit by {excess} lines

Suggestions from dqx-docs:
- Combine similar changes into single entries
- Remove internal/trivial changes
- Shorten descriptions to be more concise
- Group related features together

Continue anyway? (yes/no)
```

## Success Criteria

✅ Version bumped appropriately
✅ CHANGELOG generated and validated by dqx-docs (≤50 lines)
✅ dqx-test validated: All tests pass, 100% coverage
✅ dqx-quality validated: All pre-commit hooks pass
✅ dqx-docs validated: Documentation complete
✅ Release branch created
✅ PR created with comprehensive description
✅ Next steps clearly communicated
✅ User can review before merging

This workflow ensures quality through specialized agent validation while keeping the release process streamlined.
