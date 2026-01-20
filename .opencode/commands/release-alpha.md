---
description: Create alpha prerelease
agent: build
---

Execute an automated alpha prerelease workflow for DQX. This creates a prerelease version for testing without affecting the main release line.

## Overview

Alpha releases are used for:
- Testing new features before full release
- Validating CI/CD pipeline changes
- Allowing early adopters to test changes
- Not suitable for production use

**Validation:** Alpha releases use the same specialized agents (dqx-test, dqx-quality, dqx-docs) as stable releases to ensure quality.

## Step 1: Determine Alpha Version

Current version from pyproject.toml:
!`grep '^version = ' pyproject.toml | cut -d '"' -f2`

Recent commits since last tag:
!`git log $(git describe --tags --abbrev=0 2>/dev/null || echo HEAD~10)..HEAD --oneline | head -10`

Calculate the next alpha version:
- If current version is `0.5.13`, next alpha should be `0.5.14a1`
- If an alpha already exists (e.g., `0.5.14a1`), increment to `0.5.14a2`
- Format: `{major}.{minor}.{patch}a{alpha_number}`

**Action:** Determine the appropriate alpha version number and ask user to confirm.

## Step 2: Generate Concise CHANGELOG Entry

Parse recent commits since last release tag:
!`git log $(git describe --tags --abbrev=0 2>/dev/null || echo HEAD~10)..HEAD --format="%s"`

Create a concise CHANGELOG entry (max 50 lines) with this format:

```markdown
## v{version} (YYYY-MM-DD)

### Feat
- **scope**: brief description (#PR if available)

### Fix
- **scope**: brief description

### Refactor
- **scope**: brief description

### Docs
- **scope**: brief description
```

Rules for conciseness:
- Only include meaningful changes (skip chore, style, test commits unless significant)
- One line per feature/fix
- Group related changes together
- Omit commit details that don't affect users
- For alpha releases, focus on what's being tested

**Note:** Since this is an alpha, the CHANGELOG entry can be brief and high-level.

## Step 2.5: Validate CHANGELOG with dqx-docs

Delegate to **dqx-docs agent** to validate the alpha CHANGELOG entry:

**Agent task:** Validate CHANGELOG format and content for alpha release

**Validation criteria:**
- Entry is max 50 lines
- Follows conventional commits format
- Clearly marked as alpha/prerelease
- Uses user-facing language
- Proper markdown formatting

**Expected agent response:**
```
✅ CHANGELOG validation passed (alpha release)
- Format: Correct
- Length: 28 lines (within 50 line limit)
- Prerelease marking: Clearly indicates alpha status
- Language: User-facing and clear
```

**If validation fails:**
```
⚠️ CHANGELOG validation issues:

1. Missing prerelease indicator
   Add note that this is an alpha release for testing

2. Length: 62 lines (exceeds 50 line limit)
   Condense entries, focus on key changes being tested

3. Technical jargon at line 15
   Make language more user-friendly
```

**Action:** Fix issues and re-validate before proceeding.

## Step 3: Update Version Files

1. Update `pyproject.toml`:
   - Change version field to the alpha version

2. Update `uv.lock`:
   - Run: `uv lock --no-upgrade`

3. Update `CHANGELOG.md`:
   - Insert the new alpha entry at the top (line 1)
   - Use today's date in YYYY-MM-DD format

**Important:** Do NOT modify any other files.

## Step 4: Create Release Branch

1. Ensure we're on main branch with latest changes:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Create release branch:
   ```bash
   git checkout -b release/v{alpha_version}
   ```

3. Stage modified files:
   ```bash
   git add CHANGELOG.md pyproject.toml uv.lock
   ```

4. Commit with conventional format:
   ```bash
   git commit -m "chore(release): bump version to {alpha_version}"
   ```

## Step 5: Run Validation Suite

Execute validation using specialized agents (same standards as stable releases).

### 5a. Test Validation

Delegate to **dqx-test agent** to validate test coverage:

**Agent task:** Run full test suite and verify 100% coverage

**Commands for agent:**
- `uv run pytest` - Run all tests
- `uv run pytest --cov=src/dqx --cov-report=term-missing` - Generate coverage

**Success criteria:**
- All tests pass (~1659+ tests expected)
- Coverage is exactly 100% (5088/5088 statements expected)

**Expected agent response:**
```
✅ Test validation passed
- Tests: 1659 passed, 0 failed
- Coverage: 100% (5088/5088 statements)
```

**If validation fails:**
```
❌ Test validation failed

{detailed_agent_feedback}

Alpha releases require the same test standards as stable releases.
Fix issues before proceeding.
```

**ABORT if dqx-test reports failure.**

### 5b. Quality Validation

Delegate to **dqx-quality agent** to validate code quality:

**Agent task:** Run all quality checks and pre-commit hooks

**Commands for agent:**
- `uv run ruff format`
- `uv run ruff check --fix`
- `uv run mypy src tests`
- `uv run pre-commit run --all-files`

**Success criteria:**
- All 21 pre-commit hooks pass
- No formatting, linting, or type errors

**Expected agent response:**
```
✅ Quality validation passed
- Formatting: All files correct
- Linting: No issues
- Type checking: All properly typed
- Pre-commit: All 21 hooks passed
```

**If validation fails:**
```
❌ Quality validation failed

{detailed_agent_feedback}

Alpha releases require the same quality standards as stable releases.
Fix issues before proceeding.
```

**ABORT if dqx-quality reports failure.**

### 5c. Documentation Validation

Delegate to **dqx-docs agent** to validate documentation:

**Agent task:** Verify documentation completeness for alpha release

**Checks for agent:**
- CHANGELOG clearly marks this as alpha
- Any new features are documented
- Alpha release warnings are present where appropriate

**Success criteria:**
- Documentation clearly indicates alpha status
- New features documented (even if experimental)

**Expected agent response:**
```
✅ Documentation validation passed (alpha)
- CHANGELOG: Clearly marked as alpha release
- New features: Documented with experimental notes
- Warnings: Alpha status properly indicated
```

**If validation fails:**
```
❌ Documentation validation failed

{detailed_agent_feedback}

Ensure alpha status is clearly communicated in documentation.
```

**ABORT if dqx-docs reports failure.**

### 5d. Validation Summary

**All three agents must report success:**

```
✅ dqx-test: All tests pass, 100% coverage
✅ dqx-quality: All quality checks pass
✅ dqx-docs: Documentation complete (alpha marked)
```

**If any validation fails:**
```
❌ Alpha Release Validation Failed

Agent results:
- dqx-test: {status}
- dqx-quality: {status}
- dqx-docs: {status}

Alpha releases require the same quality standards as stable releases.

Action required:
1. Review detailed agent feedback above
2. Fix all reported issues
3. Delete local branch: git branch -D release/v{alpha_version}
4. Re-run /release-alpha after fixes

RELEASE ABORTED
```

## Step 6: Push Branch and Create PR

1. Push the release branch:
   ```bash
   git push -u origin release/v{alpha_version}
   ```

2. Create a pull request using GitHub CLI:
   ```bash
   gh pr create --title "chore(release): release alpha version {alpha_version}" --body "<PR_BODY>"
   ```

**PR Body Template:**
```markdown
## Summary

Alpha prerelease: **v{alpha_version}**

This is a prerelease version for testing purposes. It will be published to PyPI as a prerelease and marked as such on GitHub.

## Changes Since Last Release

<Insert concise CHANGELOG entries here>

## Verification

✅ **All pre-commit checks passed** (validated by dqx-quality)
✅ **All tests passed:** {test_count} tests (validated by dqx-test)
✅ **Coverage:** 100% (validated by dqx-test)
✅ **Documentation complete** (validated by dqx-docs)

## Agent Validation

This alpha release was validated by specialized DQX agents:
- **dqx-test**: Test suite and coverage validation
- **dqx-quality**: Code quality and standards enforcement
- **dqx-docs**: Documentation completeness verification

All validation criteria met (same standards as stable releases).

## Alpha Release Process

After merging this PR:
1. User will manually create and push the tag `v{alpha_version}`
2. GitHub Actions will automatically:
   - Build the package
   - Test on Python 3.11, 3.12, 3.13
   - Publish to PyPI (marked as prerelease)
   - Create GitHub Release (marked as prerelease)

## Testing

To test this alpha version:
```bash
pip install dqlib=={alpha_version}
```

## Related

- Prerelease for testing new features
- Will not affect stable release channel
- Not recommended for production use
```

3. Display the PR URL to the user

**Do NOT enable auto-merge.** User will review and merge manually.

## Step 7: Display Summary and Next Steps

```
🎉 Alpha Release Preparation Complete!

📊 Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Version:        v{alpha_version}
Branch:         release/v{alpha_version}
CHANGELOG:      ✅ Updated ({line_count} lines)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 Agent Validation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dqx-test:       ✅ {test_count} tests passed, 100% coverage
dqx-quality:    ✅ All 21 pre-commit hooks passed
dqx-docs:       ✅ Documentation complete (alpha marked)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 Pull Request:
{pr_url}

📋 Next Steps:

1. **Review the PR** (click link above)
   - Verify alpha version number is correct
   - Check CHANGELOG entries
   - Confirm all validation passed

2. **Merge the PR** when ready

3. **Create and push the tag:**
   After merging, run:

   git checkout main
   git pull origin main
   git tag -a v{alpha_version} -m "Alpha release {alpha_version}"
   git push origin v{alpha_version}

   Or use: /release-tag

4. **Monitor the release:**
   - GitHub Actions: https://github.com/{owner}/{repo}/actions
   - PyPI: https://pypi.org/project/dqlib/
   - Will be marked as prerelease

⏱️  Estimated publish time: ~2 minutes after tag push

📦 Users can test with: pip install dqlib=={alpha_version}
```

## Important Notes

### Alpha Release Characteristics
- Version format: `0.5.14a1` (PEP 440 compliant)
- Marked as prerelease on GitHub
- Published to PyPI as prerelease
- Can be installed with: `pip install dqlib==0.5.14a1`
- Does not affect stable version channel
- Users must explicitly specify alpha version to install

### Validation is Critical
- Alpha releases use the same validation as stable releases
- All tests must pass (dqx-test)
- 100% coverage is required (dqx-test)
- All pre-commit hooks must pass (dqx-quality)
- Documentation must be complete (dqx-docs)
- No shortcuts for alpha releases

### Agent-Based Validation
- **dqx-test**: Enforces 100% coverage
- **dqx-quality**: Enforces all code quality standards
- **dqx-docs**: Ensures documentation completeness

All three agents must report success before PR creation.

### Branch Cleanup
After tag is created, the release branch can be deleted:
```bash
git branch -d release/v{alpha_version}
git push origin --delete release/v{alpha_version}
```

## Error Handling

### dqx-test agent reports failure:
```
❌ Test Validation Failed

The dqx-test agent reported issues:
{detailed_agent_feedback}

Action required:
1. Fix test failures or add coverage
2. Delete release branch: git branch -D release/v{alpha_version}
3. Re-run /release-alpha after fixes

ALPHA RELEASE ABORTED
```

### dqx-quality agent reports failure:
```
❌ Quality Validation Failed

The dqx-quality agent reported issues:
{detailed_agent_feedback}

Action required:
1. Fix quality issues (formatting, linting, types)
2. Delete release branch: git branch -D release/v{alpha_version}
3. Re-run /release-alpha after fixes

ALPHA RELEASE ABORTED
```

### dqx-docs agent reports failure:
```
❌ Documentation Validation Failed

The dqx-docs agent reported issues:
{detailed_agent_feedback}

Action required:
1. Fix documentation issues
2. Ensure alpha status is clearly marked
3. Delete release branch: git branch -D release/v{alpha_version}
4. Re-run /release-alpha after fixes

ALPHA RELEASE ABORTED
```

## Success Criteria

✅ Alpha version determined correctly
✅ CHANGELOG generated and validated by dqx-docs (≤50 lines)
✅ dqx-test validated: All tests pass, 100% coverage
✅ dqx-quality validated: All pre-commit hooks pass
✅ dqx-docs validated: Documentation complete, alpha marked
✅ Release branch created
✅ PR created with clear description
✅ User can review and merge at their convenience

This workflow ensures alpha releases meet the same quality standards as stable releases while clearly marking them as prereleases for testing.
