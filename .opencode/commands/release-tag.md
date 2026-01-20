---
description: Create tag after release PR merge
agent: build
---

Create and push a release tag after the PR has been merged.

## Prerequisites

- Release PR must be merged to main
- You must be on the main branch
- Working directory must be clean

## Step 1: Verify Current State

Check current branch:
!`git branch --show-current`

Check git status:
!`git status --short`

Get current version from pyproject.toml:
!`grep '^version = ' pyproject.toml | cut -d '"' -f2`

**Validations:**
- Must be on `main` branch
- Working directory must be clean (no uncommitted changes)
- Version in pyproject.toml should match the version to be tagged

## Step 1.5: Optional Final Smoke Test

Optionally delegate to **dqx-test agent** for a final smoke test before creating tag:

**Agent task:** Quick smoke test to verify main branch is stable

**Commands for agent:**
- `uv run pytest -x` - Run tests, stop on first failure (fast feedback)

**Purpose:**
- Catch any issues that might have been introduced during PR merge
- Quick sanity check before tag creation
- Optional but recommended for critical releases

**Expected agent response:**
```
✅ Smoke test passed
- Quick test run completed successfully
- No failures detected
- Safe to create tag
```

**If smoke test fails:**
```
⚠️ Smoke test failed

{agent_feedback}

This is unexpected since the PR should have passed all tests.
Possible causes:
- Merge conflict resolution introduced an issue
- Main branch was modified after PR
- Environment-specific issue

Recommendation: Investigate before creating tag
```

**Note:** This is a quick check, not full validation (which was done in PR).

## Step 2: Pull Latest Changes

Ensure main is up to date:
```bash
git pull origin main
```

Verify the release commit is present:
!`git log --oneline -5`

## Step 3: Determine Tag Version

The tag version should match the version in pyproject.toml.

Ask the user: "The current version in pyproject.toml is {version}. Create tag v{version}? (yes/no)"

If the user confirms, proceed to tag creation.

## Step 4: Create Annotated Tag

Create an annotated tag with a descriptive message:

```bash
git tag -a v{version} -m "Release version {version}"
```

Verify the tag was created:
```bash
git tag -l v{version}
```

## Step 5: Push Tag to Origin

Push the tag to trigger the release workflow:

```bash
git push origin v{version}
```

**This will trigger GitHub Actions to:**
1. Build the package
2. Test on Python 3.11, 3.12, 3.13
3. Publish to PyPI (marked as prerelease if alpha/beta/rc)
4. Create GitHub Release

**Note:** The GitHub Actions workflow includes the same quality checks (dqx-test and dqx-quality standards) that were validated in the PR.

## Step 6: Monitor Release Workflow

Show the release workflow status:
!`gh run list --workflow=Release --limit 1`

Get the workflow run URL:
!`gh run list --workflow=Release --limit 1 --json url --jq '.[0].url'`

**What to monitor:**

The GitHub Actions workflow will:
1. **Build**: Create distribution packages
2. **Test**: Run on Python 3.11, 3.12, 3.13
   - Same dqx-test standards (100% coverage)
   - Same dqx-quality standards (all hooks pass)
3. **Publish**: Upload to PyPI
4. **Release**: Create GitHub Release with notes
5. **Notify**: Send notifications

**Expected duration:** ~2 minutes

## Step 7: Provide Summary

Display success message with all relevant links:

```
🎉 Release Tag Created!

🏷️  Tag: v{version}
📦 Package will be published to PyPI

🔗 Links:
- GitHub Release: https://github.com/{owner}/{repo}/releases/tag/v{version}
- GitHub Actions: {workflow_url}
- PyPI: https://pypi.org/project/dqlib/{version}/

📊 Status:
- Tag pushed: ✅
- Workflow triggered: ✅
- Expected publish time: ~2 minutes

📥 Installation:
Users can install with:
    pip install dqlib=={version}
    # or
    pip install --upgrade dqlib

🔍 Monitor:
Watch the release progress at: {workflow_url}

The GitHub Actions workflow includes the same validation that was done in the PR:
- dqx-test: Verifies tests and coverage
- dqx-quality: Verifies code quality
- All tests run on Python 3.11, 3.12, 3.13

{If alpha/beta/rc: The package will be marked as "Pre-release" on PyPI and GitHub.}
```

## Error Handling

### If not on main branch:
```
❌ Error: Not on main branch

Current branch: {current_branch}

Please checkout main first:
    git checkout main
    git pull origin main

Then run this command again.
```

### If working directory is dirty:
```
❌ Error: Working directory has uncommitted changes

Please commit or stash your changes first:
    git status

Then run this command again.
```

### If version mismatch:
```
⚠️  Warning: Version mismatch

Expected tag: v{expected_version}
pyproject.toml: {current_version}

The tag version should match pyproject.toml.
Please verify before proceeding.
```

### If tag already exists:
```
❌ Error: Tag v{version} already exists

Existing tags:
!`git tag -l | grep {version}`

If you need to recreate the tag:
    git tag -d v{version}
    git push origin --delete v{version}

Then run this command again.

⚠️  Warning: Deleting and recreating tags can confuse PyPI.
Only do this if absolutely necessary.
```

### If smoke test fails (optional):
```
⚠️  Smoke Test Failed

The dqx-test agent reported issues on main branch:
{agent_feedback}

This is unexpected since the PR should have passed all tests.

Recommended actions:
1. Investigate the failure
2. Run full test suite locally: uv run pytest
3. Check if main branch was modified after PR merge
4. Fix any issues before creating tag

Do you want to create the tag anyway? (yes/no)
```

## Notes

- This command works for both stable and prerelease (alpha/beta/rc) tags
- For stable releases, use after `/release` PR is merged
- For alpha releases, use after `/release-alpha` PR is merged
- Tags trigger GitHub Actions automatically
- Prerelease flag is automatically set based on tag name (alpha/beta/rc)
- Deleting and recreating tags should be avoided when possible

## Agent Integration

### Optional Smoke Test (dqx-test)
- Quick validation before tag creation
- Catches merge-related issues early
- Not a full validation (PR already validated)
- Recommended for critical releases

### GitHub Actions Validation
The release workflow triggered by the tag includes:
- **dqx-test standards**: All tests pass, 100% coverage
- **dqx-quality standards**: All pre-commit hooks pass
- **Multi-version testing**: Python 3.11, 3.12, 3.13

The same validation performed in the PR is verified again in CI.

## Related Commands

- **Before**: `/release` or `/release-alpha` to create PR
- **Current**: `/release-tag` to create and push tag
- **Validation**: `/release-validate` to check readiness before starting release

This command completes the release process by creating the tag that triggers publication.
