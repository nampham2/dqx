# GitHub CI/CD Operations Guide

This guide covers daily operations and interactions with the CI/CD system after initial setup.

## Daily Development Workflow

### Working with Pre-commit Hooks

Pre-commit hooks run automatically before each commit to catch issues early.

#### Running Hooks Locally

```bash
# Run all hooks on staged files
uv run hooks

# Run all hooks on all files
uv run hooks --all

# Run specific hook
uv run hooks mypy

# See available options
uv run hooks --help
```

#### Common Hook Failures and Fixes

**Ruff formatting issues:**
```bash
# Auto-fix formatting
uv run ruff check --fix .
uv run ruff format .

# Then stage the fixes
git add -u
```

**Mypy type errors:**
- Add missing type hints
- Fix type inconsistencies
- Use `# type: ignore[error-code]` sparingly with justification

**Trailing whitespace:**
- Most editors can auto-remove on save
- The hook will fix automatically

#### Bypassing Hooks (Use Sparingly)

```bash
# Skip all hooks - use only when necessary
git commit --no-verify -m "emergency: fix critical bug"

# Skip specific hooks
SKIP=mypy git commit -m "wip: debugging in progress"
```

### Commit Standards

Follow conventional commits for clear history and automatic versioning.

#### Format
```
type(scope): subject

body (optional)

footer (optional)
```

#### Common Types
- `feat`: New feature (triggers minor version)
- `fix`: Bug fix (triggers patch version)
- `docs`: Documentation only
- `style`: Code style (no logic changes)
- `refactor`: Code restructure
- `perf`: Performance improvement
- `test`: Test additions/fixes
- `chore`: Maintenance tasks

#### Examples
```bash
git commit -m "feat(analyzer): add batch processing support"
git commit -m "fix(validator): handle null values correctly"
git commit -m "docs: update API examples"
```

### Creating Pull Requests

1. **Create feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Push and create PR:**
   ```bash
   git push -u origin feat/your-feature-name
   gh pr create --title "feat: your feature" --body "Description"
   ```

3. **Required elements:**
   - Descriptive title following commit conventions
   - Clear description of changes
   - Link to related issue (if any)
   - Screenshots for UI changes

## Code Review Process

### Understanding CodeRabbit AI Reviews

CodeRabbit automatically reviews PRs within 5 minutes.

#### Review Components

1. **Summary Section**
   - High-level overview
   - Key changes identified
   - Potential issues flagged

2. **Detailed Comments**
   - Line-by-line suggestions
   - Best practice violations
   - Security concerns
   - Performance issues

3. **Review Status**
   - ✅ Approved: No critical issues
   - 💬 Comments: Suggestions provided
   - 🔄 Changes requested: Critical issues found

#### Interacting with CodeRabbit

**Common commands in PR comments:**
```
@coderabbitai review
@coderabbitai resolve
@coderabbitai ignore
@coderabbitai help
```

**Responding to suggestions:**
1. **Accept suggestion:** Click "Accept" on the comment
2. **Dismiss:** Reply explaining why it's not applicable
3. **Ask for clarification:** Reply with your question

#### Customizing Reviews

Edit `.coderabbit.yaml` to adjust:
- Review strictness
- Custom patterns
- Excluded paths
- Project-specific rules

### Handling Review Feedback

1. **Address all comments** - Even if just to explain why not changed
2. **Batch fixes** - Make all changes before requesting re-review
3. **Update PR description** - Note what feedback was addressed
4. **Request re-review** - Use GitHub's re-review feature

## Managing Dependencies

### Dependabot Pull Requests

Dependabot creates PRs for dependency updates automatically.

#### Understanding Update PRs

**PR Title Format:**
```
Bump package-name from 1.2.3 to 1.2.4
```

**PR includes:**
- Changelog excerpts
- Commit list
- Compatibility score
- Release notes link

#### Review Process

1. **Check CI status** - All tests must pass
2. **Review changelog** - Look for breaking changes
3. **Check compatibility** - Verify with your Python versions
4. **Test locally if needed:**
   ```bash
   gh pr checkout 123
   uv sync
   uv run pytest tests/
   ```

#### Dependabot Commands

Use these in PR comments:

```
@dependabot rebase
@dependabot recreate
@dependabot merge
@dependabot squash and merge
@dependabot cancel merge
@dependabot reopen
@dependabot close
@dependabot ignore this major version
@dependabot ignore this minor version
@dependabot ignore this dependency
```

#### Merge Strategies

**Security updates:** Merge immediately after tests pass

**Minor updates:**
- Group weekly
- Test together
- Merge as batch

**Major updates:**
- Test thoroughly
- Check migration guides
- Consider compatibility

### Managing Multiple Updates

```bash
# List all Dependabot PRs
gh pr list --label dependencies

# Bulk operations with GitHub CLI
gh pr list --label dependencies --json number --jq '.[].number' | \
  xargs -I {} gh pr comment {} --body "@dependabot rebase"
```

## Release Process

### Preparing a Release

1. **Update version in `pyproject.toml`:**
   ```toml
   version = "0.4.0"
   ```

2. **Update CHANGELOG.md:**
   - Add release date
   - Review all changes
   - Highlight breaking changes

3. **Create release PR:**
   ```bash
   git checkout -b release/v0.4.0
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: prepare release v0.4.0"
   git push -u origin release/v0.4.0
   gh pr create --title "Release v0.4.0" --label release
   ```

### Using Release Drafter

Release Drafter automatically maintains draft release notes.

#### How It Works

1. **Monitors merged PRs** - Collects since last release
2. **Categorizes by labels** - Groups changes by type
3. **Determines version** - Based on change types
4. **Updates draft** - After each PR merge

#### PR Labels for Versioning

**Major version (breaking changes):**
- `breaking-change`
- `breaking`

**Minor version (features):**
- `feature`
- `enhancement`
- `feat`

**Patch version (fixes):**
- `fix`
- `bug`
- `perf`
- `docs`

#### Editing Release Notes

1. Go to Releases → Draft release
2. Review auto-generated notes
3. Add highlights section
4. Include migration guide for breaking changes
5. Preview before publishing

### Publishing a Release

1. **Final checks:**
   - All CI passes on main
   - Version updated in code
   - Documentation updated

2. **Create and push tag:**
   ```bash
   git checkout main
   git pull origin main
   git tag v0.4.0
   git push origin v0.4.0
   ```

3. **Publish GitHub release:**
   - Go to draft release
   - Select the tag
   - Review notes one more time
   - Click "Publish release"

4. **Monitor deployment:**
   - Check Actions tab for release workflow
   - Verify PyPI upload
   - Test installation: `pip install dqx==0.4.0`

### Post-Release Tasks

1. **Verify deployment:**
   ```bash
   # Test from PyPI
   pip install dqx==0.4.0
   python -c "import dqx; print(dqx.__version__)"
   ```

2. **Update documentation:**
   - Check ReadTheDocs built new version
   - Update version references
   - Announce in relevant channels

3. **Create next development cycle:**
   ```bash
   # Bump to next dev version
   # Update pyproject.toml to 0.5.0-dev
   git commit -m "chore: bump version to 0.5.0-dev"
   ```

## Monitoring CI/CD

### GitHub Actions Dashboard

Access at: `https://github.com/<owner>/dqx/actions`

#### Key Metrics
- Success rate by workflow
- Average run time
- Recent failures
- Usage minutes

#### Workflow Management

**Re-run failed jobs:**
1. Click on failed workflow run
2. Click "Re-run failed jobs"
3. Or "Re-run all jobs" if needed

**Cancel stuck workflows:**
```bash
gh run list --status in_progress
gh run cancel <run-id>
```

**Download artifacts:**
```bash
gh run download <run-id> -n <artifact-name>
```

### Debugging CI Failures

1. **Check summary** - Look for error annotations
2. **Expand failed step** - Find exact error
3. **Download logs** - For detailed analysis
4. **Run locally** - Try to reproduce

#### Common Issues and Solutions

**Test failures on CI only:**
- Check for environment differences
- Look for timing issues
- Verify test data availability

**Timeout errors:**
- Increase timeout in workflow
- Split long-running tests
- Add progress output

**Permission errors:**
- Check token permissions
- Verify secrets are set
- Review repository settings

## Quick Reference

### Essential Commands

```bash
# Pre-commit hooks
uv run hooks --all           # Run all hooks
uv run hooks --help          # Show options

# GitHub CLI
gh pr create                 # Create PR
gh pr list                   # List PRs
gh pr checks                 # Show CI status
gh workflow run              # Trigger workflow

# Git
git tag v0.4.0              # Create version tag
git push origin v0.4.0      # Push tag

# Python/uv
uv sync                     # Install dependencies
uv run pytest               # Run tests
uv build                    # Build package
```

### Status Badge URLs

```markdown
![Tests](https://github.com/<owner>/dqx/workflows/tests/badge.svg)
![Coverage](https://github.com/<owner>/dqx/coverage.svg)
![Docs](https://readthedocs.org/projects/dqx/badge/)
![PyPI](https://badge.fury.io/py/dqx.svg)
```

### Useful Links

- [GitHub Actions Docs](https://docs.github.com/actions)
- [Conventional Commits](https://www.conventionalcommits.org)
- [CodeRabbit Docs](https://coderabbit.ai/docs)
- [Dependabot Docs](https://docs.github.com/code-security/dependabot)

---

For initial setup instructions, see the [GitHub CI/CD Setup Guide](./github-cicd-setup-guide.md).
