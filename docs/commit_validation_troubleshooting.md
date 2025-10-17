# Conventional Commit Troubleshooting Guide

This guide helps resolve common issues with conventional commit validation in the DQX project.

## Common Issues and Solutions

### 1. Commit Message Rejected

**Problem**: Your commit is blocked with an error like:
```
commitizen.....................................................................Failed
- hook id: commitizen
- exit code: 1

commit validation: failed!
commit message does not match the pattern: <type>(<scope>): <subject>
```

**Solution**: Your commit message doesn't follow the conventional format. Use:
```bash
# Correct format
git commit -m "feat(analyzer): add query optimization"

# Or use interactive mode
uv run cz commit
```

### 2. Unknown Commit Type

**Problem**: Error about invalid commit type.

**Valid types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code formatting
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Test changes
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes
- `revert`: Reverting commits

### 3. Invalid Scope

**Problem**: Scope not recognized.

**Valid scopes** (optional):
- `analyzer`: SQL analysis engine
- `api`: User-facing API
- `graph`: Graph implementation
- `evaluator`: Assertion evaluation
- `provider`: Metric provider
- `specs`: Metric specifications
- `validator`: Suite validation
- `display`: Result visualization
- `orm`: Database persistence
- `extensions`: Data source extensions
- `common`: Shared types
- `dialect`: SQL dialects
- `functions`: Math functions
- `models`: Data models
- `ops`: Metric operations
- `states`: State management
- `utils`: Utilities

### 4. Commit Message Too Long

**Problem**: Subject line exceeds 72 characters.

**Solution**: Keep the subject concise:
```bash
# Too long ❌
git commit -m "feat(analyzer): implement comprehensive query optimization for handling large datasets with multiple joins and complex aggregations"

# Better ✅
git commit -m "feat(analyzer): optimize queries for large datasets

Add query optimization for:
- Multiple joins
- Complex aggregations
- Large result sets"
```

### 5. Breaking Changes

**Problem**: Not sure how to mark breaking changes.

**Solution**: Two options:
```bash
# Option 1: Using ! marker
git commit -m "feat(api)!: require where() clause for assertions"

# Option 2: Using BREAKING CHANGE footer
git commit -m "feat(api): update assertion interface

BREAKING CHANGE: assert_that() now requires where() clause
Migration: Add .where(name='description') to all assertions"
```

### 6. Multi-line Commits

**Problem**: Need to add body or footer to commit.

**Solution**: Use quotes or editor:
```bash
# Using quotes
git commit -m "fix(evaluator): handle division by zero

This fixes the issue where percentage calculations fail
when the denominator is zero.

Closes #123"

# Using editor (opens your default editor)
git commit
```

### 7. Amending Commits

**Problem**: Made a typo in commit message.

**Solution**:
```bash
# Fix the last commit message
git commit --amend -m "feat(api): correct commit message"

# Or amend interactively
git commit --amend
```

### 8. Checking Commit History

**Problem**: Want to verify past commits follow convention.

**Solution**:
```bash
# Check last commit
./bin/run-hooks.sh --check-commit

# Check range of commits
uv run cz check --rev-range origin/main..HEAD

# View commit history with formatting
git log --oneline --pretty=format:"%h %s"
```

### 9. Emergency Commits

**Problem**: Need to commit urgently without validation.

**Solution** (use sparingly):
```bash
# Skip all hooks
git commit --no-verify -m "emergency: fix production issue"

# Better: follow convention even in emergencies
git commit -m "fix(api): emergency patch for production crash"
```

### 10. Commitizen Not Found

**Problem**: `uv run cz commit` fails with "command not found".

**Solution**:
```bash
# Sync dependencies
uv sync

# Verify installation
uv run cz version

# Reinstall if needed
uv pip install commitizen
```

## Best Practices

### DO ✅
- Use present tense: "add feature" not "added feature"
- Be specific: "fix null pointer in analyzer" not "fix bug"
- Reference issues: "Closes #123" in footer
- Keep subject under 72 characters
- Add body for complex changes

### DON'T ❌
- Mix changes: One commit per logical change
- Use generic messages: "fix", "update", "changes"
- Include file names in subject (Git shows this)
- End subject with period
- Commit commented code

## Quick Reference

```bash
# Feature
git commit -m "feat(provider): add median metric support"

# Bug fix
git commit -m "fix(evaluator): handle empty datasets gracefully"

# Docs
git commit -m "docs: add troubleshooting guide for commits"

# Breaking change
git commit -m "refactor(api)!: rename is_equal to is_eq"

# With body and footer
git commit -m "fix(analyzer): prevent SQL injection in column names

Sanitize column names before query generation to prevent
potential SQL injection vulnerabilities.

Security: High priority fix
Closes #456"
```

## Getting Help

- Run `uv run cz commit` for interactive guidance
- Check `./bin/run-hooks.sh --check-commit` to validate
- Read the [Conventional Commits spec](https://www.conventionalcommits.org/)
- Ask the team if unsure about type or scope
