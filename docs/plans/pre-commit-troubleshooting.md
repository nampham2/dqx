# Pre-commit Troubleshooting Guide

This guide helps you resolve common issues with pre-commit hooks in the DQX project.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Hook-Specific Problems](#hook-specific-problems)
3. [Performance Issues](#performance-issues)
4. [Environment Issues](#environment-issues)
5. [CI/CD Issues](#cicd-issues)
6. [Emergency Procedures](#emergency-procedures)

## Common Issues

### Issue: "An error has occurred: InvalidConfigError"

**Symptom:**
```
An error has occurred: InvalidConfigError:
=====> .pre-commit-config.yaml is not a file
```

**Solution:**
```bash
# Ensure you're in the project root
pwd  # Should show /path/to/dqx

# Check if the config file exists
ls -la .pre-commit-config.yaml

# If missing, restore from git
git checkout .pre-commit-config.yaml
```

### Issue: "pre-commit is not installed"

**Symptom:**
```
bash: pre-commit: command not found
```

**Solution:**
```bash
# Install pre-commit with uv
uv sync

# Verify installation
uv run pre-commit --version

# Install hooks
uv run pre-commit install
```

### Issue: "Failed to build wheel for package"

**Symptom:**
Pre-commit fails while installing hook environments.

**Solution:**
```bash
# Clear pre-commit cache
uv run pre-commit clean

# Update pre-commit environments
uv run pre-commit install --install-hooks

# If still failing, clear all caches
rm -rf ~/.cache/pre-commit
uv run pre-commit install --install-hooks
```

### Issue: Hooks not running on commit

**Symptom:**
Git commits succeed without running hooks.

**Solution:**
```bash
# Check if hooks are installed
cat .git/hooks/pre-commit

# Reinstall hooks
uv run pre-commit install

# Verify hooks are executable
chmod +x .git/hooks/pre-commit

# Test manually
uv run pre-commit run
```

## Hook-Specific Problems

### Ruff Format Issues

**Problem:** Code keeps getting reformatted

**Solution:**
```bash
# Check your VS Code settings
cat .vscode/settings.json

# Ensure Ruff extension is installed
code --install-extension charliermarsh.ruff

# Format all files once
uv run ruff format src/ tests/

# Stage and commit the formatted files
git add -u
git commit -m "style: apply consistent formatting"
```

### Ruff Check Issues

**Problem:** Import order keeps changing

**Solution:**
```bash
# Let Ruff fix imports automatically
uv run ruff check --fix src/ tests/

# For specific import issues:
# 1. Group imports: standard library, third-party, local
# 2. Sort alphabetically within groups
# 3. Use absolute imports for project modules
```

### MyPy Type Checking Issues

**Problem:** MyPy is too slow or has false positives

**Temporary Solution:**
```bash
# Skip mypy for a single commit
SKIP=mypy git commit -m "wip: quick fix"

# Or use the fast mode
./bin/run-hooks.sh --fast
```

**Permanent Solution:**
```bash
# Fix type issues properly
uv run mypy src/ --show-error-codes

# Common fixes:
# - Add type annotations
# - Use Optional[] for nullable values
# - Import types from typing module
```

### Debug Statement Detection

**Problem:** Commit blocked by debug statements

**Example:**
```
Check for debug statements...............................................Failed
- hook id: debug-statements
- exit code: 1

test.py:4:4: breakpoint called
```

**Solution:**
```python
# Remove debug statements:
# - print() statements (in production code)
# - breakpoint() calls
# - import pdb; pdb.set_trace()
# - console.log (in JS files)

# For legitimate debug output, use logging:
import logging

logger = logging.getLogger(__name__)
logger.debug("Debug information here")
```

## Performance Issues

### Slow Hook Execution

**Problem:** Hooks take too long to run

**Solutions:**

1. **Use parallel execution:**
   ```bash
   # Install pre-commit with parallel support
   uv add pre-commit[parallel]
   ```

2. **Skip slow hooks during development:**
   ```bash
   # Skip mypy temporarily
   ./bin/run-hooks.sh --fast

   # Run only formatting hooks
   ./bin/run-hooks.sh --fix
   ```

3. **Run hooks on specific files:**
   ```bash
   # Instead of running on all files
   ./bin/run-hooks.sh src/dqx/api.py
   ```

### Large File Issues

**Problem:** "File too large" errors

**Solution:**
```bash
# Check file sizes
find . -type f -size +1M -ls

# Add large files to .gitignore
echo "large_data.csv" >> .gitignore

# If you must commit large files, configure the hook:
# Edit .pre-commit-config.yaml and adjust:
# args: ['--maxkb=2000']  # Allow up to 2MB
```

## Environment Issues

### Python Version Mismatch

**Problem:** Hooks fail with Python syntax errors

**Solution:**
```bash
# Check Python version
python --version

# Ensure Python 3.11+ is used
# If using pyenv:
pyenv local 3.11.10

# Update .python-version file
echo "3.11.10" > .python-version
```

### Virtual Environment Issues

**Problem:** Import errors or missing packages

**Solution:**
```bash
# Ensure uv is managing the environment
uv sync

# Activate the environment manually if needed
source .venv/bin/activate

# Verify packages are installed
uv pip list | grep pre-commit
```

## CI/CD Issues

### GitHub Actions Failures

**Problem:** Pre-commit passes locally but fails in CI

**Common Causes & Solutions:**

1. **Different Python versions:**
   ```yaml
   # Ensure .github/workflows/pre-commit.yml uses same Python version
   - uses: actions/setup-python@v5
     with:
       python-version: '3.11'  # Match local version
   ```

2. **Missing dependencies:**
   ```bash
   # Ensure uv.lock is committed
   git add uv.lock
   git commit -m "chore: update lock file"
   ```

3. **Line ending differences:**
   ```bash
   # Configure git to handle line endings
   git config core.autocrlf input

   # Re-checkout files
   git rm --cached -r .
   git reset --hard
   ```

### Cache Issues in CI

**Problem:** Old cached dependencies cause failures

**Solution:**
Update cache keys in `.github/workflows/pre-commit.yml`:
```yaml
key: ${{ runner.os }}-uv-${{ hashFiles('pyproject.toml', 'uv.lock') }}-v2
#                                                                      ^^^^
# Increment version to bust cache
```

## Emergency Procedures

### Bypass Hooks (Use Sparingly!)

**When to use:**
- Urgent hotfixes
- Reverting broken changes
- Committing work-in-progress before switching branches

**How to bypass:**
```bash
# Skip all hooks
git commit --no-verify -m "hotfix: emergency fix"

# Skip specific hooks
SKIP=mypy,ruff-check git commit -m "wip: partial implementation"
```

**Important:** Always run hooks manually afterward:
```bash
./bin/run-hooks.sh --all
```

### Reset Pre-commit

**Complete reset procedure:**
```bash
# 1. Uninstall hooks
uv run pre-commit uninstall

# 2. Clear all caches
rm -rf ~/.cache/pre-commit

# 3. Remove and reinstall pre-commit
uv remove pre-commit
uv add --dev pre-commit

# 4. Reinstall hooks
uv run pre-commit install

# 5. Test
uv run pre-commit run --all-files
```

### Debugging Hook Failures

**Enable verbose output:**
```bash
# See what pre-commit is doing
uv run pre-commit run --verbose

# Check specific hook
uv run pre-commit run mypy --verbose

# See hook configuration
uv run pre-commit run --show-diff-on-failure
```

**Check hook logs:**
```bash
# Pre-commit stores logs here
cat ~/.cache/pre-commit/pre-commit.log

# Check specific hook environment
ls ~/.cache/pre-commit/repo*/
```

## Best Practices

1. **Regular Maintenance:**
   ```bash
   # Update hooks periodically
   uv run pre-commit autoupdate

   # Clean old environments
   uv run pre-commit gc
   ```

2. **Team Coordination:**
   - Commit .pre-commit-config.yaml changes separately
   - Announce hook updates to the team
   - Document custom hooks in README

3. **Local Development:**
   - Run hooks before pushing: `./bin/run-hooks.sh --all`
   - Use VS Code integration for real-time feedback
   - Keep hooks fast by using --fast mode during development

## Getting Help

If you encounter issues not covered here:

1. Check pre-commit documentation: https://pre-commit.com/
2. Search existing GitLab issues
3. Create a new issue with:
   - Full error message
   - Output of `uv run pre-commit --version`
   - Output of `python --version`
   - Steps to reproduce

Remember: Pre-commit hooks are here to help maintain code quality. If they're consistently blocking your work, there might be a configuration issue that needs addressing.
