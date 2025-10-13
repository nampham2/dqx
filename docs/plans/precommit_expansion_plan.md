# Pre-commit Expansion Implementation Plan

## Overview
Extend pre-commit hooks to validate example code, shell scripts, documentation, and configuration files across the codebase.

This plan adds comprehensive quality checks for:
- Python examples in `/examples` directory
- Bash scripts in `/bin` and `.github` directories
- Python code blocks in markdown documentation
- YAML configuration files

The goal is to ensure ALL code and configuration in the repository meets our quality standards, not just the main source code.

## Background

### What is Pre-commit?
Pre-commit is a framework that runs checks on your code before each git commit. It helps catch issues early and maintains code quality automatically. Think of it as an automated code reviewer that runs locally.

### Current State
- **Pre-commit hooks**: Currently check Python files in `src/` and `tests/` with:
  - `ruff format` - Python code formatter (like prettier for Python)
  - `ruff check` - Python linter (finds code issues)
  - `mypy` - Python type checker (validates type annotations)
- **Coverage**: Pytest coverage only tracks `src/` code (excludes examples - this is correct)
- **Shell scripts**: Not validated at all
- **Example code**: Not type-checked by mypy

### Problem
1. Example code in `/examples` could have type errors
2. Shell scripts could have bugs or portability issues
3. CI/CD scripts in `.github/` aren't validated
4. Python code in documentation could be incorrectly formatted
5. YAML files could have syntax errors or inconsistent style

## Goal
Ensure ALL code and configuration in the repository is validated:
- Python examples get type-checked and linted
- Shell scripts get validated and formatted consistently
- Python code in markdown files follows project style
- YAML files are validated for syntax and style
- No changes to coverage (examples should stay excluded)

## Prerequisites

Before starting, ensure you have:
```bash
# The project uses uv for dependency management
which uv  # Should show uv is installed

# Pre-commit should be installed
uv run pre-commit --version

# Current hooks should be working
./bin/run-hooks.sh --all  # Should run without errors
```

## Implementation Tasks

### Task 1: Add Shellcheck to Pre-commit Configuration

**Purpose**: Validate shell scripts for common errors and portability issues.

**What is Shellcheck?**: A linter for bash/shell scripts that finds bugs, deprecated syntax, and portability issues.

#### Step 1.1: Write a failing test
Create a test script to verify shellcheck will catch errors:

```bash
# Create test file: test_shellcheck.sh
cat > test_shellcheck.sh << 'EOF'
#!/bin/bash
# This script has intentional errors for testing

# Error 1: Unquoted variable (SC2086)
file=$1
rm $file  # Should be: rm "$file"

# Error 2: Useless echo (SC2005)
echo $(date)  # Should be: date

# Error 3: Missing quotes in test (SC2086)
if [ $file == "test" ]; then  # Should be: if [ "$file" == "test" ]
    echo "test file"
fi
EOF

chmod +x test_shellcheck.sh
```

#### Step 1.2: Run pre-commit to confirm it doesn't check shell scripts yet
```bash
# This should pass (incorrectly) because shellcheck isn't configured
uv run pre-commit run --files test_shellcheck.sh
```

Expected: Pre-commit runs but doesn't report the shell script errors.

#### Step 1.3: Add shellcheck to pre-commit config

**File to modify**: `.pre-commit-config.yaml`

Add this new repo block after the existing `pre-commit-hooks` repo:

```yaml
  # Shell script validation
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.10.0.1
    hooks:
      - id: shellcheck
        name: shellcheck - lint shell scripts
        args: ['--severity=warning']  # Show warnings and errors
        files: '\.(sh|bash)$'
        types: [shell]
        require_serial: false
```

**Note**: Place this BEFORE the `repo: local` section to maintain the pattern of external repos first, then local hooks.

#### Step 1.4: Test that shellcheck now catches errors
```bash
# Install the new hook
uv run pre-commit install

# Run on our test file - should FAIL
uv run pre-commit run --files test_shellcheck.sh
```

Expected output should show shellcheck errors like:
- "Double quote to prevent globbing"
- "Useless echo"
- "Quote this to prevent word splitting"

#### Step 1.5: Fix the test script and verify
```bash
# Fix the issues
cat > test_shellcheck.sh << 'EOF'
#!/bin/bash
# This script is now correct

# Fixed: Quoted variable
file=$1
rm "$file"

# Fixed: Direct command instead of echo
date

# Fixed: Quoted test
if [ "$file" == "test" ]; then
    echo "test file"
fi
EOF

# Should now pass
uv run pre-commit run --files test_shellcheck.sh
```

#### Step 1.6: Clean up and commit
```bash
# Remove test file
rm test_shellcheck.sh

# Test on actual project scripts
uv run pre-commit run shellcheck --all-files

# If any issues are found in existing scripts, fix them first
# Then commit
git add .pre-commit-config.yaml
git commit -m "feat: add shellcheck to pre-commit for shell script validation

- Validates .sh and .bash files for common errors
- Helps prevent shell scripting bugs and portability issues
- Configured to show warnings and errors"
```

### Task 2: Update MyPy Configuration to Include Examples

**Purpose**: Ensure Python example code is type-checked.

#### Step 2.1: Write a failing test
Create an example with type errors:

```bash
# Create test file: examples/test_types.py
cat > examples/test_types.py << 'EOF'
"""Test file with intentional type errors."""

def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Type error: passing string to function expecting int
result = add_numbers("5", 10)

# Type error: wrong return type annotation
def get_name() -> int:
    """This claims to return int but returns str."""
    return "John"
EOF
```

#### Step 2.2: Verify mypy doesn't check examples yet
```bash
# Current mypy configuration only checks src and tests
uv run mypy examples/test_types.py
# Should output: "Success: no issues found" (incorrectly)

# Also verify via pre-commit
uv run pre-commit run mypy --all-files
# Should pass without checking examples
```

#### Step 2.3: Update mypy hook in pre-commit

**File to modify**: `.pre-commit-config.yaml`

Find the mypy hook in the `local` hooks section and update the args:

```yaml
      # MyPy - Static type checker (doesn't modify files)
      - id: mypy
        name: mypy type check
        entry: uv run mypy
        language: system
        types: [python]
        require_serial: true
        pass_filenames: false
        args: [src, tests, examples]  # Add 'examples' here
```

#### Step 2.4: Test that mypy now checks examples
```bash
# Should now FAIL with type errors
uv run pre-commit run mypy --all-files
```

Expected errors:
- "Argument 1 to "add_numbers" has incompatible type "str"; expected "int""
- "Incompatible return value type (got "str", expected "int")"

#### Step 2.5: Fix type errors and verify
```bash
# Fix the example
cat > examples/test_types.py << 'EOF'
"""Test file with correct types."""

def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Correct: passing int to function expecting int
result = add_numbers(5, 10)

# Correct: return type matches annotation
def get_name() -> str:
    """This correctly returns str."""
    return "John"
EOF

# Should now pass
uv run pre-commit run mypy --all-files
```

#### Step 2.6: Clean up and commit
```bash
# Remove test file
rm examples/test_types.py

# Run full check on actual examples
uv run pre-commit run mypy --all-files

# Fix any real issues found in existing examples
# Then commit
git add .pre-commit-config.yaml
git commit -m "feat: add examples directory to mypy type checking

- Examples are now type-checked to ensure correctness
- Helps maintain quality of demo/example code
- Prevents examples from having type errors"
```

### Task 3: Update CI/CD Quality Script

**Purpose**: Ensure CI/CD runs the same checks as local development.

#### Step 3.1: Review current quality script
```bash
cat .github/quality.sh
# Currently shows: uv run mypy src tests
```

#### Step 3.2: Update quality script

**File to modify**: `.github/quality.sh`

Update the mypy command to include examples:

```bash
#!/bin/bash

uv run ruff check
uv run mypy src tests examples
```

#### Step 3.3: Test the updated script
```bash
# Make it executable if needed
chmod +x .github/quality.sh

# Run it
./.github/quality.sh

# Should run successfully and check all three directories
```

#### Step 3.4: Commit the change
```bash
git add .github/quality.sh
git commit -m "feat: add examples to CI/CD quality checks

- CI now type-checks examples directory
- Ensures GitHub Actions catches type errors in examples
- Maintains consistency with local pre-commit checks"
```

### Task 4: Update run-hooks.sh Documentation

**Purpose**: Document that the script now checks shell scripts.

#### Step 4.1: Update script header

**File to modify**: `bin/run-hooks.sh`

Update the header comment to mention shell script checking:

```bash
#!/bin/bash
# Run pre-commit hooks manually with options
# Usage: ./bin/run-hooks.sh [options] [files...]
#
# Checks:
#   - Python code formatting and linting (ruff)
#   - Python type checking (mypy)
#   - Shell script validation (shellcheck)
#   - General file quality (trailing whitespace, file size, etc.)
#
# Options:
#   --all     Run on all files
#   --fast    Skip slow hooks (mypy)
#   --fix     Only run hooks that auto-fix issues
```

#### Step 4.2: Test and commit
```bash
# Verify script still works
./bin/run-hooks.sh --all

git add bin/run-hooks.sh
git commit -m "docs: update run-hooks.sh documentation for shellcheck"
```

### Task 5: Add Shell Script Formatting with shfmt

**Purpose**: Ensure shell scripts follow consistent formatting standards.

**What is shfmt?**: A shell script formatter that enforces consistent style (like ruff for Python but for shell scripts).

#### Step 5.1: Write a test script with formatting issues
```bash
# Create poorly formatted script
cat > test_format.sh << 'EOF'
#!/bin/bash
# Inconsistent formatting

function  badly_formatted(){
echo "no indentation"
if [[ $1 == "test" ]];then
echo "bad spacing"
    fi
  }

   # Random indentation
      VAR="value"
EOF

chmod +x test_format.sh
```

#### Step 5.2: Add shfmt to pre-commit config

**File to modify**: `.pre-commit-config.yaml`

Add this after the shellcheck configuration:

```yaml
  # Shell script formatting
  - repo: https://github.com/scop/pre-commit-shfmt
    rev: v3.8.0-1
    hooks:
      - id: shfmt
        name: shfmt - format shell scripts
        args: ['-i', '2', '-w']  # 2-space indent, write changes
        files: '\.(sh|bash)$'
        types: [shell]
```

#### Step 5.3: Test shfmt formatting
```bash
# Install new hooks
uv run pre-commit install --install-hooks

# Run shfmt - should reformat the file
uv run pre-commit run shfmt --files test_format.sh

# Check the formatted result
cat test_format.sh
```

Expected: Script should now have consistent 2-space indentation and proper formatting.

#### Step 5.4: Clean up and commit
```bash
# Remove test file
rm test_format.sh

# Run on actual scripts
uv run pre-commit run shfmt --all-files

# Commit formatting changes if any
git add -A
git commit -m "feat: add shfmt for shell script formatting

- Enforces consistent 2-space indentation
- Standardizes shell script style across the project
- Auto-formats scripts on commit"
```

### Task 6: Add Python Code Formatting in Documentation

**Purpose**: Ensure Python code blocks in markdown files follow project formatting standards.

**What is blacken-docs?**: Formats Python code blocks inside markdown/rst files to match project style.

#### Step 6.1: Create test markdown with poorly formatted Python
```bash
# Create test documentation
cat > test_docs.md << 'EOF'
# Test Documentation

Here's some Python code:

```python
# Poorly formatted code
def   bad_function(x,y,z):
     return x+y+z

class  BadClass:
    def __init__(self,name):
        self.name=name
```

Another example:

```python
# More bad formatting
data={'key1':'value1','key2':'value2','key3':'value3'}
result=bad_function(1,2,3)
```
EOF
```

#### Step 6.2: Add blacken-docs to pre-commit config

**File to modify**: `.pre-commit-config.yaml`

Add this after the shfmt configuration:

```yaml
  # Python code formatting in documentation
  - repo: https://github.com/asottile/blacken-docs
    rev: 1.16.0
    hooks:
      - id: blacken-docs
        name: blacken-docs - format Python in markdown
        additional_dependencies: [black==24.1.0]
        args: ['--line-length=120']  # Match project line length
```

#### Step 6.3: Test blacken-docs
```bash
# Install hooks
uv run pre-commit install --install-hooks

# Run blacken-docs - should format Python code blocks
uv run pre-commit run blacken-docs --files test_docs.md

# Check the result
cat test_docs.md
```

Expected: Python code blocks should now be properly formatted with correct spacing.

#### Step 6.4: Clean up and commit
```bash
# Remove test file
rm test_docs.md

# Run on all markdown files
uv run pre-commit run blacken-docs --all-files

# Commit any formatting changes
git add -A
git commit -m "feat: add blacken-docs for Python code in documentation

- Formats Python code blocks in markdown files
- Ensures documentation examples follow project style
- Uses same line length as main code (120 chars)"
```

### Task 7: Add YAML Validation

**Purpose**: Validate YAML files for syntax errors and style consistency.

**What is yamllint?**: A linter for YAML files that checks syntax, key duplication, line length, etc.

#### Step 7.1: Create test YAML with issues
```bash
# Create problematic YAML
cat > test_config.yaml << 'EOF'
# Bad YAML formatting
key1:  value1
key2:    value2  # inconsistent spacing
key3: value3
  key4:  nested  # wrong indentation

duplicate_key: first
duplicate_key: second  # duplicate key

very_long_line: "This is a very long line that exceeds reasonable limits and should probably be split into multiple lines for better readability"

trailing_spaces: value
no_final_newline: true
EOF

# Note: Don't add final newline to test that check
```

#### Step 7.2: Add yamllint to pre-commit config

**File to modify**: `.pre-commit-config.yaml`

Add this after blacken-docs:

```yaml
  # YAML validation
  - repo: https://github.com/adrienverge/yamllint
    rev: v1.35.1
    hooks:
      - id: yamllint
        name: yamllint - validate YAML files
        args: ['--config-data', '{extends: default, rules: {line-length: {max: 120}, truthy: disable}}']
        types: [yaml]
```

#### Step 7.3: Test yamllint
```bash
# Install hooks
uv run pre-commit install --install-hooks

# Run yamllint - should show errors
uv run pre-commit run yamllint --files test_config.yaml
```

Expected errors:
- Wrong indentation
- Duplicate keys
- Line too long
- Trailing spaces
- No newline at end of file

#### Step 7.4: Fix and verify
```bash
# Fix the YAML
cat > test_config.yaml << 'EOF'
# Good YAML formatting
key1: value1
key2: value2  # consistent spacing
key3: value3
key4: nested  # correct indentation

unique_key1: first
unique_key2: second  # unique keys

very_long_line: >
  This is a very long line that is now split
  into multiple lines for better readability

trailing_spaces: value
final_newline: true
EOF

# Should now pass
uv run pre-commit run yamllint --files test_config.yaml
```

#### Step 7.5: Clean up and commit
```bash
# Remove test file
rm test_config.yaml

# Run on all YAML files
uv run pre-commit run yamllint --all-files

# Fix any issues found, then commit
git add -A
git commit -m "feat: add yamllint for YAML validation

- Validates YAML syntax and style
- Prevents duplicate keys and syntax errors
- Enforces consistent formatting
- Configured for 120 char line length to match project"
```

### Task 8: Update Documentation and Final Integration

**Purpose**: Update documentation and perform final integration testing with all new tools.

#### Step 8.1: Update run-hooks.sh documentation

**File to modify**: `bin/run-hooks.sh`

Update the header to include all new checks:

```bash
#!/bin/bash
# Run pre-commit hooks manually with options
# Usage: ./bin/run-hooks.sh [options] [files...]
#
# Checks:
#   - Python code formatting and linting (ruff)
#   - Python type checking (mypy)
#   - Shell script validation (shellcheck)
#   - Shell script formatting (shfmt)
#   - Python code in markdown formatting (blacken-docs)
#   - YAML file validation (yamllint)
#   - General file quality (trailing whitespace, file size, etc.)
#
# Options:
#   --all     Run on all files
#   --fast    Skip slow hooks (mypy)
#   --fix     Only run hooks that auto-fix issues
```

#### Step 8.2: Update the --fix option to include new formatters

**File to modify**: `bin/run-hooks.sh`

Update the --fix case to include the new formatting hooks:

```bash
        --fix)
            HOOK_ID="ruff-format,ruff-check,trailing-whitespace,end-of-file-fixer,shfmt,blacken-docs"
            shift
            ;;
```

#### Step 8.3: Full test of all changes
```bash
# Run all hooks on all files
./bin/run-hooks.sh --all

# Test specific file types
uv run pre-commit run --files examples/*.py  # Python examples
uv run pre-commit run --files bin/*.sh       # Shell scripts
uv run pre-commit run --files .github/*.sh   # CI scripts
```

#### Step 8.4: Test that coverage still excludes examples
```bash
# Run coverage report
./.github/coverage.sh

# Verify output doesn't include examples/ files
# The coverage should only report on src/dqx files
```

#### Step 8.5: Create demos to show all tools work

**Test shell script formatting:**
```bash
# Create unformatted shell script
cat > demo_bad.sh << 'EOF'
#!/bin/bash
function test() {
echo "bad indent"
}
EOF

# Show shfmt fixes it
uv run pre-commit run shfmt --files demo_bad.sh
cat demo_bad.sh  # Should show proper formatting
rm demo_bad.sh
```

**Test Python in markdown:**
```bash
# Create markdown with bad Python
cat > demo_bad.md << 'EOF'
# Demo
```python
def bad(x,y):return x+y
```
EOF

# Show blacken-docs fixes it
uv run pre-commit run blacken-docs --files demo_bad.md
cat demo_bad.md  # Should show formatted Python
rm demo_bad.md
```

**Test YAML validation:**
```bash
# Create a bad example file
cat > examples/demo_bad.py << 'EOF'
def bad_function(x: int) -> str:
    return x  # Type error: returning int instead of str
EOF

# Show that pre-commit catches it
uv run pre-commit run --files examples/demo_bad.py
# Should fail with type error

# Clean up
rm examples/demo_bad.py
```

```bash
# Create bad YAML
cat > demo_bad.yaml << 'EOF'
key:  value
duplicate: one
duplicate: two
EOF

# Show yamllint catches issues
uv run pre-commit run yamllint --files demo_bad.yaml
# Should show errors for trailing spaces and duplicate keys

rm demo_bad.yaml
```

#### Step 8.6: Commit final documentation updates
```bash
git add bin/run-hooks.sh
git commit -m "docs: update run-hooks.sh for all new pre-commit tools

- Document shfmt, blacken-docs, and yamllint
- Add new formatters to --fix option
- Complete integration of all quality tools"
```

## Testing Your Implementation

### Manual Testing Checklist
- [ ] Shell scripts in `/bin` are validated by shellcheck
- [ ] Shell scripts in `.github` are validated by shellcheck
- [ ] Shell scripts are consistently formatted by shfmt
- [ ] Python files in `/examples` are type-checked by mypy
- [ ] Python files in `/examples` are formatted by ruff
- [ ] Python code in markdown files is formatted by blacken-docs
- [ ] YAML files are validated by yamllint
- [ ] Coverage reports still exclude `/examples`
- [ ] `./bin/run-hooks.sh --all` runs without errors
- [ ] `.github/quality.sh` checks examples

### Automated Test Commands
```bash
# Test everything
./bin/run-hooks.sh --all

# Test only Python type checking
uv run mypy src tests examples

# Test only shell scripts
uv run pre-commit run shellcheck --all-files
uv run pre-commit run shfmt --all-files

# Test documentation
uv run pre-commit run blacken-docs --all-files

# Test YAML files
uv run pre-commit run yamllint --all-files

# Test CI script
./.github/quality.sh
```

## Rollback Plan

If something goes wrong, you can rollback each change:

```bash
# Revert last commit
git revert HEAD

# Or reset to before changes (if not pushed)
git reset --hard origin/main

# Reinstall hooks
uv run pre-commit install
```

## Common Issues and Solutions

### Issue: "shellcheck: command not found"
Pre-commit will automatically install shellcheck. If it fails:
```bash
# Force reinstall
uv run pre-commit clean
uv run pre-commit install --install-hooks
```

### Issue: Many existing shell script errors
If existing scripts have many shellcheck warnings:
```bash
# Fix automatically fixable issues first
shellcheck -f diff bin/*.sh .github/*.sh | patch -p1

# Then manually fix remaining issues
```

### Issue: Type errors in existing examples
Fix them! Examples should be correct:
```bash
# See all type errors
uv run mypy examples --show-error-codes

# Fix each file, then test
uv run mypy examples/specific_file.py
```

### Issue: YAML validation errors
Common yamllint errors and fixes:
```bash
# See all YAML issues
uv run pre-commit run yamllint --all-files

# Common fixes:
# - Remove trailing spaces
# - Fix indentation (use 2 spaces)
# - Add newline at end of file
# - Remove duplicate keys
```

### Issue: Python code in markdown not formatting
Ensure code blocks are properly marked:
```markdown
```python
# Code must be in python-tagged blocks
def example():
    pass
```
```

## Summary

After completing all tasks:
1. All shell scripts are validated for common errors and formatted consistently
2. All Python examples are type-checked
3. Python code in documentation follows project style
4. YAML files are validated for syntax and style
5. CI/CD runs the same checks as local development
6. Coverage still correctly excludes examples

The codebase now has comprehensive quality checks for all code, documentation, and configuration files.

## Final Commit

After all tasks are complete, create a summary commit:

```bash
git add -A
git commit -m "feat: expand pre-commit to comprehensive code quality

- Add shellcheck for bash script validation
- Add shfmt for shell script formatting
- Include examples/ in mypy type checking
- Add blacken-docs for Python in markdown
- Add yamllint for YAML validation
- Update CI quality script to check examples
- Maintain coverage exclusion for examples

This ensures all code, documentation, and configuration in the
repository meets consistent quality standards"
```

## Next Steps

Consider these future improvements:
1. Set up pre-commit.ci for automatic PR checks
2. Add `prettier` for JSON/JavaScript/CSS formatting
3. Add `hadolint` for Dockerfile linting
4. Add custom hooks for project-specific validations
5. Configure hook-specific ignore patterns if needed
