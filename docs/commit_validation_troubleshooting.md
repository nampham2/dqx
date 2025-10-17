# Commitizen Commit Validation Troubleshooting

## Issue: Commitizen Hook Not Running

### Problem
The commitizen pre-commit hook was configured but wasn't validating commit messages. It would show "(no files to check)Skipped" and allow any commit message format.

### Root Cause
The commitizen hook runs at the `commit-msg` stage, which requires:
1. Special installation of the commit-msg git hook
2. Proper handling of the commit message file path
3. Correct pre-commit configuration

### Solution

#### 1. Install commit-msg Hook
The setup script already includes this, but if needed manually:
```bash
uv run pre-commit install --hook-type commit-msg
```

#### 2. Create Wrapper Script
Created `bin/check-commit-msg.sh` to handle the commit message file properly:

```bash
#!/bin/bash
# Wrapper script for commitizen to work with pre-commit commit-msg hook
# Pre-commit sets PRE_COMMIT_COMMIT_MSG_FILENAME for commit-msg hooks

set -e

# For commit-msg hooks, pre-commit provides the file via environment variable
# or as the first argument
if [ -n "$PRE_COMMIT_COMMIT_MSG_FILENAME" ]; then
  COMMIT_MSG_FILE="$PRE_COMMIT_COMMIT_MSG_FILENAME"
elif [ -n "$1" ]; then
  COMMIT_MSG_FILE="$1"
else
  # Default to .git/COMMIT_EDITMSG
  COMMIT_MSG_FILE=".git/COMMIT_EDITMSG"
fi

if [ ! -f "$COMMIT_MSG_FILE" ]; then
  echo "Error: Commit message file not found at: $COMMIT_MSG_FILE"
  exit 1
fi

# Run commitizen check with the commit message file
uv run cz check --commit-msg-file "$COMMIT_MSG_FILE"
```

#### 3. Update Pre-commit Configuration
Configure the hook as a local hook in `.pre-commit-config.yaml`:

```yaml
  # Local hooks that use our project's virtual environment
  - repo: local
    hooks:
      # Commitizen - Conventional commit validation
      - id: commitizen-check
        name: Check commit message follows conventional format
        entry: ./bin/check-commit-msg.sh
        language: system
        stages: [commit-msg]
        pass_filenames: false  # Important: must be false
        always_run: true
```

### Testing

Test with invalid message:
```bash
git commit -m "bad message"
# Should fail with pattern explanation
```

Test with valid message:
```bash
git commit -m "fix(pre-commit): fix commitizen validation"
# Should succeed
```

### Conventional Commit Format

Valid commit messages must follow:
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

Examples:
- `feat(analyzer): add query optimization`
- `fix(api): handle null values correctly`
- `docs: update installation guide`
