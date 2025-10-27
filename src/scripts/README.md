# Development Scripts

This directory contains development utility scripts that are not part of the main `dqx` package.

## Available Commands

### Running Tests with Coverage
```bash
python scripts/commands.py coverage
```

### Running Pre-commit Hooks
```bash
python scripts/commands.py hooks [options]

# Options:
#   --all           Run on all files
#   --fast          Skip slow hooks (mypy)
#   --fix           Only run auto-fixing hooks
#   --check-commit  Check last commit message
```

### Cleaning Cache Files
```bash
python scripts/commands.py cleanup [options]

# Options:
#   --dry-run       Show what would be deleted
#   --verbose, -v   Show detailed progress
#   --quiet, -q     Suppress output
#   --all           Include .venv directory (dangerous!)
```

## Alternative: Using Python Module Execution
```bash
# From the project root:
python -m scripts.commands coverage
python -m scripts.commands hooks --help
python -m scripts.commands cleanup --dry-run
```

## Note
These scripts are development tools and are intentionally kept separate from the main `dqx` package to avoid including development dependencies in the distributed package.
