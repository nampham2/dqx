# Active Context

## Current Work Focus
- Successfully implemented `uv run hooks` command to replace the shell script `bin/run-hooks.sh`
- The new Python command provides identical functionality with better argument parsing and error handling

## Recent Changes (2025-01-27)
### Just Completed:
- Added `run_hooks()` function to `scripts/commands.py`:
  - Uses argparse for robust argument parsing with automatic --help
  - Supports all options: `--all`, `--fast`, `--fix`, `--check-commit`, `--help`
  - Accepts file paths as positional arguments
  - Maintains the same colored output using ANSI codes
  - DRY implementation with shared functions and clean code structure

- Updated `pyproject.toml`:
  - Added `hooks = "scripts.commands:run_hooks"` to `[project.scripts]` section
  - Command is now accessible via `uv run hooks`

- Removed `bin/run-hooks.sh`:
  - Shell script successfully replaced with Python implementation
  - All functionality preserved in the new command

## Key Implementation Details
- The function follows the same pattern as `run_coverage()` in the same file
- Uses subprocess to execute pre-commit commands with proper environment handling
- SKIP environment variable set when using `--fast` option
- Individual hook execution when using `--fix` option
- Proper exit code handling to match shell script behavior

## Command Usage
- `uv run hooks` - Run on staged files
- `uv run hooks --all` - Run on all files
- `uv run hooks --fast` - Skip mypy for faster checks
- `uv run hooks --fix` - Run only auto-fixing hooks
- `uv run hooks --check-commit` - Check last commit message
- `uv run hooks src/dqx/api.py` - Run on specific files
- `uv run hooks --help` - Show help message

## Next Steps
- The implementation is complete and tested
- Consider updating any documentation that references the old shell script
- Monitor for any edge cases in actual usage

## Important Patterns and Preferences
- Python commands in DQX are implemented in `scripts/commands.py`
- Commands are exposed via `[project.scripts]` in `pyproject.toml`
- Prefer Python implementations over shell scripts for better portability
- Use argparse for command-line argument parsing
- Maintain consistent output formatting with colored text

## Learnings and Project Insights
- The project uses uv for dependency management and command execution
- Pre-commit hooks are an important part of the development workflow
- Color output enhances user experience in CLI tools
- DRY principles apply to CLI tool implementation as well
