# Active Context

## Current Work Focus

### Plugin Instance Registration Implementation (In Progress)
- Created TDD implementation plan for adding PostProcessor instance support to register_plugin method
- Working on feat/register-plugin-instances branch
- Plan includes 6 independently committable task groups following strict TDD
- Each task group passes all tests, linting, and pre-commit checks
- Maintains backward compatibility with string-based registration

### Pre-commit Configuration Improvements (Completed)
- Fixed duplicate pre-commit hook output by adding explicit `stages: [pre-commit]` to all file-checking hooks
- This ensures hooks only run during their intended stage, eliminating the duplicate "(no files to check)" messages
- Commit message validation continues to run separately at the commit-msg stage
- Solution implemented in commit 6e34bad

## Recent Changes

### Pre-commit Hooks (2025-10-19)
- Added explicit stages to all hooks in `.pre-commit-config.yaml`
- File-checking hooks now only run at pre-commit stage
- Commit message validation runs only at commit-msg stage
- Eliminates duplicate output during commits

### Test Coverage Improvements (2025-10-19)
- Fixed test_api_coverage.py by updating to use MetricProvider methods
- Added test_api_timer_fallback.py for timer error handling
- Achieved 100% test coverage for src/dqx/api.py

### Graph Module Refactoring (2025-10-18)
- Successfully implemented strongly-typed parent hierarchy
- All parent references now use specific node types
- Graph traversal maintains type safety throughout

## Next Steps

1. Continue with graph_built parameter removal on feature branch
2. Update any remaining tests affected by the changes
3. Ensure all functionality works without graph_built parameter

## Important Patterns and Preferences

### Git Workflow
- Using conventional commits (enforced by commitizen)
- Pre-commit hooks run comprehensive checks including:
  - Python syntax validation
  - Code formatting (ruff)
  - Type checking (mypy)
  - Security checks
  - File quality checks
- Hooks are configured to run at specific stages to avoid duplication

### Testing Standards
- Maintain 100% test coverage
- Use timer fallback patterns for resilient plugin execution
- Follow TDD approach for new features

## Learnings and Insights

### Pre-commit Hook Stages
- Hooks without explicit `stages` defined will run at all installed hook stages
- Adding `stages: [pre-commit]` restricts hooks to only run during pre-commit
- The commit-msg stage should only run commit message validation hooks
- This prevents duplicate output and improves developer experience

### Timer Error Handling
- VerificationSuite._process_plugins has fallback logic for timer failures
- Falls back to _execution_start time if available
- Returns 0.0 if neither timer nor execution start is available
- Important for plugin execution context reliability
