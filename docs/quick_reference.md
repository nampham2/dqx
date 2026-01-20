# DQX Quick Reference

> **Note**: This is a quick reference card. See [AGENTS.md](../AGENTS.md) for complete details.

## Most Common Commands

```bash
# Testing
uv run pytest                                     # Run all tests
uv run pytest --cov=src/dqx --cov-report=term    # With coverage
uv run pytest -k "pattern"                        # Specific tests

# Quality
uv run ruff format                                # Format code
uv run ruff check --fix                           # Lint & fix
uv run mypy src tests                             # Type check
uv run pre-commit run --all-files                 # All hooks

# Development
uv sync                                           # Install deps
uv run mkdocs serve                               # Serve docs
```

See [AGENTS.md Commands Reference](../AGENTS.md#commands-reference) for all options.

## Code Standards Quick Reference

| Standard | Rule | Reference |
|----------|------|-----------|
| Import order | stdlib → third-party → local | [AGENTS.md §import-order](../AGENTS.md#import-order) |
| Type hints | Strict mode, all functions | [AGENTS.md §type-hints](../AGENTS.md#type-hints) |
| Docstrings | Google style, required | [AGENTS.md §docstrings](../AGENTS.md#docstrings) |
| Line length | 120 characters max | [AGENTS.md §formatting](../AGENTS.md#formatting) |
| Coverage | 100% required | [AGENTS.md §coverage-requirements](../AGENTS.md#coverage-requirements) |

## Quality Gates Checklist

Before completion:
- [ ] All tests passing: `uv run pytest`
- [ ] Coverage: 100%: `uv run pytest --cov=src/dqx --cov-report=term`
- [ ] Pre-commit passing: `uv run pre-commit run --all-files`
- [ ] No type errors: `uv run mypy src tests`

See [AGENTS.md Quality Gates](../AGENTS.md#quality-gates) for details.

## Feature Workflow

1. **Planning**: Request feature → `dqx-plan` creates design docs
2. **Implementation**: Approve → `dqx-implement` executes TDD phases
3. **PR Creation**: Approve → `dqx-pr` creates pull request
4. **Feedback**: Request → `dqx-feedback` addresses review comments

See [AGENTS.md Feature Workflow](../AGENTS.md#feature-development-workflow) for complete workflow.

## Complete Documentation

For comprehensive documentation:
- [AGENTS.md](../AGENTS.md) - Complete coding guidelines
- [workflow_example.md](workflow_example.md) - End-to-end example
