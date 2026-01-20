# DQX Quick Reference

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

## Code Standards Quick Reference

| Standard | Rule |
|----------|------|
| Import order | stdlib → third-party → local |
| Type hints | Strict mode, all functions |
| Docstrings | Google style, required |
| Line length | 120 characters max |
| Coverage | 100% required |

## Quality Gates Checklist

Before completion:
- [ ] All tests passing: `uv run pytest`
- [ ] Coverage: 100%: `uv run pytest --cov=src/dqx --cov-report=term`
- [ ] Pre-commit passing: `uv run pre-commit run --all-files`
- [ ] No type errors: `uv run mypy src tests`

## Feature Workflow

1. **Planning**: Request feature → `dqx-plan` creates design docs
2. **Implementation**: Approve → `dqx-implement` executes TDD phases
3. **PR Creation**: Approve → `dqx-pr` creates pull request
4. **Feedback**: Request → `dqx-feedback` addresses review comments

## Complete Documentation

For comprehensive documentation, see [workflow_example.md](workflow_example.md) for an end-to-end example.
