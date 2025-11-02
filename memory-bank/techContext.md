# DQX Technical Context

## Technology Stack

### Core Dependencies
- **Python 3.11+**: Required for modern type hints and performance
- **SymPy (1.14.0+)**: Symbolic mathematics for expressions
- **DuckDB (1.3.2+)**: In-memory SQL engine for analytics
- **PyArrow (21.0.0+)**: Efficient columnar data processing
- **SQLAlchemy (2.0.43+)**: Database abstraction for metric storage
- **Rich (14.1.0+)**: Terminal formatting for result display (moved from dev to core)
- **Returns (0.26.0+)**: Functional error handling (Result type)
- **sqlparse (0.5.0+)**: SQL formatting and beautification

### Supporting Libraries
- **numpy (2.3.2+)**: Numerical operations
- **datasketches (5.2.0+)**: Approximate algorithms (cardinality)
- **msgpack (1.1.1+)**: Efficient binary serialization

### Development Dependencies
- **uv**: Modern Python package manager (replacement for pip/poetry)
- **pytest (8.4.1+)**: Testing framework
- **mypy (1.17.1+)**: Static type checking
- **ruff (0.12.10+)**: Fast Python linter and formatter
- **pre-commit (4.3.0+)**: Git hook framework
- **pytest-cov (6.2.1+)**: Test coverage reporting
- **faker (37.5.3+)**: Test data generation
- **yamllint**: YAML file validation (new)
- **shfmt**: Shell script formatting (new)
- **shellcheck**: Shell script validation (new)

## Development Setup

### Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd dqx

# Run setup script (installs uv and dependencies)
./bin/setup-dev-env.sh
```

### Virtual Environment Management
- Managed by `uv` automatically
- Python version from `.python-version` file (3.11)
- All commands run through `uv run <command>`

### Common Commands
```bash
# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_api.py -v

# Check coverage
uv run pytest --cov=dqx --cov-report=html

# Run pre-commit checks
./bin/run-hooks.sh

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Format code
uv run ruff check --fix src/
```

## Project Structure

### Source Layout
```
src/dqx/
├── __init__.py          # Package initialization
├── api.py               # User-facing API
├── analyzer.py          # SQL analysis engine
├── common.py            # Shared types and protocols
├── compute.py           # Computation utilities
├── datasource.py        # Data source implementations
├── dialect.py           # SQL dialect implementations
├── display.py           # Result visualization
├── evaluator.py         # Assertion evaluation
├── functions.py         # Mathematical functions
├── models.py            # Data models
├── ops.py               # Metric operations
├── plugins.py           # Plugin system
├── provider.py          # Metric provider
├── specs.py             # Metric specifications
├── states.py            # State management
├── timer.py             # Performance timing
├── utils.py             # Utility functions
├── validator.py         # Suite validation
├── graph/              # Graph implementation
├── orm/                # Database persistence
└── plugins/            # Built-in plugins
```

### Test Structure
```
tests/
├── test_*.py           # Unit tests (matching src/)
├── e2e/                # End-to-end tests
├── fixtures/           # Test data
└── conftest.py         # Pytest configuration
```

## Configuration Files

### pyproject.toml
- Project metadata and dependencies
- Tool configurations (ruff, mypy, pytest)
- Build system configuration
- Version management

### .pre-commit-config.yaml
- Automated code quality checks
- Runs on every commit
- Includes: ruff, mypy, trailing whitespace, yamllint, shfmt, shellcheck
- Ensures consistent code quality

### .python-version
- Specifies Python 3.11 for the project
- Used by uv for environment creation

## SQL Dialects

### DuckDB (Default)
- In-memory analytical database
- Supports window functions, CTEs
- Fast columnar processing
- No external dependencies

### BigQuery
- Google Cloud data warehouse
- Requires additional setup
- Supports massive scale

### PyArrow Integration
- Works through DuckDB
- Efficient for local data files
- Parquet file support

### Future Dialects
- PostgreSQL (planned for v0.4.0)
- Snowflake (under consideration)
- MySQL/MariaDB (evaluation phase)

## Database Persistence

### Supported Backends
- **SQLite**: Local development
- **PostgreSQL**: Production deployments
- **In-Memory**: Testing and demos

### Schema
- Metrics table: Stores computed values
- Results table: Stores assertion outcomes
- Automatic migration on startup

## Type System

### Type Hints Everywhere
```python
def average(
    self, column: str, dataset: str | None = None, key: ResultKeyProvider | None = None
) -> sp.Expr:
    pass
```

### Protocols for Extensions
```python
class MetricSpec(Protocol):
    def to_sql(self, table_alias: str, dialect: Dialect) -> str: ...


class PostProcessor(Protocol):
    @staticmethod
    def metadata() -> PluginMetadata: ...

    def process(self, context: PluginExecutionContext) -> None: ...
```

### Result Types
```python
from returns.result import Result, Success, Failure

Result[float, str]  # Success has float, Failure has error string
```

## Testing Philosophy

### Core Principle
**Test code quality equals source code quality**. Tests should be concise, modern, and maintainable with full type annotations.

### Test-Driven Development
1. Write failing test first
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Maintain or exceed current coverage levels

### Test Categories
- **Unit tests**: Individual components
- **Integration tests**: Component interactions
- **E2E tests**: Full workflow validation
- **Demo tests**: Example usage patterns
- **Plugin tests**: Plugin functionality

### Testing Best Practices

#### 1. Real Objects Over Mocks
- **Use real objects whenever possible** - mocks hide bugs
- Prefer in-memory databases over database mocks
- Use real instances of classes, not mock objects
- Only mock external services that can't be controlled

#### 2. Type Safety Required
- **All test code MUST have type annotations**
- No `Any` types unless absolutely necessary
- Full type hints for fixtures, parameters, and returns
- Type check tests with mypy

#### 3. Returns Library Testing
- **Always use pattern matching** for Result/Maybe types
- Never use `isinstance(result, Success/Failure)`
- Never use `hasattr` to check for methods
- Pattern matching is the ONLY correct approach

```python
# Testing Result types
match result:
    case Success(value):
        assert value == expected
    case Failure(error):
        pytest.fail(f"Expected Success: {error}")

# Testing Maybe types
match maybe_value:
    case Some(value):
        assert value.name == "test"
    case Nothing():
        pytest.fail("Expected Some, got Nothing")
```

#### 4. Minimal Tests, Maximal Coverage
- Write fewest tests that provide most coverage
- Use parametrized tests for similar scenarios
- Focus on critical paths and edge cases
- No need for 100% coverage unless requested

#### 5. Functional Testing Style
- Chain operations functionally where appropriate
- Use Result/Maybe composition features
- Test pipelines end-to-end
- Verify functional transformations

### Testing Standards
- Clear, descriptive test names
- Isolated tests with no dependencies
- Test error paths as thoroughly as success paths
- Use fixtures for common setup
- Group related tests in classes
- Maintain test performance (fast feedback)

## Performance Considerations

### Query Optimization
- Single-pass SQL generation
- CTE-based query structure
- Push computation to database
- Minimize data transfer
- SQL formatting for readability

### Memory Management
- No data materialization in Python
- Streaming results processing
- Constant memory usage
- Lazy symbol evaluation

### Plugin Performance
- 60-second hard timeout
- Time tracking for each plugin
- Isolated execution environment

## Security Considerations

### SQL Injection Prevention
- Parameterized queries only
- Column name validation
- No dynamic SQL construction

### Data Access
- Read-only data source access
- Separate metric storage credentials
- No data modification capabilities

### Plugin Security
- Time-limited execution
- Error isolation
- No access to sensitive internals

## Git Workflow

### Conventional Commits Required
All commits and PR titles **MUST** follow conventional commit format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**: feat, fix, docs, style, refactor, perf, test, build, ci, chore

### Commit Process
1. **Read context first**: `git --no-pager diff`, check README.md, code, tests
2. **Prepare message**: <10 lines, focus on high-level goal
3. **Get permission**: Show message before committing
4. **Never auto-push**: Only push when explicitly asked

**For complex commits**:
```bash
# Create message file
echo "type(scope): description

Body..." > .tmp/commit-msg.txt

git commit --file .tmp/commit-msg.txt
rm .tmp/commit-msg.txt
```

### Pull Request Process
1. **Check clean branch**: `git --no-pager status`
2. **Read all changes**: `git --no-pager diff main...HEAD`
3. **Push branch**: `git push -u origin <branch>`
4. **Create PR**: Title in conventional format, body <10 lines

**For complex PRs**:
```bash
# Create PR body
cat > .tmp/pr-body.md << EOF
High-level summary...
EOF

gh pr create --title "type(scope): description" \
             --body-file .tmp/pr-body.md
rm .tmp/pr-body.md
```

### Branch Naming
- feature/description
- bugfix/issue-description
- release/version
- hotfix/urgent-fix

### Git Commands
- **Always use --no-pager**: `git --no-pager log`
- **Never use `git add -A`** without checking status
- **Commit frequently**: Small, focused commits
- **Never skip pre-commit hooks**: Fix and retry

## Deployment Patterns

### Library Usage
```bash
pip install dqx
```

### Docker Container
```dockerfile
FROM python:3.11-slim
RUN pip install dqx
# Add your checks
```

### CI/CD Integration
- Run checks in pipeline
- Fail build on quality issues
- Track metrics over time
- Plugin-based reporting

## Debugging Tips

### Enable Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Inspect Graph
```python
suite.graph.display()  # Visual representation
```

### Check SQL Generation
```python
analyzer._build_query(metrics)  # See generated SQL
```

### Debug Plugins
```python
# Check loaded plugins
suite.plugin_manager.get_plugins()

# Test plugin directly
plugin.process(context)
```

## Common Issues

### Import Errors
- Ensure Python 3.11+
- Check virtual environment activation
- Verify all dependencies installed

### Type Errors
- Run mypy for detailed messages
- Check protocol implementations
- Verify type annotations

### Test Failures
- Check test isolation
- Verify database state
- Look for timing dependencies

### Plugin Issues
- Check plugin registration
- Verify metadata() returns PluginMetadata
- Ensure process() handles all context fields

## Recent Changes

### Removed Features
- Batch processing support (simplified architecture)
- Threading infrastructure (no longer needed)
- Parallel execution capabilities
- Extensions directory (consolidated into datasource.py)

### Added Tools
- yamllint for YAML validation
- shfmt for shell script formatting
- shellcheck for shell script linting
- sqlparse for SQL formatting

### Dependency Changes
- Rich moved from dev to core dependencies
- Added sqlparse as core dependency
- Updated pre-commit hooks configuration
- Consolidated data sources from extensions/ to datasource.py module
