# DQX Technical Context

## Technology Stack

### Core Dependencies
- **Python 3.11+**: Required for modern type hints and performance
- **SymPy (1.14.0+)**: Symbolic mathematics for expressions
- **DuckDB (1.3.2+)**: In-memory SQL engine for analytics
- **PyArrow (21.0.0+)**: Efficient columnar data processing
- **SQLAlchemy (2.0.43+)**: Database abstraction for metric storage
- **Rich (14.1.0+)**: Terminal formatting for result display
- **Returns (0.26.0+)**: Functional error handling (Result type)

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
├── dialect.py           # SQL dialect implementations
├── display.py           # Result visualization
├── evaluator.py         # Assertion evaluation
├── functions.py         # Mathematical functions
├── models.py            # Data models
├── ops.py               # Metric operations
├── provider.py          # Metric provider
├── specs.py             # Metric specifications
├── states.py            # State management
├── utils.py             # Utility functions
├── validator.py         # Suite validation
├── extensions/          # Data source extensions
├── graph/              # Graph implementation
└── orm/                # Database persistence
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

### .pre-commit-config.yaml
- Automated code quality checks
- Runs on every commit
- Includes: ruff, mypy, trailing whitespace

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
```

### Result Types
```python
from returns.result import Result, Success, Failure

Result[float, str]  # Success has float, Failure has error string
```

## Testing Philosophy

### Test-Driven Development
1. Write failing test first
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Maintain 100% coverage

### Test Categories
- **Unit tests**: Individual components
- **Integration tests**: Component interactions
- **E2E tests**: Full workflow validation
- **Demo tests**: Example usage patterns

### Testing Best Practices
- Prefer native objects over mocks
- Test actual behavior, not implementation
- Clear test names describing scenarios
- Isolated tests with no dependencies

## Performance Considerations

### Query Optimization
- Single-pass SQL generation
- CTE-based query structure
- Push computation to database
- Minimize data transfer

### Memory Management
- No data materialization in Python
- Streaming results processing
- Constant memory usage
- Lazy symbol evaluation

## Security Considerations

### SQL Injection Prevention
- Parameterized queries only
- Column name validation
- No dynamic SQL construction

### Data Access
- Read-only data source access
- Separate metric storage credentials
- No data modification capabilities

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
