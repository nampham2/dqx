# DQX Technical Context

## Technology Stack

### Core Dependencies
- **Python**: 3.11+ (up to 3.12)
- **DuckDB**: 1.3.2+ - Primary SQL engine
- **PyArrow**: 21.0.0+ - Arrow data format support
- **SQLAlchemy**: 2.0.43+ - SQL abstraction
- **Returns**: 0.26.0+ - Functional error handling
- **SymPy**: 1.14.0+ - Symbolic mathematics
- **Rich**: 14.1.0+ - Terminal UI
- **Msgpack**: 1.1.1+ - Binary serialization
- **SQLParse**: 0.5.0+ - SQL parsing

### Development Tools
- **uv**: Package manager (replaces pip/poetry)
- **pytest**: Testing framework with 100% coverage target
- **mypy**: Static type checking with strict mode
- **ruff**: Fast Python linter and formatter
- **pre-commit**: Git hooks for code quality
- **commitizen**: Conventional commit enforcement
- **mkdocs**: Documentation generation

## Development Setup

### Initial Setup
```bash
# Clone repository
git clone git@github.com:nampham2/dqx.git
cd dqx

# Run setup script
./bin/setup-dev-env.sh

# This script will:
# 1. Install uv package manager
# 2. Create virtual environment
# 3. Install all dependencies
# 4. Set up pre-commit hooks
```

### Daily Workflow
```bash
# Always work in virtual environment
source .venv/bin/activate  # or let uv handle it

# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_cache.py::test_cache_hit

# Run with coverage
uv run pytest --cov=dqx --cov-report=html

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Testing Workflow

### Test-Driven Development (TDD) Process

**MANDATORY for every feature/bugfix**:

1. **Write Failing Test**:
```python
def test_new_feature() -> None:
    """Test that new feature works correctly."""
    # Arrange
    provider = MetricProvider(db, ExecutionId())

    # Act
    result = provider.new_operation()

    # Assert
    match result:
        case Success(value):
            assert value == expected
        case Failure(error):
            pytest.fail(f"Operation failed: {error}")
```

2. **Run Test to Confirm Failure**:
```bash
uv run pytest tests/test_new_feature.py -v
# Should see RED - test fails as expected
```

3. **Write Minimal Implementation**:
```python
def new_operation(self) -> Result[float, str]:
    """Implement just enough to pass test."""
    return Success(expected_value)
```

4. **Run Test to Confirm Success**:
```bash
uv run pytest tests/test_new_feature.py -v
# Should see GREEN - test passes
```

5. **Refactor While Keeping Tests Green**:
```bash
# Make improvements
# Run tests after each change
uv run pytest
# All tests must remain GREEN
```

### Testing Best Practices

**Real Objects Over Mocks**:
```python
# ✅ GOOD - Real objects
db = MetricDB(":memory:")
cache = MetricCache(db)

# ❌ BAD - Mocks
mock_db = Mock()
mock_cache = Mock()
```

**Pattern Matching for Results**:
```python
# ✅ GOOD - Pattern matching
match result:
    case Success(value):
        assert value == 42
    case Failure(error):
        pytest.fail(f"Unexpected failure: {error}")

# ❌ BAD - isinstance
assert isinstance(result, Success)
assert result.unwrap() == 42
```

## Git Workflow

### Branch Creation
```bash
# Feature branch
git checkout -b feature/custom-sql-operation

# Bugfix branch
git checkout -b bugfix/cache-thread-safety

# WIP branch (when exploring)
git checkout -b wip/investigating-performance
```

### Commit Process
```bash
# 1. Check changes
git --no-pager status
git --no-pager diff

# 2. Stage specific files
git add src/dqx/ops.py tests/test_ops.py

# 3. Commit with conventional format
git commit -m "feat(ops): add CustomSQL operation"

# 4. For complex commits
cat > .tmp/commit-msg.txt << EOF
feat(ops): add CustomSQL operation with universal parameter support

- Allows user-defined SQL expressions as metrics
- Parameters passed to CTE level for filtering/grouping
- Hash-based naming for unique identification
- Comprehensive test coverage
EOF

git commit --file .tmp/commit-msg.txt
rm .tmp/commit-msg.txt
```

### Pre-commit Hooks

**Never skip hooks**. If they fail:

```bash
# Run all hooks manually
uv run pre-commit run --all-files

# Common fixes:
# - Ruff formatting
uv run ruff format src/ tests/

# - Type errors
# Fix type annotations in code

# - Import sorting
# Ruff handles this automatically

# Try commit again
git commit
```

## Documentation Workflow

### Code Documentation
- All public APIs must have docstrings
- Use Google-style docstrings
- Include type information in docstrings
- Provide usage examples for complex APIs

Example:
```python
def analyze_single(
    self,
    datasource: SqlDataSource,
    metrics: Sequence[MetricSpec],
    key: ResultKey,
) -> AnalysisReport:
    """Analyze a single datasource for specified metrics.

    This method executes SQL queries to compute the requested metrics
    for a specific date and datasource combination.

    Args:
        datasource: Data source implementing SqlDataSource protocol
        metrics: List of metrics to compute
        key: Result key containing suite, date, and tags

    Returns:
        AnalysisReport containing computed metric values

    Example:
        >>> ds = DuckRelationDataSource(relation, "sales")
        >>> metrics = [MetricSpec.sum("revenue")]
        >>> key = ResultKey("daily", date(2024, 1, 1))
        >>> report = analyzer.analyze_single(ds, metrics, key)
    """
```

### Memory Bank Updates
- Update when discovering patterns
- Document after major changes
- Include learnings and insights
- Keep current with codebase

### Documentation Standards
- **README.md**: High-level project overview and examples
- **Memory Bank**: Detailed patterns and context
- **Code Comments**: Explain WHY, not WHAT
- **Commit Messages**: Clear conventional format
- **PR Descriptions**: High-level goals

## Build and Release

### Package Building
```bash
# Build package
uv build

# Output in dist/
# - dqlib-0.5.9.tar.gz
# - dqlib-0.5.9-py3-none-any.whl
```

### Version Management
- Semantic versioning (MAJOR.MINOR.PATCH)
- Version in pyproject.toml
- Use commitizen for version bumps
- Tag releases with git

### Release Process
```bash
# 1. Update version
# Edit pyproject.toml

# 2. Commit version bump
git commit -m "chore(release): bump version to 0.6.0"

# 3. Tag release
git tag -a v0.6.0 -m "Release version 0.6.0"

# 4. Push with tags
git push origin main --tags

# 5. Build and publish (if applicable)
uv build
# Upload to PyPI
```

## Environment Configuration

### Development Environment Variables
```bash
# Optional - for specific database testing
export DQX_TEST_DB="postgresql://user:pass@localhost/testdb"

# Enable debug logging
export DQX_DEBUG=1

# Set metric expiration (hours)
export DQX_METRIC_EXPIRATION=72
```

### VS Code Configuration
```json
{
  "python.linting.enabled": false,
  "python.formatting.provider": "none",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```

## Database Configuration

### Supported Databases
1. **DuckDB** (primary)
   - In-memory or file-based
   - Best performance for analytics
   - Native Arrow integration

2. **BigQuery** (supported)
   - Requires credentials
   - Different SQL dialect
   - Batch optimization available

### Connection Patterns
```python
# DuckDB in-memory
conn = duckdb.connect(":memory:")

# DuckDB persistent
conn = duckdb.connect("metrics.db")

# BigQuery
# Requires environment setup and credentials
```

## Performance Considerations

### SQL Generation
- Single query per dataset/date
- CTE-based for readability
- Batch multiple metrics together
- Minimize data scanning

### Caching Strategy
- In-memory cache for hot data
- Database fallback for cold data
- Thread-safe with minimal locks
- Performance statistics tracking

### Memory Management
- Lazy evaluation patterns
- Stream large results
- Clear caches when needed
- Monitor memory usage

## Security Guidelines

### Code Security
- No hardcoded credentials
- Use environment variables
- Validate all inputs
- SQL injection prevention via parameterization

### Dependency Security
- Regular dependency updates
- Security scanning in CI
- No unnecessary dependencies
- Audit third-party code

## Debugging Tips

### Common Issues

1. **Pattern Matching Errors**:
```python
# Check all cases handled
match result:
    case Success(v): ...
    case Failure(e): ...
    case _: pytest.fail("Unexpected case")
```

2. **Cache Consistency**:
```python
# Force cache refresh
cache.clear()
# Or check cache stats
print(f"Hit ratio: {cache.stats.hit_ratio()}")
```

3. **SQL Generation**:
```python
# Enable SQL logging
import logging
logging.getLogger("dqx.dialect").setLevel(logging.DEBUG)
```

### Debugging Tools
- Rich inspect: `from rich import inspect; inspect(obj)`
- SQL query logging
- Cache statistics
- Execution timing with Timer

## Integration Patterns

### As a Library
```python
from dqx import VerificationSuite, MetricProvider
from dqx.datasource import DuckRelationDataSource

# Create suite
suite = VerificationSuite("daily_checks")

# Add checks
with suite.check("Revenue validation") as check:
    revenue = ctx.metric_provider().sum("revenue")
    check.assert_that(revenue).is_positive()

# Execute
results = suite.run(datasources, dates)
```

### With Existing Systems
- Export results as JSON
- Store metrics in database
- Plugin system for custom processing
- Webhook notifications (via plugins)

## Documentation Reading Guide

### Essential Reading Order
1. **README.md** - Project overview and examples
2. **Memory Bank files** - Deep understanding
3. **tests/e2e/** - End-to-end examples
4. **Returns docs** - https://returns.readthedocs.io/en/latest/pages/result.html

### What NOT to Read
- **docs/** folder - Under construction, may be incorrect
- Old commit messages - Focus on current state
- Implementation details in comments - Read the code

### Learning Resources
- Integration tests show real usage
- E2E tests demonstrate workflows
- Memory bank captures patterns
- Git log shows evolution

## Tool Usage Patterns

### uv Commands
```bash
# Install dependencies
uv pip install -r requirements.txt

# Add new dependency
uv pip install package-name
uv pip freeze > requirements.txt

# Run any command in venv
uv run python script.py
uv run pytest
uv run mypy src/

# Build package
uv build
```

### pytest Patterns
```bash
# Run with output
uv run pytest -s

# Run specific test
uv run pytest tests/test_file.py::test_name

# Run with debugging
uv run pytest --pdb

# Parallel execution
uv run pytest -n auto

# Coverage report
uv run pytest --cov=dqx --cov-report=html
# Open htmlcov/index.html
```

### Type Checking
```bash
# Check all
uv run mypy src/

# Check specific file
uv run mypy src/dqx/cache.py

# Strict mode (default)
uv run mypy --strict src/
```

## Continuous Integration

### GitHub Actions
- Runs on every push/PR
- Tests multiple Python versions
- Type checking required
- Coverage must not decrease
- Pre-commit hooks enforced

### Local CI Simulation
```bash
# Run full CI locally
uv run pre-commit run --all-files
uv run pytest
uv run mypy src/
uv build
```

## Future Technical Directions

### Under Consideration
- Async/await support for I/O operations
- Additional SQL dialects (PostgreSQL, Snowflake)
- Streaming result processing
- Distributed execution support
- Real-time validation mode

### Technical Debt
- Some old code uses Optional instead of Maybe
- Migration from unittest to pytest style
- Consolidation of test utilities
- Documentation generation automation
