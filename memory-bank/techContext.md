# DQX Technical Context

## Technology Stack

### Core Languages & Frameworks
- **Python 3.11/3.12**: Primary language
- **DuckDB ≥ 1.3.2**: SQL analytical engine
- **PyArrow ≥ 21.0.0**: Columnar data processing
- **SymPy ≥ 1.14.0**: Symbolic mathematics
- **DataSketches ≥ 5.2.0**: Probabilistic data structures
- **SQLAlchemy ≥ 2.0.43**: Database ORM/abstraction
- **Returns ≥ 0.26.0**: Functional programming utilities

### Development Tools
- **uv**: Modern Python package manager (preferred over pip)
- **pytest**: Testing framework with coverage support
- **mypy**: Static type checking
- **ruff**: Fast Python linter and formatter
- **pre-commit**: Git hook framework for code quality

### Development Environment
- **IDE**: Visual Studio Code (primary)
- **Shell**: zsh (default on macOS)
- **OS**: macOS (primary development)
- **Git**: Version control with GitLab
- **Python Version Management**: pyenv or uv

## Dependency Management

### Package Management with uv
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name

# Run commands in virtual environment
uv run python script.py
uv run pytest
uv run mypy src/
```

### Key Dependencies Explained

#### DuckDB
- **Purpose**: High-performance analytical SQL engine
- **Why**: Columnar storage, vectorized execution, embedded database
- **Usage**: All SQL query execution and data processing

#### PyArrow
- **Purpose**: Columnar memory format and data processing
- **Why**: Efficient data interchange, Parquet support, zero-copy reads
- **Usage**: Data source implementation, batch processing

#### SymPy
- **Purpose**: Symbolic mathematics library
- **Why**: Expression handling, mathematical operations, lazy evaluation
- **Usage**: Assertion expressions, metric combinations

#### DataSketches
- **Purpose**: Statistical sketching algorithms
- **Why**: Memory-efficient approximate computations
- **Usage**: Cardinality estimation, quantile sketches

#### SQLAlchemy
- **Purpose**: SQL toolkit and ORM
- **Why**: Database abstraction, connection management
- **Usage**: MetricDB persistence layer

#### Returns
- **Purpose**: Functional programming primitives
- **Why**: Result type for error handling, Maybe type for nulls
- **Usage**: Throughout codebase for safe error handling

## Development Setup

### Initial Setup Script
```bash
#!/bin/bash
# bin/setup-dev-env.sh

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run initial tests
uv run pytest tests/ -v
```

### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml highlights
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff-check  # Linting
      - id: ruff-format # Formatting

  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy        # Type checking

  - repo: https://github.com/shellcheck-py/shellcheck-py
    hooks:
      - id: shellcheck  # Shell script validation
```

## Code Quality Standards

### Type Annotations
- **Requirement**: All functions must have type annotations
- **Style**: Use modern Python typing (3.11+ features)
- **Protocols**: Prefer Protocol over ABC for interfaces

```python
from typing import Protocol, Sequence


class MetricSpec(Protocol):
    metric_type: MetricType

    @property
    def name(self) -> str: ...

    @property
    def analyzers(self) -> Sequence[Op]: ...
```

### Code Style
- **Formatter**: ruff format (Black-compatible)
- **Linter**: ruff check with extensive rules
- **Line Length**: 88 characters (Black standard)
- **Imports**: Sorted and grouped automatically

### Documentation
- **Docstrings**: Google format required
- **Type Stubs**: py.typed marker for type checking
- **Examples**: Comprehensive examples/ directory

## Testing Infrastructure

### Test Organization
```
tests/
├── test_*.py           # Unit tests
├── e2e/                # End-to-end tests (DO NOT MODIFY)
├── fixtures/           # Test data and fixtures
├── extensions/         # Extension-specific tests
├── graph/              # Graph module tests
└── orm/                # ORM tests
```

### Testing Commands
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=dqx --cov-report=html

# Run specific test file
uv run pytest tests/test_api.py -v

# Run tests in parallel
uv run pytest -n auto

# Run only marked tests
uv run pytest -m demo
```

### Test Coverage Requirements
- **Target**: 98%+ overall coverage
- **Achieved 100%**: graph.py, display.py, analyzer.py
- **Critical**: Never reduce coverage without approval

## Build and Release

### Version Management
- **Tool**: Commitizen for semantic versioning
- **Format**: MAJOR.MINOR.PATCH (e.g., 0.4.0)
- **Breaking Changes**: Increment MAJOR version

### Commit Standards
```bash
# Interactive commit
cz commit

# Commit types
feat:     # New feature
fix:      # Bug fix
docs:     # Documentation only
style:    # Code style (formatting)
refactor: # Code refactoring
perf:     # Performance improvement
test:     # Test additions/changes
chore:    # Build process or auxiliary tools
```

### Release Process
1. Update version with `cz bump`
2. Update CHANGELOG.md
3. Run full test suite
4. Create git tag
5. Push to GitLab
6. Create release notes

## Performance Considerations

### Profiling Tools
```python
# Memory profiling
from memory_profiler import profile


@profile
def memory_intensive_function():
    pass


# Time profiling
import cProfile

cProfile.run("suite.run(datasources, key)")
```

### Optimization Techniques
1. **SQL Query Batching**: Combine multiple metrics
2. **Set-Based Deduplication**: Efficient operation tracking
3. **Lazy Evaluation**: Compute only when needed
4. **Statistical Sketches**: Trade accuracy for memory

## Security Considerations

### Input Validation
- SQL injection prevention through parameterization
- File path validation for data sources
- Size limits on data processing

### Dependencies
- Regular security updates
- Minimal dependency footprint
- No unnecessary network calls

## Debugging Tips

### Logging Configuration
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("dqx")
```

### Common Issues

#### 1. Import Errors
```bash
# Ensure virtual environment is active
uv run python -c "import dqx"

# Check installed packages
uv pip list
```

#### 2. Type Checking Failures
```bash
# Run mypy with clear cache
uv run mypy src/ --no-incremental
```

#### 3. Test Failures
```bash
# Run with verbose output
uv run pytest -vvs tests/failing_test.py

# Debug with pdb
uv run pytest --pdb
```

## Integration Patterns

### Airflow Integration
```python
from airflow.decorators import task


@task
def run_dqx_validation():
    from dqx.api import VerificationSuiteBuilder

    suite = VerificationSuiteBuilder("Airflow Suite", db).build()
    return suite.run(datasources, key)
```

### dbt Integration
```yaml
# dbt test using DQX
tests:
  - dqx_validation:
      suite_name: "dbt Data Quality"
      checks:
        - completeness_check
        - freshness_check
```

### CI/CD Pipeline
```yaml
# .gitlab-ci.yml example
test:
  script:
    - uv sync
    - uv run pytest --cov=dqx
    - uv run mypy src/
    - uv run ruff check src/
```

## Monitoring and Observability

### Metrics Collection
```python
# Built-in metrics
from dqx.api import GraphStates

# Check execution state
for metric in context._graph.metrics():
    logger.info(f"Metric {metric.name}: {metric.state()}")
```

### Performance Monitoring
- Query execution time tracking
- Memory usage monitoring
- Metric computation statistics

## Troubleshooting Guide

### Common Problems

#### 1. DuckDB Connection Issues
- Check file permissions
- Verify disk space
- Review connection string

#### 2. Memory Errors
- Reduce batch size
- Enable streaming mode
- Use statistical sketches

#### 3. Slow Performance
- Check SQL query plans
- Verify indexes exist
- Review data distribution

### Debug Mode
```python
# Enable detailed logging
import os

os.environ["DQX_DEBUG"] = "1"

# Inspect graph structure
print(context._graph.display_tree())

# Check pending metrics
for metric in context._graph.pending_metrics():
    print(f"Pending: {metric.spec.name}")
```

## Future Technical Directions

### Planned Enhancements
1. **GPU Acceleration**: RAPIDS.ai integration
2. **Distributed Computing**: Ray/Dask support
3. **Streaming Support**: Apache Flink connector
4. **Cloud Native**: Kubernetes operators

### Technical Debt
1. Improve error message clarity
2. Add query plan optimization
3. Enhance type inference
4. Implement metric caching layer

### Research Areas
1. Incremental computation algorithms
2. Advanced statistical sketches
3. Query optimization strategies
4. Real-time anomaly detection
