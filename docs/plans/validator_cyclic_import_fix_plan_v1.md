# Validator Cyclic Import Fix Plan v1

## Background

The DQX codebase has a cyclic import issue detected by mypy involving the validator module:

```text
'MetricProvider' may not be defined if module is imported before module,
as the of MetricProvider occurs after the cyclic of dqx.validator.
```

### Import Chain Analysis
1. `api.py` imports `SuiteValidator` from `validator.py`
2. `validator.py` imports `MetricProvider` from `provider.py` (under `TYPE_CHECKING`)
3. While there's no direct cycle, mypy detects a potential import ordering issue

## Solution Overview

Use a Protocol-based approach to break the cyclic dependency by defining a minimal interface in `validator.py` that represents only the methods needed by validators, eliminating the need to import `MetricProvider`.

## Implementation Tasks

### Task Group 1: Create MetricProviderProtocol (TDD)

**Objective**: Define a minimal Protocol interface for validators to use instead of importing MetricProvider.

#### Task 1.1: Write tests for MetricProviderProtocol
Create tests to ensure the protocol properly defines the required interface:

```python
# In tests/test_validator.py (add to existing tests)

def test_metric_provider_protocol_compatibility():
    """Test that MetricProvider implements MetricProviderProtocol."""
    from dqx.provider import MetricProvider
    from dqx.validator import MetricProviderProtocol

    # This should not raise any type errors when checked with mypy
    provider = MetricProvider(mock_db, "test-id")
    assert isinstance(provider, MetricProviderProtocol)
```

#### Task 1.2: Implement MetricProviderProtocol
Add the Protocol definition at the top of `validator.py`:

```python
# In validator.py, after existing imports but before any TYPE_CHECKING block
from typing import Protocol, Any
from collections.abc import Iterable
import sympy as sp

class MetricProviderProtocol(Protocol):
    """Minimal protocol for metric provider used by validators."""

    @property
    def metrics(self) -> list[Any]:
        """List of symbolic metrics."""
        ...

    def get_symbol(self, symbol: sp.Symbol) -> Any:
        """Get symbolic metric for a symbol."""
        ...

    def symbols(self) -> Iterable[sp.Symbol]:
        """Get all symbols."""
        ...

    def remove_symbol(self, symbol: sp.Symbol) -> None:
        """Remove a symbol."""
        ...
```

#### Task 1.3: Remove the TYPE_CHECKING import
Remove the entire TYPE_CHECKING block that imports MetricProvider:

```python
# REMOVE THIS ENTIRE BLOCK:
if TYPE_CHECKING:
    from dqx.provider import MetricProvider
```

**Verification**: Run `uv run mypy src/dqx/validator.py` - should pass without import errors.

### Task Group 2: Update Validator Type Hints

**Objective**: Update all validators to use MetricProviderProtocol instead of MetricProvider.

#### Task 2.1: Update DatasetValidator
Change the constructor parameter type:

```python
# In validator.py, class DatasetValidator
def __init__(self, provider: MetricProviderProtocol) -> None:
    """Initialize validator with provider."""
    super().__init__()
    self._provider = provider
```

#### Task 2.2: Update UnusedSymbolValidator
Change the constructor parameter type:

```python
# In validator.py, class UnusedSymbolValidator
def __init__(self, provider: MetricProviderProtocol) -> None:
    """Initialize validator with provider."""
    super().__init__()
    self._provider = provider
    self._used_symbols: set[sp.Symbol] = set()
    self._removed_symbols: list[str] = []
```

#### Task 2.3: Update SuiteValidator.validate method
Update the type hint in the validate method signature:

```python
# In validator.py, class SuiteValidator
def validate(self, graph: Graph, provider: MetricProviderProtocol) -> ValidationReport:
    """Run validation on a graph.

    Args:
        graph: The graph to validate
        provider: MetricProvider for dataset validation (required)

    Returns:
        ValidationReport with all issues found
    """
```

#### Task 2.4: Update SuiteValidator.validators method
Update the return type annotation:

```python
# In validator.py, class SuiteValidator
def validators(self, provider: MetricProviderProtocol) -> list[BaseValidator]:
    return [
        DuplicateCheckNameValidator(),
        EmptyCheckValidator(),
        DuplicateAssertionNameValidator(),
        DatasetValidator(provider),
        UnusedSymbolValidator(provider),
    ]
```

**Verification**: Run `uv run pytest tests/test_validator.py -v` - all tests should pass.

### Task Group 3: Final Verification

#### Task 3.1: Run full type checking
```bash
uv run mypy src/dqx/validator.py src/dqx/api.py src/dqx/provider.py
```
Should pass without cyclic import warnings.

#### Task 3.2: Run all validator-related tests
```bash
uv run pytest tests/test_validator.py tests/test_dataset_validator*.py -v
```
All tests should pass.

#### Task 3.3: Run pre-commit checks
```bash
uv run hooks
```
Should pass all checks including mypy and ruff.

## Success Criteria

1. No cyclic import warnings from mypy
2. All existing tests pass without modification
3. Type safety is maintained - mypy correctly validates that MetricProvider implements the protocol
4. No runtime behavior changes

## Notes for Implementation

- The Protocol uses `Any` type for return values to avoid importing SymbolicMetric
- This is acceptable because:
  - Validators only need to know these methods exist
  - They don't manipulate the returned objects directly
  - Type safety is maintained at the API boundary
- No changes needed in `api.py` or `provider.py`
- The Protocol approach follows SOLID principles (dependency inversion)
