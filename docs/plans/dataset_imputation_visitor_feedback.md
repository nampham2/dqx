# Dataset Imputation Visitor Implementation Feedback

## Overview
This document provides feedback on the proposed dataset imputation visitor implementation plan after reviewing the existing codebase and understanding the context.

## Key Clarifications from Nam
1. **SymPy constants are not a problem** - Datasets are only specified for SymbolicMetric, not for SymPy symbols
2. **Dataset imputation should be idempotent** - Running it multiple times should produce the same result
3. **SymbolicMetric has a single dataset** - There should never be conflicts from multiple assertions

## Good News - Foundation Already Exists

### 1. NodeVisitor Protocol
- Already defined in `src/dqx/graph/base.py` as a Protocol
- No need to create a base class, just implement the protocol
- Requires `visit` and `visit_async` methods

### 2. Symbol Lookup
- `MetricProvider.get_symbol()` method works exactly as needed
- Takes a SymPy symbol and returns the corresponding SymbolicMetric

### 3. Graph Traversal Infrastructure
- `Graph` class already has DFS/BFS methods that accept visitors
- Well-tested traversal algorithms ready to use

## Recommended Improvements to the Plan

### 1. Simplify Visitor Implementation
Since `NodeVisitor` is a Protocol, not a base class:

```python
class DatasetImputationVisitor:  # No inheritance needed
    """Visitor that propagates and validates dataset information through the graph."""

    def __init__(self, available_datasets: list[str], provider: MetricProvider):
        if not available_datasets:
            raise DQXError("At least one dataset must be provided!")
        self.available_datasets = available_datasets
        self.provider = provider
        self.errors: list[str] = []

    def visit(self, node: BaseNode) -> None:
        """Implementation of NodeVisitor protocol."""
        if isinstance(node, RootNode):
            self._visit_root(node)
        elif isinstance(node, CheckNode):
            self._visit_check(node)
        elif isinstance(node, AssertionNode):
            self._visit_assertion(node)

    async def visit_async(self, node: BaseNode) -> None:
        """Async implementation - can just call visit() if no async work needed."""
        self.visit(node)
```

### 2. Update Graph.impute_datasets() Signature
The method needs to accept the provider parameter:

```python
def impute_datasets(self, datasets: list[str], provider: MetricProvider) -> None:
    """Propagate dataset information through the graph using visitor pattern."""
    from dqx.graph.visitors import DatasetImputationVisitor

    visitor = DatasetImputationVisitor(datasets, provider)
    self.dfs(visitor)  # Use DFS to ensure parent datasets are set before children

    if visitor.has_errors():
        raise DQXError(visitor.get_error_message())
```

### 3. Ensure Idempotency
Since dataset imputation should be idempotent:
- Only modify datasets that are None (not explicitly set)
- Consider clearing previously imputed datasets before starting
- Preserve explicitly set datasets on SymbolicMetrics

### 4. Enhanced Test Cases
Add these test cases to ensure robustness:

```python
def test_imputation_is_idempotent(self):
    """Running imputation twice produces the same result."""
    # Set up graph
    # Run imputation once
    # Capture state
    # Run imputation again
    # Assert state unchanged


def test_preserves_explicitly_set_datasets(self):
    """Datasets explicitly set on SymbolicMetric are preserved."""
    # Create metric with dataset="prod"
    # Run imputation with ["prod", "staging"]
    # Assert metric still has dataset="prod"


def test_handles_missing_symbols_gracefully(self):
    """Visitor handles symbols without SymbolicMetrics."""
    # Create assertion with symbol not in provider
    # Run imputation
    # Should not fail, just skip the symbol
```

### 5. Improved Error Handling
Consider adding methods for better error access:

```python
def get_errors(self) -> list[str]:
    """Get list of individual error messages."""
    return self.errors.copy()


def get_error_summary(self) -> str:
    """Get summary of all errors."""
    if not self.errors:
        return ""
    error_count = len(self.errors)
    return f"Dataset validation failed with {error_count} error(s):\n" + "\n".join(
        f"  - {err}" for err in self.errors
    )
```

### 6. Optional Fail-Fast Mode
For large graphs, consider adding an option to stop on first error:

```python
def __init__(
    self,
    available_datasets: list[str],
    provider: MetricProvider,
    fail_fast: bool = False,
):
    self.fail_fast = fail_fast
    # ... other init code ...


def _add_error(self, error: str) -> None:
    self.errors.append(error)
    if self.fail_fast:
        raise DQXError(error)
```

## Minor Corrections to the Plan

### 1. AssertionNode Initialization
- The plan mentions removing `datasets` from `AssertionNode.__init__`
- Current code doesn't take datasets in `__init__`, just initializes empty list
- Only need to remove the `impute_datasets` method

### 2. Parent Access Pattern
- The plan correctly uses `node.parent` in `_visit_assertion`
- This works because `BaseNode` has parent attribute set by `CompositeNode.add_child()`

### 3. Traversal Order
- Use DFS instead of BFS to ensure parent nodes are processed before children
- This guarantees CheckNode datasets are set before AssertionNode validation

## Benefits of This Refactoring

1. **Separation of Concerns**: Dataset logic moved out of node classes
2. **Centralized Error Handling**: All validation errors in one place
3. **Better Testability**: Visitor can be tested independently
4. **Follows Established Patterns**: Uses existing visitor infrastructure
5. **More Maintainable**: All dataset logic in one file

## Implementation Order Recommendation

1. Write comprehensive tests first (TDD)
2. Implement basic visitor structure
3. Add CheckNode logic (validation + inheritance)
4. Add AssertionNode/SymbolicMetric logic
5. Add error reporting
6. Update Graph class
7. Remove old methods from nodes
8. Update API layer
9. Add integration tests
10. Update documentation

## Conclusion

The plan is well-structured and will significantly improve the codebase. With these minor adjustments:
- Recognizing NodeVisitor is a Protocol, not a base class
- Ensuring idempotency for repeated runs
- Adding comprehensive test coverage
- Proper error handling and reporting

The implementation should be straightforward and achieve the goal of cleaner, more maintainable code.
