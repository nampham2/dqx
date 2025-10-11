# Verification Suite Validator Plan - Architectural Feedback

## Overview

This document provides constructive feedback on the verification suite validator implementation plan. The validator aims to catch configuration errors (duplicate names, empty checks, etc.) before suite execution.

## Positive Aspects to Keep

1. **TDD Approach**: Excellent alignment with project's test-driven development principles
2. **Clear Module Structure**: Good separation of concerns with base, report, rules, and suite_validator modules
3. **Extensible Design**: Easy to add new validation rules following the established pattern
4. **Correct Integration Timing**: Validation in `collect()` or early in `run()` BEFORE `impute_datasets()` is the right approach

## Recommended Changes

### 1. Remove Severity from ValidationIssue

**Current design has unnecessary complexity:**
```python
@dataclass
class ValidationIssue:
    rule: str
    severity: Literal["error", "warning", "info"]  # <-- Remove this
    message: str
    node_path: list[str]
```

**Why remove it:**
- Severity is a property of the rule type, not individual issues
- Avoids confusion with existing assertion severity levels (P0, P1, P2, P3)
- Simplifies the design - each rule inherently produces errors OR warnings

**Suggested approach:**
```python
@dataclass
class ValidationIssue:
    rule: str
    message: str
    node_path: list[str]

# Let rules declare their type
class DuplicateCheckNameRule:
    name = "duplicate_check_names"
    is_error = True  # Always produces errors

class EmptyCheckRule:
    name = "empty_checks"
    is_error = False  # Always produces warnings
```

### 2. Remove UnnamedAssertionRule

The current DQX API already enforces named assertions through the `AssertionDraft` -> `AssertionReady` pattern:
- Users MUST call `where(name="...")` before making any assertion
- The API won't allow unnamed assertions
- This rule would never trigger in practice

Keep only rules that can actually detect issues:
- `DuplicateCheckNameRule` (error)
- `DuplicateAssertionNameRule` (error)
- `EmptyCheckRule` (warning)

### 3. Consider Visitor Pattern for Consistency

DQX already uses the visitor pattern extensively (e.g., `DatasetImputationVisitor`). Consider making validation rules follow this pattern:

```python
from dqx.graph.base import NodeVisitor

class DuplicateCheckNameVisitor(NodeVisitor):
    """Visitor that detects duplicate check names."""

    def __init__(self):
        self.issues = []
        self.check_names = defaultdict(list)

    def visit_check_node(self, node: CheckNode) -> None:
        self.check_names[node.name].append(node)

    def visit_root_node(self, node: RootNode) -> None:
        # Process children
        for child in node.children:
            child.accept(self)

    def finalize(self) -> list[ValidationIssue]:
        # Process duplicates after traversal
        for name, nodes in self.check_names.items():
            if len(nodes) > 1:
                self.issues.append(ValidationIssue(
                    rule="duplicate_check_names",
                    message=f"Duplicate check name: '{name}' appears {len(nodes)} times",
                    node_path=["root", f"check:{name}"]
                ))
        return self.issues
```

This aligns better with existing DQX patterns and provides cleaner separation of traversal from validation logic.

### 4. Simplify Error Handling

- Only raise `DQXError` when there are actual errors (not warnings)
- Include all issues in the error message for better debugging
- Consider adding a `fail_on_warnings` flag for strict mode

```python
def collect(self, context: Context, key: ResultKey) -> None:
    # ... existing code ...

    # Run validation
    report = self._validator.validate(context._graph)

    if report.has_errors():
        raise DQXError(f"Suite validation failed:\n{report}")
    elif report.has_warnings():
        logger.warning(f"Suite validation warnings:\n{report}")
```

### 5. Testing Considerations

Add tests for:
- **Thread safety**: Context uses thread-local storage, ensure validation works correctly with concurrent check execution
- **Large suites**: Performance testing with many checks/assertions
- **Edge cases**: Empty suite, suite with only one check, etc.

## Implementation Tips

1. **Path Representation**: Consider using actual node references instead of string paths for better programmatic access
2. **Rule Configuration**: Allow rules to be configurable (e.g., severity level per rule instance)
3. **Performance**: For large suites, consider lazy validation or parallel rule execution

## Summary

The validation plan is well-structured and will provide valuable early error detection. With these adjustments:
- Remove unnecessary severity complexity
- Align with existing DQX patterns (visitor pattern)
- Focus on rules that can actually trigger
- Keep the correct validation timing (before dataset operations)

The implementation will integrate seamlessly with DQX's architecture while maintaining simplicity and extensibility.
