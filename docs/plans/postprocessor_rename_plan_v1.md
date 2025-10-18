# PostProcessor Rename Implementation Plan v1

## Overview
This plan details the renaming of `ResultProcessor` Protocol to `PostProcessor` and simplifies plugin validation by removing custom validation code in favor of Python's built-in `isinstance()` checking.

**Key Decisions:**
- No backward compatibility required (clean break)
- Follow TDD approach - write tests first, then implementation
- KISS principle - use Python's built-in Protocol checking instead of custom validation
- YAGNI principle - don't add extra validation we don't need
- **Git commits only when tests pass** - maintain clean git history

## Background
The DQX plugin system currently uses a Protocol called `ResultProcessor` for plugins that process validation results. We're renaming it to `PostProcessor` for better clarity and removing unnecessary custom validation code.

## Implementation Tasks

### Phase 1: Update Tests First (TDD)

#### Task 1.1: Update test imports and references
**Files to modify:** `tests/test_plugin_manager.py`

1. Search for all occurrences of "ResultProcessor" in the test file
2. Update test method names that reference ResultProcessor
3. Update assertion error messages that check for "ResultProcessor protocol"

**Example changes:**
```python
# OLD
def test_plugin_implements_protocol(self) -> None:
    """Test that plugins must implement ResultProcessor protocol."""

# NEW
def test_plugin_implements_protocol(self) -> None:
    """Test that plugins must implement PostProcessor protocol."""
```

#### Task 1.2: Add test for simplified validation
**File:** `tests/test_plugin_manager.py`

Add a new test to ensure the simplified validation works correctly:

```python
def test_plugin_validation_uses_isinstance(self) -> None:
    """Test that plugin validation uses isinstance for protocol checking."""
    manager = PluginManager()

    # Test with a class that doesn't implement the protocol
    with pytest.raises(ValueError, match="doesn't implement PostProcessor protocol"):
        manager.register_plugin("tests.fixtures.InvalidPlugin")

    # Test with a class that has wrong metadata return type
    with pytest.raises(ValueError, match="metadata\\(\\) must return a PluginMetadata instance"):
        manager.register_plugin("tests.fixtures.WrongMetadataPlugin")
```

#### Task 1.3: Run tests to confirm they fail
```bash
uv run pytest tests/test_plugin_manager.py -v
```

Expected: Tests should fail because PostProcessor doesn't exist yet.

**DO NOT COMMIT YET - Tests are failing**

### Phase 2: Rename Protocol and Update Implementation

#### Task 2.1: Rename the Protocol class
**File:** `src/dqx/plugins.py`

1. Find the Protocol definition (around line 23)
2. Rename class and update docstring:

```python
# OLD
@runtime_checkable
class ResultProcessor(Protocol):
    """Protocol for DQX result processor plugins."""

# NEW
@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for DQX post-processor plugins."""
```

#### Task 2.2: Update type annotations
**File:** `src/dqx/plugins.py`

Update all type hints that reference ResultProcessor:

1. In `PluginManager.__init__` (around line 53):
```python
# OLD
self._plugins: dict[str, ResultProcessor] = {}

# NEW
self._plugins: dict[str, PostProcessor] = {}
```

2. In `get_plugins` method (around line 60):
```python
# OLD
def get_plugins(self) -> dict[str, ResultProcessor]:

# NEW
def get_plugins(self) -> dict[str, PostProcessor]:
```

#### Task 2.3: Remove _validate_plugin_class method
**File:** `src/dqx/plugins.py`

Delete the entire `_validate_plugin_class` method (approximately lines 74-120). This follows YAGNI - we don't need custom validation when `isinstance()` does the job.

### Phase 3: Simplify register_plugin Method

#### Task 3.1: Rewrite register_plugin with isinstance validation
**File:** `src/dqx/plugins.py`

Replace the `register_plugin` method with this simplified version:

```python
def register_plugin(self, class_name: str) -> None:
    """
    Register a plugin by its fully qualified class name.

    Args:
        class_name: Fully qualified class name (e.g., 'dqx.plugins.AuditPlugin')

    Raises:
        ValueError: If class cannot be imported or doesn't implement PostProcessor
    """
    try:
        # Parse the class name
        parts = class_name.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid class name format: {class_name}")

        module_name, cls_name = parts

        # Import the module
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ValueError(f"Cannot import module {module_name}: {e}")

        # Get the class
        if not hasattr(module, cls_name):
            raise ValueError(f"Module {module_name} has no class {cls_name}")

        plugin_class = getattr(module, cls_name)

        # Instantiate the plugin
        plugin = plugin_class()

        # Use isinstance to check protocol implementation (KISS principle)
        if not isinstance(plugin, PostProcessor):
            raise ValueError(f"Plugin class {class_name} doesn't implement PostProcessor protocol")

        # Validate metadata returns correct type
        metadata = plugin.metadata()
        if not isinstance(metadata, PluginMetadata):
            raise ValueError(f"Plugin class {class_name}'s metadata() must return a PluginMetadata instance")

        plugin_name = metadata.name

        # Store the plugin
        self._plugins[plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin_name} (from {class_name})")

    except Exception as e:
        # Re-raise ValueError, let other exceptions propagate
        if not isinstance(e, ValueError):
            raise ValueError(f"Failed to register plugin {class_name}: {e}")
        raise
```

### Phase 4: Update Documentation

#### Task 4.1: Update plugin system documentation
**File:** `docs/plugin_system.md`

Change import statement:
```python
# OLD
from dqx.plugins import ResultProcessor

# NEW
from dqx.plugins import PostProcessor
```

Update any example implementations to use `PostProcessor`.

#### Task 4.2: Update architecture documentation
**File:** `docs/plans/plugin_architecture_plan_v2.md`

Update any references to ResultProcessor in the architecture documentation.

### Phase 5: Validation and Commit

#### Task 5.1: Run all tests
```bash
# Run specific plugin tests
uv run pytest tests/test_plugin_manager.py -v
uv run pytest tests/test_plugin_integration.py -v
uv run pytest tests/test_plugin_public_api.py -v

# If all pass, run full test suite
uv run pytest tests/ -v
```

**STOP HERE if any tests fail - fix them before proceeding**

#### Task 5.2: Run type checking
```bash
uv run mypy src/
```

**STOP HERE if mypy finds issues - fix them before proceeding**

#### Task 5.3: Run linting
```bash
uv run ruff check src/ tests/
```

**STOP HERE if ruff finds issues - fix them before proceeding**

#### Task 5.4: Run pre-commit hooks
```bash
./bin/run-hooks.sh
```

This will run all pre-commit checks including:
- ruff (linting)
- mypy (type checking)
- trailing whitespace
- end of file fixes

**STOP HERE if pre-commit finds issues - fix them before proceeding**

#### Task 5.5: Final test run
```bash
uv run pytest tests/ -v --cov=dqx
```

Ensure:
- All tests pass
- Coverage remains at 100%

#### Task 5.6: Commit changes
Only after all tests pass and all checks are clean:

```bash
git add -A
git commit -m "refactor: rename ResultProcessor to PostProcessor and simplify validation

- Rename ResultProcessor Protocol to PostProcessor for clarity
- Remove custom _validate_plugin_class method (YAGNI)
- Use isinstance() for protocol checking (KISS)
- Update all type hints and documentation
- No backward compatibility as per requirements"
```

## Testing Approach

### Unit Tests
- Update existing tests to use PostProcessor
- Add test for isinstance() validation
- Ensure error messages are correct

### Integration Tests
- Verify plugin loading still works
- Test with actual plugin implementations
- Ensure AuditPlugin continues to work

### Coverage Requirements
- Maintain 100% test coverage
- All new code must be tested
- Remove tests for deleted code

## Rollback Plan

If issues arise:
1. `git reset --hard HEAD~1` to undo the commit
2. `git checkout main` to return to main branch
3. `git branch -D refactor/postprocessor-rename` to delete the branch

## Success Criteria

1. All occurrences of `ResultProcessor` replaced with `PostProcessor`
2. Custom validation code removed
3. Tests pass with 100% coverage
4. Type checking passes
5. Linting passes
6. Documentation updated
7. Git history remains clean (single commit)

## Notes for Engineers

- This is a breaking change - no backward compatibility
- Follow TDD strictly - write/update tests before implementation
- Use KISS principle - simpler is better
- Apply YAGNI - don't add features we don't need
- Only commit when ALL tests pass
- If unsure about any step, ask for clarification

## Common Pitfalls to Avoid

1. **Don't forget to update documentation** - Both user-facing and architectural docs need updates
2. **Don't commit with failing tests** - This breaks the build for everyone
3. **Don't skip pre-commit hooks** - They catch issues before they reach the repo
4. **Don't add extra validation** - We're simplifying, not adding complexity
5. **Don't forget to update error messages** - They should reference PostProcessor, not ResultProcessor
