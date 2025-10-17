# Coding Standards - Python & Type Hints

## Core Standards

- Follow PEP 8 style guide for Python code
- Use 4-space indentation (no tabs)
- **Type hints required**: `def func(x: int) -> str:` (mandatory)
- Use docstrings for all public modules, classes, and functions
- **Always use f-strings**: `f"Value: {var}"` not `"Value: " + str(var)`

## Modern Type Hints (PEP 604)

```python
# ✅ CORRECT - Modern syntax
def process_data(items: list[str], config: dict[str, int] | None = None) -> str | None:
    return f"Processed {len(items)} items"


# ❌ AVOID - Old syntax
from typing import Dict, List, Optional, Union


def process_data(
    items: List[str], config: Optional[Dict[str, int]] = None
) -> Union[str, None]:
    return "Processed " + str(len(items)) + " items"
```

## Linting & Formatting

- Line length limit: 140 characters (configured in pyproject.toml)

## Best Practices and Patterns

- Use dataclasses or Pydantic models for data structures
- Prefer composition over inheritance
- Use async/await for I/O operations
- Follow SOLID principles
- Use dependency injection where appropriate
- Implement proper error handling with custom exceptions
- **Start with simplest solution that works**
- **Add complexity only when requirements demand it**

## Writing Code Principles

- When submitting work, verify that you have FOLLOWED ALL RULES. (See Rule #1)
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones. Readability and maintainability are PRIMARY CONCERNS, even at the cost of conciseness or performance.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission. If you're considering this, YOU MUST STOP and ask first.
- YOU MUST get Nam's explicit approval before implementing ANY backward compatibility.
- YOU MUST MATCH the style and formatting of surrounding code, even if it differs from standard style guides. Consistency within a file trumps external standards.
- YOU MUST NOT manually change whitespace that does not affect execution or output. Otherwise, use a formatting tool.
- Fix broken things immediately when you find them. Don't ask permission to fix bugs.

## Naming Conventions

- Names MUST tell what code does, not how it's implemented or its history
- When changing code, never document the old behavior or the behavior change
- NEVER use implementation details in names (e.g., "ZodValidator", "MCPWrapper", "JSONParser")
- NEVER use temporal/historical context in names (e.g., "NewAPI", "LegacyHandler", "UnifiedTool", "ImprovedInterface", "EnhancedParser")
- NEVER use pattern names unless they add clarity (e.g., prefer "Tool" over "ToolFactory")

Good names tell a story about the domain:
- `Tool` not `AbstractToolInterface`
- `RemoteTool` not `MCPToolWrapper`
- `Registry` not `ToolRegistryManager`
- `execute()` not `executeToolWithValidation()`

## Code Comments

- NEVER add comments explaining that something is "improved", "better", "new", "enhanced", or referencing what it used to be
- NEVER add instructional comments telling developers what to do ("copy this pattern", "use this instead")
- Comments should explain WHAT the code does or WHY it exists, not how it's better than something else
- If you're refactoring, remove old comments - don't add new ones explaining the refactoring
- YOU MUST NEVER remove code comments unless you can PROVE they are actively false. Comments are important documentation and must be preserved.
- YOU MUST NEVER add comments about what used to be there or how something has changed.
- YOU MUST NEVER refer to temporal context in comments (like "recently refactored" "moved") or code. Comments should be evergreen and describe the code as it is. If you name something "new" or "enhanced" or "improved", you've probably made a mistake and MUST STOP and ask me what to do.

Examples:
- // BAD: This uses Zod for validation instead of manual checking
- // BAD: Refactored from the old validation system
- // BAD: Wrapper around MCP tool protocol
- // GOOD: Executes tools with validated arguments

If you catch yourself writing "new", "old", "legacy", "wrapper", "unified", or implementation details in names or comments, STOP and find a better name that describes the thing's actual purpose.

## Project-Specific Code Style

- Generated code should strictly follow best practices.
- Always add type annotation to variables and function signatures.
- Always check the code and tests with ruff and mypy before running code coverage.
- Always check the code and tests with ruff and mypy before running execution.
- Run mypy before ruff.
- Prefer running `uv run ruff check --fix` over manual editing the files to fix linting issues.
- Always write tests for new code and keep overall 100% code coverage.
- No backward compatibility is needed when refactoring code unless specified by users.
- Think about circular dependency problem and avoid it in planning.
- The python docstring is in Google format.
- Always have detailed docstrings for functions.
- Always validate the input parameters first in each function implementation.
- Before finishing:
  - Always check the source and tests code base with ruff and mypy.
  - Keep README.md in sync with the changes.
  - Update docstrings of the modified methods to reflect the changes.

## Git Commit Standards

- Write clean, professional commit messages focusing on technical content
- Use clear, descriptive commit messages explaining the "why" not just the "what"

## Common Methods and Utilities

- **Use standard Python logging**: `import logging; logger = logging.getLogger(__name__)`
