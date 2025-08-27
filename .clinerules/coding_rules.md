# Project specific rules

## Virtual environment
  - The project is managed with uv
  - Use `uv run` to run commands (python, pytest, ruff, mypy ...) in the virtual environment
  - The shell is zsh and not bash. Please generate proper bash scripts.

## Code Style and best practices
  - Generated code should strictly follow best practices.
  - Always check the code and tests with ruff and mypy before running code coverage.
  - Always check the code and tests with ruff and mypy before running execution.
  - Run mypy before ruff.
  - Always write tests for new code and keep overall 100% code coverage.
  - No backward compatibility is needed when refactoring code unless specified by users.
  - Think about circular dependency problem and avoid it in planning.
  - The python docstring is in Google format.
  - Before finishing:
    - Always check the source and tests code base with ruff and mypy.
    - Keep README.md in sync with the changes.

## Design patterns and algorithms
  - Employ design patterns when planning and act.
  Provide different solutions with different trade off and ask users to make decision on which solution to choose.
  - Thinks about algorithm complexity and communicate clearly to users.
  - Prefer simple, elegant solution over more complex ones.
