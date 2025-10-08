# Project specific rules
Please also use the rules in new_rules.md and follow them strictly.

## Virtual environment
  - The project is managed with uv
  - Use `uv run` to run commands (python, pytest, ruff, mypy ...) in the virtual environment
  - The shell is zsh and not bash. Please generate proper bash scripts.

## Code Style and best practices
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

## Unit tests
  - Prefer native objects over mocks.
  - Make sure that all tests are isolated.

## Design patterns and algorithms
  - Employ design patterns when planning and act.
  Provide different solutions with different trade off and ask users to make decision on which solution to choose.
  - Thinks about algorithm complexity and communicate clearly to users.
  - Prefer simple, elegant solution over more complex ones.
  - Always analyze code for potential duplicates and give proposal to remove them.

## Examples
  - To run code coverage: `uv run pytest tests/test_symbol_table.py -v --cov=dqx.symbol_table`
  - To run ruff and mypy for the whole project: `.github/quality.sh && .github/coverage.sh`
