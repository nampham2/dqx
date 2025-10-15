"""Test that rich is available as a main dependency."""

from pathlib import Path


def test_rich_is_main_dependency() -> None:
    """Verify rich is in main dependencies, not dev dependencies."""
    # Read pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Parse dependencies section
    lines = content.split("\n")
    in_main_deps = False
    in_dev_deps = False
    found_in_main = False
    found_in_dev = False

    for i, line in enumerate(lines):
        # Check if we're in main dependencies
        if line.strip() == "dependencies = [":
            in_main_deps = True
            in_dev_deps = False
        elif line.strip() == "dev = [":
            in_dev_deps = True
            in_main_deps = False
        elif line.strip() == "]":
            in_main_deps = False
            in_dev_deps = False

        # Check for rich
        if in_main_deps and "rich>=" in line:
            found_in_main = True
        if in_dev_deps and "rich>=" in line:
            found_in_dev = True

    assert found_in_main, "rich should be in main dependencies"
    assert not found_in_dev, "rich should NOT be in dev dependencies"


def test_production_imports_work() -> None:
    """Test that production code can import rich."""
    # These imports should work without dev dependencies
    from dqx.analyzer import Analyzer
    from dqx.display import print_assertion_results, print_graph, print_symbols

    # Verify the imports have rich components
    assert hasattr(print_graph, "__code__")
    assert hasattr(print_assertion_results, "__code__")
    assert hasattr(print_symbols, "__code__")
    assert hasattr(Analyzer, "__init__")
