#!/usr/bin/env python3
"""
AST-based script to update Metric.build() calls to include dataset parameter.

This script will:
1. Find all calls to models.Metric.build() or Metric.build()
2. Add dataset parameter based on context
3. Preserve code formatting as much as possible
"""

import ast
import sys
from pathlib import Path


class MetricBuildUpdater(ast.NodeTransformer):
    """AST transformer to update Metric.build() calls."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.modified = False

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Visit function calls and update Metric.build() calls."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Check if this is a Metric.build() call
        if self._is_metric_build_call(node):
            self._update_metric_build_call(node)
            self.modified = True

        return node

    def _is_metric_build_call(self, node: ast.Call) -> bool:
        """Check if a call node is a Metric.build() call."""
        if isinstance(node.func, ast.Attribute):
            # Check for models.Metric.build or Metric.build
            if node.func.attr == "build":
                if isinstance(node.func.value, ast.Attribute):
                    # models.Metric.build
                    return (
                        node.func.value.attr == "Metric"
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == "models"
                    )
                elif isinstance(node.func.value, ast.Name):
                    # Metric.build
                    return node.func.value.id == "Metric"
        return False

    def _update_metric_build_call(self, node: ast.Call) -> None:
        """Update a Metric.build() call to include dataset parameter."""
        # Count positional arguments
        num_positional = len(node.args)

        # Check if dataset is already provided as keyword
        has_dataset_kwarg = any(kw.arg == "dataset" for kw in node.keywords)

        if has_dataset_kwarg:
            # Already has dataset, nothing to do
            return

        # Determine default dataset value based on file context
        dataset_value = self._determine_dataset_value()

        if num_positional == 2:
            # Has metric and key as positional, add dataset as third
            node.args.append(ast.Constant(value=dataset_value))
        elif num_positional >= 3:
            # Already has 3+ positional args, dataset might be there
            # or it might be state. Check if we need to add as kwarg
            if num_positional == 3:
                # Might be (metric, key, state) - add dataset as kwarg
                node.keywords.append(ast.keyword(arg="dataset", value=ast.Constant(value=dataset_value)))
        else:
            # Less than 2 positional args, add as keyword
            node.keywords.append(ast.keyword(arg="dataset", value=ast.Constant(value=dataset_value)))

    def _determine_dataset_value(self) -> str:
        """Determine appropriate dataset value based on file context."""
        # Use file path to determine context
        path_str = str(self.file_path)

        # Common test dataset names based on file patterns
        if "test_analyzer" in path_str:
            return "test_data"
        elif "test_provider" in path_str:
            return "test_dataset"
        elif "test_" in path_str:
            return "test_ds"
        elif "orm" in path_str and "test_" in path_str:
            return "test_dataset"
        else:
            # Default for unknown contexts
            return "dataset"


def update_file(file_path: Path) -> bool:
    """Update a single Python file."""
    try:
        with open(file_path, "r") as f:
            source = f.read()

        # Parse the source code
        tree = ast.parse(source)

        # Transform the AST
        updater = MetricBuildUpdater(file_path)
        new_tree = updater.visit(tree)

        if updater.modified:
            # Convert back to source code
            import astor

            new_source = astor.to_source(new_tree)

            # Write back to file
            with open(file_path, "w") as f:
                f.write(new_source)

            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    # Check if astor is available
    try:
        import astor  # noqa: F401
    except ImportError:
        print("Error: astor package is required. Install with: uv pip install astor")
        sys.exit(1)

    # Define test directories to search
    test_dirs = [
        Path("tests/test_analyzer.py"),
        Path("tests/test_provider.py"),
        Path("tests/orm/test_repositories.py"),
    ]

    # Process each file
    updated_files = []
    for test_file in test_dirs:
        if test_file.exists():
            print(f"Processing {test_file}...")
            if update_file(test_file):
                updated_files.append(test_file)
                print(f"  ✓ Updated {test_file}")
            else:
                print(f"  - No changes needed in {test_file}")
        else:
            print(f"  ⚠ File not found: {test_file}")

    # Summary
    print(f"\nSummary: Updated {len(updated_files)} file(s)")
    if updated_files:
        print("Updated files:")
        for f in updated_files:
            print(f"  - {f}")

    # Note about manual review
    if updated_files:
        print("\nIMPORTANT: Please review the changes and run tests to ensure correctness.")
        print("The script uses heuristics to determine dataset values.")
        print("You may need to adjust some values based on actual test context.")


if __name__ == "__main__":
    main()
