#!/usr/bin/env python3
"""Fix remaining test issues after datasource API refactoring."""

import re
from pathlib import Path


def fix_file(filepath: Path) -> bool:
    """Fix issues in a single file."""
    content = filepath.read_text()
    original_content = content

    # Fix 3-argument from_arrow calls
    # Pattern: DuckRelationDataSource.from_arrow(data, "name", "anything")
    pattern1 = r'DuckRelationDataSource\.from_arrow\(([^,]+),\s*"([^"]+)",\s*"[^"]+"\)'
    content = re.sub(pattern1, r'DuckRelationDataSource.from_arrow(\1, "\2")', content)

    # Fix cases where .name is accessed on strings in run() calls
    # This happens when tests pass strings instead of datasource objects
    # We need to be more careful here - only fix actual issues

    # Look for patterns like: for ds in datasources: ... ds.name
    # where datasources is a list of strings
    if "for ds in datasources:" in content and "ds.name" in content:
        # Check if this is in a context where datasources are strings
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "ds.name" in line and i > 0:
                # Look backwards to see if ds comes from a string list
                for j in range(max(0, i - 10), i):
                    if "datasources: list[str]" in lines[j] or 'datasources = ["' in lines[j]:
                        # Replace ds.name with ds
                        lines[i] = lines[i].replace("ds.name", "ds")
                        break
        content = "\n".join(lines)

    # Fix test-specific dataset name mismatches
    # Many tests create datasources with generic names but checks expect specific names

    # For test_extended_metric_recursive_imputation.py
    if "test_extended_metric_recursive_imputation.py" in str(filepath):
        # Fix test_extended_metric_recursive_dataset_imputation
        content = content.replace(
            'ds1 = DuckRelationDataSource.from_arrow(arrow1, "data")',
            'ds1 = DuckRelationDataSource.from_arrow(arrow1, "ds1")',
        )
        content = content.replace(
            'ds2 = DuckRelationDataSource.from_arrow(arrow2, "data")',
            'ds2 = DuckRelationDataSource.from_arrow(arrow2, "ds2")',
        )
        # Fix test_nested_extended_metrics
        content = content.replace(
            'ds = DuckRelationDataSource.from_arrow(arrow_table, "data")',
            'ds = DuckRelationDataSource.from_arrow(arrow_table, "prod")',
        )
        # Fix test_circular_dependency_handling - check expects "circuit"
        if "test_circular_dependency_handling" in content:
            content = re.sub(
                r'(def test_circular_dependency_handling.*?ds = DuckRelationDataSource\.from_arrow\(arrow_table, )"data"(\))',
                r'\1"circuit"\2',
                content,
                flags=re.DOTALL,
            )

    # For test_lag_date_handling.py
    if "test_lag_date_handling.py" in str(filepath):
        # Many checks expect "ds1"
        content = re.sub(
            r'DuckRelationDataSource\.from_arrow\(([^,]+), "data"\)',
            r'DuckRelationDataSource.from_arrow(\1, "ds1")',
            content,
        )

    # For test_symbol_ordering.py
    if "test_symbol_ordering.py" in str(filepath):
        # Checks expect "orders"
        content = content.replace(
            'ds = DuckRelationDataSource.from_arrow(table, "data")',
            'ds = DuckRelationDataSource.from_arrow(table, "orders")',
        )

    # For test_extended_metric_symbol_info.py
    if "test_extended_metric_symbol_info.py" in str(filepath):
        # Check expects "ds1"
        content = content.replace(
            'ds = DuckRelationDataSource.from_arrow(arrow_table, "data")',
            'ds = DuckRelationDataSource.from_arrow(arrow_table, "ds1")',
        )

    # For test_extended_metric_symbol_info_fix.py
    if "test_extended_metric_symbol_info_fix.py" in str(filepath):
        # Check expects "ds1"
        content = content.replace(
            'ds = DuckRelationDataSource.from_arrow(arrow_table, "data")',
            'ds = DuckRelationDataSource.from_arrow(arrow_table, "ds1")',
        )
        # For nested test, check expects "test"
        if "test_nested_lag_metrics" in content:
            content = re.sub(
                r'(def test_nested_lag_metrics.*?ds = DuckRelationDataSource\.from_arrow\(arrow_table, )"data"(\))',
                r'\1"test"\2',
                content,
                flags=re.DOTALL,
            )

    if content != original_content:
        filepath.write_text(content)
        return True
    return False


def main():
    """Fix all test files."""
    test_files = [
        "tests/test_api_coverage.py",
        "tests/test_duplicate_count_integration.py",
        "tests/test_assertion_result_collection.py",
        "tests/test_extended_metric_recursive_imputation.py",
        "tests/test_lag_date_handling.py",
        "tests/test_symbol_ordering.py",
        "tests/test_extended_metric_symbol_info.py",
        "tests/test_extended_metric_symbol_info_fix.py",
    ]

    fixed_count = 0
    for test_file in test_files:
        filepath = Path(test_file)
        if filepath.exists():
            if fix_file(filepath):
                print(f"Fixed: {filepath}")
                fixed_count += 1
            else:
                print(f"No changes needed: {filepath}")
        else:
            print(f"File not found: {filepath}")

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
