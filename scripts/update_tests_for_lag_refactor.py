#!/usr/bin/env python3
"""Script to update all test files for the lag refactor.

This script automatically updates test files to replace the old key=ResultKeyProvider()
API with the new lag=N parameter API.

The transformation rules:
1. mp.average("col", key=ctx.key()) -> mp.average("col", lag=0)
2. mp.average("col", key=ctx.key().lag(1)) -> mp.average("col", lag=1)
3. mp.ext.day_over_day(metric, key=ctx.key()) -> mp.ext.day_over_day(metric, lag=0)
4. mp.ext.week_over_week(metric, key=ctx.key().lag(2)) -> mp.ext.week_over_week(metric, lag=2)

This is a breaking change with no backward compatibility.
"""

import argparse
import re
from pathlib import Path


def extract_lag_from_key(key_expr: str) -> int:
    """Extract lag value from key expression.

    Args:
        key_expr: Expression like 'ctx.key()', 'ctx.key.lag(3)', or 'ctx.key().lag(3)'

    Returns:
        Lag value (0 if no lag() call found)
    """
    # Normalize the expression - handle both ctx.key.lag and ctx.key().lag
    key_expr = key_expr.strip()

    # Match .lag(N) pattern
    lag_match = re.search(r"\.lag\((\d+)\)", key_expr)
    if lag_match:
        return int(lag_match.group(1))
    return 0


def update_metric_calls(content: str) -> str:
    """Update metric provider method calls to use lag parameter.

    This handles:
    - Basic metrics: mp.average, mp.sum, mp.minimum, mp.maximum, etc.
    - Extended metrics: mp.ext.day_over_day, mp.ext.week_over_week, mp.ext.stddev
    """
    # Pattern for basic metric methods with key parameter
    # Matches: mp.method("col", key=...)
    # Using a more specific pattern that handles ctx.key.lag vs ctx.key().lag
    basic_pattern = re.compile(
        r"(mp\.(?:average|minimum|maximum|sum|null_count|variance|"
        r"duplicate_count|count_values|num_rows|first|metric))"
        r"\(([^,]+)(?:,\s*key\s*=\s*((?:ctx\.key(?:\(\))?(?:\.lag\(\d+\))?|[^)]+)))\)"
    )

    def replace_basic(match):
        method = match.group(1)
        args = match.group(2)
        key_expr = match.group(3)
        lag = extract_lag_from_key(key_expr)

        if lag == 0:
            # No lag needed, just remove key parameter
            return f"{method}({args})"
        else:
            return f"{method}({args}, lag={lag})"

    content = basic_pattern.sub(replace_basic, content)

    # Pattern for extended metrics
    # Matches: mp.ext.method(metric, key=...)
    extended_pattern = re.compile(
        r"(mp\.ext\.(?:day_over_day|week_over_week))"
        r"\((.*?),\s*key\s*=\s*([^)]+)\)"
    )

    def replace_extended(match):
        method = match.group(1)
        args = match.group(2)
        key_expr = match.group(3)
        lag = extract_lag_from_key(key_expr)

        if lag == 0:
            # No lag needed, just remove key parameter
            return f"{method}({args})"
        else:
            return f"{method}({args}, lag={lag})"

    content = extended_pattern.sub(replace_extended, content)

    # Special pattern for stddev which has additional parameters
    # Matches: mp.ext.stddev(metric, lag, n, key=...)
    stddev_pattern = re.compile(
        r"(mp\.ext\.stddev)"
        r"\((.*?),\s*(\d+),\s*(\d+),\s*key\s*=\s*([^)]+)\)"
    )

    def replace_stddev(match):
        method = match.group(1)
        metric = match.group(2)
        lag = match.group(3)
        n = match.group(4)
        # key parameter is removed for stddev
        return f"{method}({metric}, {lag}, {n})"

    content = stddev_pattern.sub(replace_stddev, content)

    # Remove imports of ResultKeyProvider if they become unused
    content = remove_unused_imports(content)

    return content


def remove_unused_imports(content: str) -> str:
    """Remove ResultKeyProvider import if it's no longer used."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        # Check if this is a ResultKeyProvider import
        if "ResultKeyProvider" in line and "from" in line and "import" in line:
            # Check if ResultKeyProvider is still used elsewhere in the file
            # (excluding this import line)
            other_content = "\n".join(ln for ln in lines if ln != line)
            if "ResultKeyProvider" not in other_content:
                # Skip this import line
                continue
        new_lines.append(line)

    return "\n".join(new_lines)


def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """Process a single test file.

    Args:
        file_path: Path to the test file
        dry_run: If True, don't write changes, just report what would change

    Returns:
        True if file was modified, False otherwise
    """
    try:
        content = file_path.read_text()
        original_content = content

        # Update the content
        content = update_metric_calls(content)

        # Check if anything changed
        if content != original_content:
            if dry_run:
                print(f"Would update: {file_path}")
                # Show a diff preview
                import difflib

                diff = difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=str(file_path),
                    tofile=str(file_path),
                    n=3,
                )
                print("".join(diff))
            else:
                file_path.write_text(content)
                print(f"Updated: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Update test files for lag refactor")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--path", type=Path, default=Path("tests"), help="Path to tests directory (default: tests)")
    args = parser.parse_args()

    # Find all Python test files
    test_files = list(args.path.glob("**/test_*.py"))

    print(f"Found {len(test_files)} test files")

    modified_count = 0
    for test_file in test_files:
        if process_file(test_file, dry_run=args.dry_run):
            modified_count += 1

    print(f"\nSummary: {'Would modify' if args.dry_run else 'Modified'} {modified_count} files")

    if args.dry_run and modified_count > 0:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
