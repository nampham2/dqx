#!/usr/bin/env python3
"""Script to fix all MetricProvider and Evaluator constructor calls in tests."""

import re
from pathlib import Path


def fix_metric_provider_calls(content):
    """Fix MetricProvider constructor calls to include data_av_threshold."""
    # Pattern to match MetricProvider calls without data_av_threshold
    pattern = r"MetricProvider\(([^)]+)\)(?!\s*#.*data_av_threshold)"

    def replace_func(match):
        args = match.group(1).strip()
        # Check if it already has data_av_threshold
        if "data_av_threshold" in args:
            return match.group(0)
        # Add data_av_threshold
        return f"MetricProvider({args}, data_av_threshold=0.8)"

    return re.sub(pattern, replace_func, content)


def fix_evaluator_calls(content):
    """Fix Evaluator constructor calls to include data_av_threshold."""
    # Pattern to match Evaluator calls with only 3 arguments
    pattern = r'Evaluator\(([^,]+),\s*([^,]+),\s*"([^"]+)"\)'

    def replace_func(match):
        provider = match.group(1).strip()
        key = match.group(2).strip()
        suite_name = match.group(3)
        return f'Evaluator({provider}, {key}, "{suite_name}", data_av_threshold=0.8)'

    return re.sub(pattern, replace_func, content)


def process_file(file_path):
    """Process a single test file."""
    try:
        content = file_path.read_text()
        original_content = content

        # Fix MetricProvider calls
        content = fix_metric_provider_calls(content)

        # Fix Evaluator calls
        content = fix_evaluator_calls(content)

        # Write back if changed
        if content != original_content:
            file_path.write_text(content)
            print(f"Fixed: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process all test files."""
    tests_dir = Path("tests")
    fixed_count = 0

    # Process all Python test files
    for test_file in tests_dir.glob("test_*.py"):
        if process_file(test_file):
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
