"""Command-line script entry points for dqx development tools."""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_coverage() -> None:
    """Run pytest with coverage reporting."""
    # Get command line arguments
    args = sys.argv[1:]  # Skip the script name
    # Setup coverage directory
    cov_dir = Path(".cov")
    if cov_dir.exists():
        shutil.rmtree(cov_dir)
    cov_dir.mkdir()

    # Set environment variable for coverage file
    env = os.environ.copy()
    env["COVERAGE_FILE"] = ".cov/coverage_dqx"

    # Run pytest with coverage
    pytest_cmd = ["uv", "run", "pytest", "--cov-report=", "--cov=dqx"]

    # Add command line arguments or default to 'tests'
    if args:
        pytest_cmd.extend(args)
    else:
        pytest_cmd.append("tests")

    result = subprocess.run(pytest_cmd, env=env)
    if result.returncode != 0:
        print("\nPytest failed with exit code:", result.returncode)
        sys.exit(result.returncode)

    # Combine coverage files
    coverage_files = glob.glob(".cov/*")
    if coverage_files:
        coverage_combine = ["uv", "run", "-m", "coverage", "combine"] + coverage_files
        subprocess.run(coverage_combine, check=True)

    # Generate coverage report
    print("\nCoverage Report:")
    print("-" * 80)
    coverage_report = ["uv", "run", "-m", "coverage", "report", "-m", "--skip-covered"]
    subprocess.run(coverage_report, check=True)

    # Cleanup
    shutil.rmtree(cov_dir, ignore_errors=True)
    coverage_file = Path(".coverage")
    if coverage_file.exists():
        coverage_file.unlink()


def run_hooks() -> None:
    """Run pre-commit hooks with various options.

    This command runs pre-commit hooks on files with options for speed,
    auto-fixing, and commit message validation.
    """
    parser = argparse.ArgumentParser(
        description="Run pre-commit hooks manually with options",
        epilog="""Examples:
  uv run hooks                    # Run on staged files
  uv run hooks --all              # Run on all files in project
  uv run hooks --fix              # Run only auto-fixing hooks
  uv run hooks --fast             # Skip mypy for faster checks
  uv run hooks --check-commit     # Check last commit message
  uv run hooks src/dqx/api.py     # Run on specific file(s)

Note: This includes shellcheck for shell scripts (.sh, .bash files)""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("files", nargs="*", help="Specific files to check (default: staged files)")
    parser.add_argument("--all", action="store_true", help="Run on all files (including src, tests, examples)")
    parser.add_argument("--fast", action="store_true", help="Skip slow hooks (mypy)")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Only run hooks that auto-fix issues (ruff-format, ruff-check, trailing-whitespace, end-of-file-fixer)",
    )
    parser.add_argument(
        "--check-commit", action="store_true", help="Check if the last commit message follows conventional format"
    )

    args = parser.parse_args()

    # ANSI color codes
    colors = {"green": "\033[0;32m", "red": "\033[0;31m", "yellow": "\033[1;33m", "reset": "\033[0m"}

    def print_color(message: str, color: str = "reset") -> None:
        """Print message with color."""
        print(f"{colors[color]}{message}{colors['reset']}")

    # Handle --check-commit separately
    if args.check_commit:
        print("🔍 Checking last commit message...")
        result = subprocess.run(
            ["uv", "run", "cz", "check", "--rev-range", "HEAD~1..HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print_color("✓ Last commit message follows conventional format!", "green")
        else:
            print_color("✗ Last commit message does NOT follow conventional format!", "red")
            print_color("Tip: Use 'uv run cz commit' to create a proper commit message", "yellow")
            sys.exit(1)
        return

    print("🔍 Running pre-commit hooks...")

    # Setup environment and base command
    env = os.environ.copy()
    if args.fast:
        env["SKIP"] = "mypy"
        print_color("⚡ Skipping slow hooks: mypy", "yellow")

    base_cmd = ["uv", "run", "pre-commit", "run"]

    # Define hook execution function for DRY
    def run_hook_command(cmd: list[str]) -> int:
        """Run a pre-commit command and return exit code."""
        return subprocess.run(cmd, env=env).returncode

    exit_code = 0

    if args.fix:
        # Run specific auto-fix hooks
        fix_hooks = ["ruff-format", "ruff-check", "trailing-whitespace", "end-of-file-fixer"]
        for hook in fix_hooks:
            print(f"Running {hook}...")
            cmd = base_cmd + [hook]
            if args.all:
                cmd.append("--all-files")
            elif args.files:
                cmd.extend(["--files"] + args.files)

            result = run_hook_command(cmd)
            if result != 0:
                exit_code = result
    else:
        # Run all hooks
        cmd = base_cmd.copy()

        if args.all:
            cmd.append("--all-files")
            print("Checking all files...")
        elif args.files:
            cmd.extend(["--files"] + args.files)
            print(f"Checking files: {' '.join(args.files)}")
        else:
            print("Checking staged files...")

        exit_code = run_hook_command(cmd)

    # Final status
    if exit_code == 0:
        print_color("✓ All hooks passed!", "green")
    else:
        print_color("✗ Some hooks failed. Please fix the issues above.", "red")
        if not args.fix:
            print_color("Tip: Use 'uv run hooks --fix' to run only auto-fixing hooks", "yellow")
        sys.exit(1)
