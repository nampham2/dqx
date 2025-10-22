"""Command-line script entry points for dqx development tools."""

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
