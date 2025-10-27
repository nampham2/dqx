#!/usr/bin/env python3
"""Wrapper for commitizen to work with pre-commit commit-msg hook."""

import os
import subprocess
import sys


def main() -> int:
    """Check commit message using commitizen."""
    # Pre-commit provides the commit message file in different ways:
    # 1. Via PRE_COMMIT_COMMIT_MSG_FILENAME environment variable
    # 2. As the first command line argument
    # 3. Default to .git/COMMIT_EDITMSG

    commit_msg_file = (
        os.environ.get("PRE_COMMIT_COMMIT_MSG_FILENAME")
        or (sys.argv[1] if len(sys.argv) > 1 else None)
        or ".git/COMMIT_EDITMSG"
    )

    if not os.path.exists(commit_msg_file):
        print(f"Error: Commit message file not found at: {commit_msg_file}")
        return 1

    # Run commitizen check
    result = subprocess.run(
        ["uv", "run", "cz", "check", "--commit-msg-file", commit_msg_file],
        capture_output=False,
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
