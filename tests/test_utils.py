"""Tests for utility functions."""
from __future__ import annotations

import re
import string

from dqx.utils import random_prefix


def test_random_prefix_default_length() -> None:
    """Test that default length is 6 characters plus underscore prefix."""
    result = random_prefix()
    assert len(result) == 7  # 1 underscore + 6 random chars
    assert result.startswith("_")


def test_random_prefix_custom_length() -> None:
    """Test custom length parameter."""
    for k in [1, 3, 10, 15]:
        result = random_prefix(k)
        assert len(result) == k + 1  # k chars + 1 underscore prefix
        assert result.startswith("_")


def test_random_prefix_zero_length() -> None:
    """Test zero length parameter."""
    result = random_prefix(0)
    assert result == "_"


def test_random_prefix_contains_only_lowercase_letters() -> None:
    """Test that result contains only underscore and lowercase ASCII letters."""
    result = random_prefix(20)
    assert result[0] == "_"
    
    # Check that all characters after underscore are lowercase ASCII letters
    remaining_chars = result[1:]
    assert all(c in string.ascii_lowercase for c in remaining_chars)


def test_random_prefix_randomness() -> None:
    """Test that function produces different results on multiple calls."""
    # Generate multiple results and check they're not all the same
    results = [random_prefix() for _ in range(10)]
    
    # It's extremely unlikely (but theoretically possible) that all 10 results are identical
    # With 6 characters from a 26-character alphabet, probability is (1/26)^60 which is negligible
    unique_results = set(results)
    assert len(unique_results) > 1, "Expected at least some variation in random results"


def test_random_prefix_pattern_matching() -> None:
    """Test that result matches expected pattern."""
    result = random_prefix()
    pattern = re.compile(r"^_[a-z]{6}$")
    assert pattern.match(result), f"Result '{result}' doesn't match expected pattern"

    # Test with different lengths
    for k in [1, 5, 10]:
        result = random_prefix(k)
        pattern = re.compile(f"^_[a-z]{{{k}}}$")
        assert pattern.match(result), f"Result '{result}' doesn't match expected pattern for k={k}"
