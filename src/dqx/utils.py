"""Utility functions for the DQX package."""

from __future__ import annotations

import random
import string
from typing import Any


def random_prefix(k: int = 6) -> str:
    """
    Generate a random table name consisting of lowercase ASCII letters.

    Args:
        k (int): The length of the random string to generate. Default is 6.

    Returns:
        str: A string starting with an underscore followed by a random sequence of lowercase ASCII letters.
    """
    return "_" + "".join(random.choices(string.ascii_lowercase, k=k))


def freeze_for_hashing(value: Any) -> Any:
    """Convert any JSON-serializable value to a hashable equivalent.

    Recursively converts mappings to sorted tuples, sequences to tuples,
    and sets to sorted tuples, while preserving scalars unchanged.

    Args:
        value: Any JSON-serializable value to make hashable

    Returns:
        A hashable equivalent that maintains deterministic ordering

    Examples:
        >>> freeze_for_hashing({"b": 2, "a": 1})
        (('a', 1), ('b', 2))

        >>> freeze_for_hashing([1, 2, [3, 4]])
        (1, 2, (3, 4))

        >>> freeze_for_hashing({3, 1, 2})
        (1, 2, 3)
    """
    if isinstance(value, dict):
        # Sort by key to ensure consistent ordering
        return tuple(sorted((k, freeze_for_hashing(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        # For sets, we need to handle the fact that elements might not be comparable
        # Convert elements first, then try to sort if possible
        frozen_items = [freeze_for_hashing(v) for v in value]
        try:
            # Try to sort - this will work for homogeneous comparable types
            # Cast to Any to satisfy mypy
            return tuple(sorted(frozen_items, key=lambda x: (x,)))  # type: ignore[return-value]
        except TypeError:
            # If items aren't comparable, convert to strings for sorting
            # This ensures deterministic ordering even for mixed types
            return tuple(sorted(frozen_items, key=lambda x: (type(x).__name__, str(x))))
    if isinstance(value, (list, tuple)):
        # Preserve order for lists/tuples
        return tuple(freeze_for_hashing(v) for v in value)
    return value
