"""Utility functions for the DQX package."""

from __future__ import annotations

import random
import string


def random_prefix(k: int = 6) -> str:
    """
    Generate a random table name consisting of lowercase ASCII letters.

    Args:
        k (int): The length of the random string to generate. Default is 6.

    Returns:
        str: A string starting with an underscore followed by a random sequence of lowercase ASCII letters.
    """
    return "_" + "".join(random.choices(string.ascii_lowercase, k=k))
