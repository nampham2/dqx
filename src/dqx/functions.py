from typing import Any

import sympy as sp

from dqx.common import DQXError

EPSILON = 1e-9


def is_geq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if the value `a` is greater than or equal to the value `b` within a small tolerance.

    Args:
        a (float): The first value to compare.
        b (float): The second value to compare.
    Returns:
        bool: True if `a` is greater than or equal to `b` within a small tolerance, False otherwise.
    """
    return a > b - tol


def is_leq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if the difference between two floating-point numbers is less than a small epsilon value.

    Args:
        a (float): The first floating-point number.
        b (float): The second floating-point number.
    Returns:
        bool: True if the difference (a - b) is less than EPSILON, False otherwise.
    """

    return a < b + tol


def is_gt(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Compare two floating-point numbers to determine if the first is greater than the second,
    considering a small epsilon value to account for floating-point precision errors.

    Args:
        a (float): The first floating-point number.
        b (float): The second floating-point number.
    Returns:
        bool: True if 'a' is greater than 'b' by more than EPSILON, False otherwise.
    """

    return a > b + tol


def is_lt(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if the value `a` is less than the value `b` by a margin greater than a small epsilon value.
    Args:
        a (float): The first value to compare.
        b (float): The second value to compare.
    Returns:
        bool: True if `a` is less than `b` by more than a small epsilon value, False otherwise.
    """

    return a < b - tol


def is_eq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if two floating-point numbers are approximately equal.
    Args:
        a (float): The first floating-point number.
        b (float): The second floating-point number.
    Returns:
        bool: True if the absolute difference between `a` and `b` is less than EPSILON, False otherwise.
    """

    return abs(a - b) < tol


def is_neq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if two floating-point numbers are not equal (outside tolerance).
    Args:
        a (float): The first floating-point number.
        b (float): The second floating-point number.
        tol (float): Tolerance for comparison.
    Returns:
        bool: True if the absolute difference between `a` and `b` is >= tolerance, False otherwise.
    """
    return abs(a - b) >= tol


def within_tol(a: float, b: float, rel_tol: float | None = None, abs_tol: float | None = None) -> bool:
    """
    Check if the absolute difference between two floating-point numbers is within a given tolerance.
    Args:
        a (float): The first floating-point number.
        b (float): The second floating-point number.
        rel_tol (float): The relative tolerance within which the two numbers are considered equal.
        abs_tol (float): The absolute tolerance within which the two numbers are considered equal.
    Returns:
        bool: True if the absolute difference between `a` and `b` is less than `tol`, False otherwise.
    """

    if rel_tol is None and abs_tol is None:
        raise DQXError("Either relative tolerance or absolute tolerance must be provided!")

    if rel_tol and abs_tol:
        raise DQXError("Both relative tolerance and absolute tolerance cannot be provided simultaneously!")

    if abs_tol:
        return abs(a - b) < abs_tol

    assert rel_tol is not None  # Type hinting
    return abs((a - b) / b) < rel_tol


def is_zero(a: float, tol: float = EPSILON) -> bool:
    """
    Check if a given floating-point number is effectively zero.
    This function compares the absolute value of the input number to a small
    threshold value (EPSILON) to determine if it is close enough to zero to be
    considered zero.
    Args:
        a (float): The floating-point number to check.
    Returns:
        bool: True if the number is effectively zero, False otherwise.
    """

    return abs(a) < tol


def is_positive(a: float, tol: float = EPSILON) -> bool:
    """
    Check if a given floating-point number is positive.
    Args:
        a (float): The floating-point number to check.
    Returns:
        bool: True if the number is greater than EPSILON, False otherwise.
    """

    return a > tol


def is_negative(a: float, tol: float = EPSILON) -> bool:
    """
    Check if a given floating-point number is considered negative.
    Args:
        a (float): The floating-point number to check.
    Returns:
        bool: True if the number is less than -EPSILON, False otherwise.
    """

    return a < -tol


def is_between(a: float, lower: float, upper: float, tol: float = EPSILON) -> bool:
    """
    Check if a value is between two bounds (inclusive).

    Args:
        a: The value to check.
        lower: The lower bound.
        upper: The upper bound.
        tol: Tolerance for floating-point comparisons (applies to both bounds).

    Returns:
        bool: True if lower ≤ a ≤ upper (within tolerance), False otherwise.
    """
    return is_geq(a, lower, tol) and is_leq(a, upper, tol)


class Coalesce(sp.Function):
    """
    Sympy function that returns the first non-None/non-NaN value from its arguments.

    Similar to SQL's COALESCE function. When evaluated, returns the first
    argument that is not None or NaN. If all arguments are None/NaN, returns NaN.

    Example:
        coalesce(average(price), 0)  # Returns average(price) if not None/NaN, else 0
        coalesce(x, y, 0)            # Returns first non-None/NaN of x, y, or 0
    """

    @classmethod
    def eval(cls, *args: Any) -> sp.Expr | None:
        """Evaluate coalesce during sympy simplification.

        Returns the first non-None argument if all arguments are concrete values.
        Returns None (unevaluated) if any argument contains free symbols.
        """
        if not args:
            raise DQXError("coalesce requires at least one argument")

        # If any argument has free symbols, we can't evaluate yet
        for arg in args:
            if hasattr(arg, "free_symbols") and arg.free_symbols:
                return None  # Return None to keep unevaluated

        # All arguments are concrete - find first non-None
        for arg in args:
            # Convert sympy numbers to Python values for None check
            if arg is not None and arg != sp.S.NaN:
                return arg

        return sp.S.NaN  # pragma: no cover - All values were None/NaN


def coalesce(*args: Any) -> sp.Expr:
    """
    Return the first non-None/non-NaN value from the arguments.

    This is a convenience wrapper around the Coalesce sympy function.
    If all arguments are None/NaN, returns NaN.

    Args:
        *args: Values to check, in order of preference.

    Returns:
        Sympy expression representing the coalesce operation.

    Example:
        coalesce(average(price), 0)  # Use 0 if average is None/NaN
    """
    return Coalesce(*args)
