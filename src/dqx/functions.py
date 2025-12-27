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
    Determine whether two floating-point numbers are equal within a specified tolerance.

    Parameters:
        a (float): First value to compare.
        b (float): Second value to compare.
        tol (float): Maximum allowed absolute difference for equality.

    Returns:
        True if the absolute difference between `a` and `b` is less than `tol`, False otherwise.
    """

    return abs(a - b) < tol


def is_neq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Determine whether two floating-point values differ by at least the given tolerance.

    Parameters:
        tol (float): Comparison tolerance; two values are considered different if their absolute difference is greater than or equal to this value.

    Returns:
        bool: True if the absolute difference between a and b is greater than or equal to tol, False otherwise.
    """
    return abs(a - b) >= tol


def within_tol(a: float, b: float, rel_tol: float | None = None, abs_tol: float | None = None) -> bool:
    """
    Determine whether two floats differ by less than a specified tolerance.

    Only one of `rel_tol` or `abs_tol` must be provided. If `abs_tol` is given, the function tests whether |a - b| < abs_tol. If `rel_tol` is given, the function tests whether |(a - b) / b| < rel_tol.

    Parameters:
        a (float): First value to compare.
        b (float): Second value to compare (used as the denominator for relative tolerance).
        rel_tol (float | None): Relative tolerance; comparison uses |(a - b) / b| < rel_tol.
        abs_tol (float | None): Absolute tolerance; comparison uses |a - b| < abs_tol.

    Returns:
        bool: `true` if the difference between `a` and `b` is within the specified tolerance, `false` otherwise.

    Raises:
        DQXError: If neither `rel_tol` nor `abs_tol` is provided, or if both are provided.
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
    Determine whether a value lies within the closed interval [lower, upper], using a tolerance for comparisons.

    Parameters:
        a (float): Value to test.
        lower (float): Lower bound of the interval.
        upper (float): Upper bound of the interval.
        tol (float): Tolerance applied to both bound comparisons.

    Returns:
        bool: `true` if `lower <= a <= upper` within `tol`, `false` otherwise.
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
        """
        Determine the coalesced value from concrete SymPy arguments or signal that evaluation should be deferred.

        Parameters:
            *args (Any): Candidate values to coalesce; evaluated only when all arguments are concrete (have no free symbols).

        Returns:
            sp.Expr | None: The first argument that is not `None` and not `NaN` when all arguments are concrete; `sp.S.NaN` if every argument is `None` or `NaN`; `None` to indicate the expression must remain unevaluated because at least one argument contains free symbols.

        Raises:
            DQXError: If called with no arguments.
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
    Selects the first argument that is neither None nor NaN.

    Parameters:
        *args (Any): Values to evaluate in order of preference.

    Returns:
        sp.Expr: A sympy expression equal to the first argument that is not None and not NaN, or NaN if no such argument exists.
    """
    return Coalesce(*args)
