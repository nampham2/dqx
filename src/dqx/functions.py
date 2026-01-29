from typing import Any

import sympy as sp

from dqx.common import DQXError

EPSILON = 1e-9

# Error messages for TRY003 compliance
COALESCE_ARG_ERROR = "coalesce requires at least one argument"


def is_geq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if the value `a` is greater than or equal to the value `b` within a small tolerance.

    Args:
        a: The first value to compare.
        b: The second value to compare.

    Returns:
        bool: `True` if `a` is greater than or equal to `b` within tolerance, `False` otherwise.
    """
    return a > b - tol


def is_leq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if the difference between two floating-point numbers is less than a small epsilon value.

    Args:
        a: The first floating-point number.
        b: The second floating-point number.

    Returns:
        bool: `True` if `a <= b` within tolerance, `False` otherwise.
    """

    return a < b + tol


def is_gt(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Compare two floating-point numbers to determine if the first is greater than the second,
    considering a small epsilon value to account for floating-point precision errors.

    Args:
        a: The first floating-point number.
        b: The second floating-point number.

    Returns:
        bool: `True` if `a > b` by more than tolerance, `False` otherwise.
    """

    return a > b + tol


def is_lt(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Check if the value `a` is less than the value `b` by a margin greater than a small epsilon value.
    Args:
        a: The first value to compare.
        b: The second value to compare.

    Returns:
        bool: `True` if `a < b` by more than tolerance, `False` otherwise.
    """

    return a < b - tol


def is_eq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Determine whether two floating-point numbers are equal within a specified tolerance.

    Args:
        a: First value to compare.
        b: Second value to compare.
        tol: Maximum allowed absolute difference for equality.

    Returns:
        bool: `True` if `abs(a - b) < tol`, `False` otherwise.
    """

    return abs(a - b) < tol


def is_neq(a: float, b: float, tol: float = EPSILON) -> bool:
    """
    Determine whether two floating-point values differ by at least the given tolerance.

    Args:
        a: The first value to compare.
        b: The second value to compare.
        tol: Comparison tolerance; defaults to EPSILON.

    Returns:
        bool: `True` if `abs(a - b) >= tol`, `False` otherwise.
    """
    return abs(a - b) >= tol


def within_tol(a: float, b: float, rel_tol: float | None = None, abs_tol: float | None = None) -> bool:
    """
    Determine whether two floats differ by less than a specified tolerance.

    Only one of `rel_tol` or `abs_tol` must be provided. If `abs_tol` is given, the function tests whether |a - b| < abs_tol. If `rel_tol` is given, the function tests whether |(a - b) / b| < rel_tol.

    Args:
        a: First value to compare.
        b: Second value to compare (used as the denominator for relative tolerance).
        rel_tol: Relative tolerance; comparison uses |(a - b) / b| < rel_tol.
        abs_tol: Absolute tolerance; comparison uses |a - b| < abs_tol.

    Returns:
        bool: `True` if the difference between `a` and `b` is within the specified tolerance, `False` otherwise.

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
        a: The floating-point number to check.

    Returns:
        bool: `True` if the number is effectively zero, `False` otherwise.
    """

    return abs(a) < tol


def is_positive(a: float, tol: float = EPSILON) -> bool:
    """
    Check if a given floating-point number is positive.

    Args:
        a: The floating-point number to check.

    Returns:
        bool: `True` if the number is greater than tolerance, `False` otherwise.
    """

    return a > tol


def is_negative(a: float, tol: float = EPSILON) -> bool:
    """
    Check if a given floating-point number is considered negative.

    Args:
        a: The floating-point number to check.

    Returns:
        bool: `True` if the number is less than negative tolerance, `False` otherwise.
    """

    return a < -tol


def is_between(a: float, lower: float, upper: float, tol: float = EPSILON) -> bool:
    """
    Determine whether a value lies within the closed interval [lower, upper], using a tolerance for comparisons.

    Args:
        a: Value to test.
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        tol: Tolerance applied to both bound comparisons.

    Returns:
        bool: `True` if `lower <= a <= upper` within `tol`, `False` otherwise.
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

        Args:
            *args: Candidate values to coalesce; evaluated only when all arguments are concrete (have no free symbols).

        Returns:
            sp.Expr | None: The first argument that is not `None` and not `NaN` when all arguments are concrete; `sp.S.NaN` if every argument is `None` or `NaN`; `None` to indicate the expression must remain unevaluated because at least one argument contains free symbols.

        Raises:
            DQXError: If called with no arguments.
        """
        if not args:
            raise DQXError(COALESCE_ARG_ERROR)

        # If any argument has free symbols, we can't evaluate yet
        for arg in args:
            if hasattr(arg, "free_symbols") and arg.free_symbols:
                return None  # Return None to keep unevaluated

        # All arguments are concrete - find first non-None
        for arg in args:
            # Convert sympy numbers to Python values for None check
            if arg is not None and arg != sp.S.NaN:
                return arg

        return sp.S.NaN  # All values were None/NaN


def coalesce(*args: Any) -> sp.Expr:
    """
    Selects the first argument that is neither None nor NaN.

    Args:
        *args: Values to evaluate in order of preference.

    Returns:
        sp.Expr: A sympy expression equal to the first argument that is not None and not NaN, or NaN if no such argument exists.
    """
    return Coalesce(*args)


def pct(value: float | int) -> float:
    """Convert percentage notation to decimal.

    This is a pure Python utility function for converting percentage values
    to their decimal equivalents. It evaluates immediately and returns a
    plain float (not a SymPy symbol).

    Usage is limited to thresholds (right-hand side of comparisons) and
    tunable values in the Python API. It is NOT available in DQL metric
    expressions.

    Args:
        value: Percentage value (e.g., 5 for 5%, 0.5 for 0.5%, 100 for 100%).
               Accepts any numeric value including negative, zero, >100, decimals.

    Returns:
        Decimal equivalent as a plain Python float.

    Examples:
        >>> pct(5)
        0.05
        >>> pct(0.5)
        0.005
        >>> pct(100)
        1.0
        >>> pct(150)
        1.5
        >>> pct(-10)
        -0.1

        # In assertions
        >>> from dqx.api import check, pct
        >>> @check(name="Nulls")
        ... def check_nulls(mp, ctx):
        ...     ctx.assert_that(mp.null_rate("col")).config(name="Null rate").is_leq(pct(5))

        # In tunables
        >>> from dqx.tunables import TunableFloat
        >>> THRESHOLD = TunableFloat("THRESHOLD", value=pct(5), bounds=(pct(0), pct(10)))
        >>> # Equivalent to: value=0.05, bounds=(0.0, 0.1)

    Note:
        Returns immediately evaluated float, NOT a SymPy expression.
        Do not use in metric definitions - only for thresholds and tunable values.
    """
    return float(value) / 100.0
