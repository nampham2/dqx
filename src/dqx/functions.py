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
