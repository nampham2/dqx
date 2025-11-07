"""Tests for CustomSQL operation."""

import pytest

from dqx.ops import CustomSQL


def test_custom_sql_basic() -> None:
    """Test basic CustomSQL operation."""
    op = CustomSQL("CAST(COUNT(*) AS DOUBLE)")

    # Name uses hash-based identifier
    assert op.name.startswith("custom_sql(")
    assert op.name.endswith(")")
    assert op.sql_expression == "CAST(COUNT(*) AS DOUBLE)"
    assert op.parameters == {}

    # Test sql_col
    assert op.sql_col.endswith(op.name)


def test_custom_sql_with_parameters() -> None:
    """Test CustomSQL with parameters."""
    params = {"category": "electronics", "min_value": "100"}
    # Note: Parameters are for CTE, not substituted in SQL
    template = "CAST(SUM(CASE WHEN category = 'electronics' AND value > 100 THEN 1 ELSE 0 END) AS DOUBLE)"

    op = CustomSQL(template, params)

    assert op.name.startswith("custom_sql(")
    assert op.name.endswith(")")
    assert op.sql_expression == template
    assert op.parameters == params


def test_custom_sql_equality() -> None:
    """Test CustomSQL equality."""
    template1 = "CAST(COUNT(*) AS DOUBLE)"
    template2 = "CAST(SUM(value) AS DOUBLE)"
    params1 = {"cat": "A"}
    params2 = {"cat": "B"}

    op1 = CustomSQL(template1)
    op2 = CustomSQL(template1)
    op3 = CustomSQL(template2)
    op4 = CustomSQL(template1, params1)
    op5 = CustomSQL(template1, params1)
    op6 = CustomSQL(template1, params2)

    # Same template, no params
    assert op1 == op2
    assert hash(op1) == hash(op2)

    # Different template
    assert op1 != op3
    assert hash(op1) != hash(op3)

    # With vs without params
    assert op1 != op4

    # Same template and params
    assert op4 == op5
    assert hash(op4) == hash(op5)

    # Same template, different params
    assert op4 != op6
    assert hash(op4) != hash(op6)


def test_custom_sql_repr() -> None:
    """Test CustomSQL string representation."""
    op = CustomSQL("CAST(AVG(score) AS DOUBLE)")

    assert repr(op) == "CustomSQL('CAST(AVG(score) AS DOUBLE)')"
    assert str(op) == "CustomSQL('CAST(AVG(score) AS DOUBLE)')"


def test_custom_sql_value_operations() -> None:
    """Test CustomSQL value operations (inherited from OpValueMixin)."""
    from dqx.common import DQXError

    op = CustomSQL("CAST(COUNT(*) AS DOUBLE)")

    # Initially no value
    with pytest.raises(DQXError, match="CustomSQL op has not been collected yet"):
        op.value()

    # Assign value
    op.assign(42.0)
    assert op.value() == 42.0

    # Clear value
    op.clear()
    with pytest.raises(DQXError):
        op.value()


def test_custom_sql_complex_expression() -> None:
    """Test CustomSQL with complex SQL expression."""
    # Complex SQL without templating
    sql_expr = """CAST(
        SUM(
            CASE
                WHEN status = 'active' AND region IN ('US', 'EU')
                THEN amount * 1.5
                ELSE 0
            END
        ) AS DOUBLE
    )"""

    # Parameters would be used for CTE filtering, not SQL substitution
    params = {"year": "2024", "quarter": "Q1"}

    op = CustomSQL(sql_expr, params)

    assert op.name.startswith("custom_sql(")
    assert op.name.endswith(")")
    assert op.sql_expression == sql_expr
    assert op.parameters == params


def test_custom_sql_various_expressions() -> None:
    """Test CustomSQL with various SQL expressions."""
    test_cases = [
        "COUNT(DISTINCT user_id)",
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)",
        "SUM(amount) FILTER (WHERE status = 'active')",
        "COUNT(*) - COUNT(DISTINCT user_id)",
    ]

    for sql_expr in test_cases:
        op = CustomSQL(sql_expr)
        assert op.sql_expression == sql_expr
        assert op.name.startswith("custom_sql(")
        assert op.name.endswith(")")
        assert len(op.name) == len("custom_sql()") + 8  # 8 char hash inside parens
