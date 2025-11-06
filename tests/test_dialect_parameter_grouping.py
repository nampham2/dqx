"""Test parameter grouping in dialects."""

from typing import Type

import pytest

from dqx.dialect import BigQueryDialect, DuckDBDialect
from dqx.ops import Average, CustomSQL, SqlOp, Sum


@pytest.mark.parametrize("dialect_class", [DuckDBDialect, BigQueryDialect])
def test_dialect_groups_operations_by_parameters(dialect_class: Type[DuckDBDialect] | Type[BigQueryDialect]) -> None:
    """Dialects should group operations by parameters."""
    dialect = dialect_class()

    # Operations with different parameters
    ops: list[SqlOp[float]] = [
        Average("price", parameters={"region": "US"}),
        Sum("amount", parameters={"region": "US"}),
        Average("cost", parameters={"region": "EU"}),
        Sum("total", parameters={"region": "EU"}),
        CustomSQL("COUNT(*)"),
    ]

    groups = dialect._group_operations_by_parameters(ops)

    # Should have 3 groups: US, EU, and empty
    assert len(groups) == 3

    # Check US group
    us_key = (("region", "US"),)
    assert us_key in groups
    assert len(groups[us_key]) == 2
    assert all(op.parameters["region"] == "US" for op in groups[us_key])

    # Check EU group
    eu_key = (("region", "EU"),)
    assert eu_key in groups
    assert len(groups[eu_key]) == 2

    # Check empty group
    empty_key = ()
    assert empty_key in groups
    assert len(groups[empty_key]) == 1


def test_parameter_grouping_with_multiple_params() -> None:
    """Test grouping with multiple parameters."""
    dialect = DuckDBDialect()

    ops: list[SqlOp[float]] = [
        Average("x", parameters={"region": "US", "category": "A"}),
        Sum("y", parameters={"region": "US", "category": "A"}),
        Average("z", parameters={"category": "A", "region": "US"}),  # Same as first
    ]

    groups = dialect._group_operations_by_parameters(ops)

    # All should be in same group (parameters are sorted)
    assert len(groups) == 1
    key = (("category", "A"), ("region", "US"))
    assert key in groups
    assert len(groups[key]) == 3
