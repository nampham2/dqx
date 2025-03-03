import pytest
from dqx import ops
from dqx.common import DQXError
from dqx.ops import Op, SketchOp, SqlOp


def test_num_rows() -> None:
    op = ops.NumRows()
    assert isinstance(op, Op)
    assert isinstance(op, SqlOp)
    assert not isinstance(op, SketchOp)
    assert op.name == "num_rows()"

    op1 = ops.NumRows()
    assert op == op1

    with pytest.raises(DQXError, match="NumRows op has not been collected yet!"):
        op.value()

    op.assign(0.2)
    assert op.value() == pytest.approx(0.2)


def test_average() -> None:
    op = ops.Average("column")
    assert isinstance(op, Op)
    assert op.name == "average(column)"
    assert set([op, op]) == {op}


def test_minimum() -> None:
    op = ops.Minimum("column")
    assert isinstance(op, Op)
    assert op.name == "minimum(column)"


def test_maximum() -> None:
    op = ops.Maximum("column")
    assert isinstance(op, Op)
    assert isinstance(op, SqlOp)
    assert op.name == "maximum(column)"


def test_sum() -> None:
    op = ops.Sum("column")
    assert isinstance(op, Op)
    assert op.name == "sum(column)"


def test_first() -> None:
    op = ops.First("column")
    assert isinstance(op, Op)
    assert op.name == "first(column)"


def test_null_count() -> None:
    op = ops.NullCount("column")
    assert isinstance(op, Op)
    assert op.name == "null_count(column)"


def test_approx_cardinality() -> None:
    op = ops.ApproxCardinality("column")
    assert isinstance(op, Op)
    assert op.name == "approx_cardinality(column)"
    assert set([op, op]) == {op}


def test_approx_cardinality_not_equal() -> None:
    approx_cardinality_op1 = ops.ApproxCardinality("column1")
    approx_cardinality_op2 = ops.ApproxCardinality("column2")
    assert approx_cardinality_op1 != approx_cardinality_op2
    assert approx_cardinality_op1 != 1


def test_equality() -> None:
    num_rows_op1 = ops.NumRows()
    num_rows_op2 = ops.NumRows()
    average_op1 = ops.Average("column")
    average_op2 = ops.Average("column")
    minimum_op1 = ops.Minimum("column")
    minimum_op2 = ops.Minimum("column")
    maximum_op1 = ops.Maximum("column")
    maximum_op2 = ops.Maximum("column")
    sum_op1 = ops.Sum("column")
    sum_op2 = ops.Sum("column")
    first_op1 = ops.First("column")
    first_op2 = ops.First("column")
    null_count_op1 = ops.NullCount("column")
    null_count_op2 = ops.NullCount("column")
    negative_count_op1 = ops.NegativeCount("column")
    negative_count_op2 = ops.NegativeCount("column")
    approx_cardinality_op1 = ops.ApproxCardinality("column")
    approx_cardinality_op2 = ops.ApproxCardinality("column")

    assert num_rows_op1 == num_rows_op2
    assert average_op1 == average_op2
    assert minimum_op1 == minimum_op2
    assert maximum_op1 == maximum_op2
    assert sum_op1 == sum_op2
    assert first_op1 == first_op2
    assert null_count_op1 == null_count_op2
    assert negative_count_op1 == negative_count_op2
    assert approx_cardinality_op1 == approx_cardinality_op2


def test_dedup_ops() -> None:
    num_rows_op = ops.NumRows()
    average_op = ops.Average("column")

    assert {num_rows_op, average_op} == {average_op, num_rows_op}
    assert {ops.Average("col"), num_rows_op, average_op} == {average_op, num_rows_op, ops.Average("col")}
    assert {ops.Average("column"), num_rows_op, average_op} == {average_op, num_rows_op}


def test_hashing() -> None:
    num_rows_op = ops.NumRows()
    average_op = ops.Average("column")
    minimum_op = ops.Minimum("column")
    maximum_op = ops.Maximum("column")
    sum_op = ops.Sum("column")
    first_op = ops.First("column")
    null_count_op = ops.NullCount("column")
    negative_count_op = ops.NegativeCount("column")
    approx_cardinality = ops.ApproxCardinality("column")

    assert hash(num_rows_op) == hash(ops.NumRows())
    assert hash(average_op) == hash(ops.Average("column"))
    assert hash(minimum_op) == hash(ops.Minimum("column"))
    assert hash(maximum_op) == hash(ops.Maximum("column"))
    assert hash(sum_op) == hash(ops.Sum("column"))
    assert hash(first_op) == hash(ops.First("column"))
    assert hash(null_count_op) == hash(ops.NullCount("column"))
    assert hash(negative_count_op) == hash(ops.NegativeCount("column"))
    assert hash(approx_cardinality) == hash(ops.ApproxCardinality("column"))


def test_average_not_equal() -> None:
    average_op1 = ops.Average("column1")
    average_op2 = ops.Average("column2")
    assert average_op1 != average_op2
    assert average_op1 != 1
