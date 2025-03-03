import datetime as dt

import pyarrow as pa
import pytest

from dqx import specs
from dqx.analyzer import Analyzer
from dqx.common import DuckDataSource, ResultKey
from dqx.extensions.pyarrow_ds import ArrowDataSource
from dqx.orm.repositories import InMemoryMetricDB


@pytest.fixture
def test_metrics() -> list[specs.MetricSpec]:
    return [
        specs.NumRows(),
        specs.Average("int_col"),
        specs.Minimum("int_col"),
        specs.Maximum("int_col"),
        specs.Sum("int_col"),
        specs.NullCount("null_col"),
        specs.NegativeCount("neg_col"),
        specs.ApproxCardinality("int_col"),
    ]


@pytest.fixture
def duck_data() -> DuckDataSource:
    int_col = pa.array(range(10))
    one_nulls: list[int | None] = list(range(10))
    one_nulls[1] = None
    one_negative: list[int] = list(range(-1, 9))

    with_null_col = pa.array(one_nulls)
    tbl = pa.Table.from_arrays([int_col, with_null_col, one_negative], names=["int_col", "null_col", "neg_col"])
    return ArrowDataSource(tbl)


@pytest.fixture
def cte(duck_data: None) -> str:
    return """SELECT * FROM tbl1"""


@pytest.fixture
def key() -> ResultKey:
    return ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-02-04"), tags={})


def test_duck_analyzer(
    test_metrics: list[specs.MetricSpec],
    key: ResultKey,
    duck_data: DuckDataSource,
) -> None:
    analyzer = Analyzer()
    report = analyzer.analyze_single(duck_data, test_metrics, key)
    assert isinstance(analyzer, Analyzer)
    assert report[(test_metrics[0], key)].value == pytest.approx(10)

    report = analyzer.analyze_single(duck_data, test_metrics, key)
    assert report[(test_metrics[0], key)].value == pytest.approx(20)

    analyzer.report.show()

    db = InMemoryMetricDB()
    analyzer.persist(db)

    # Try to persist twice
    analyzer.persist(db, overwrite=False)


def test_duck_analyzer_multiple_days(
    test_metrics: list[specs.MetricSpec],
    key: ResultKey,
    duck_data: DuckDataSource,
) -> None:
    analyzer: Analyzer = Analyzer()
    report = analyzer.analyze_single(duck_data, test_metrics, key)
    assert len(report) == 8
    report = analyzer.analyze_single(duck_data, test_metrics, key)
    assert len(report) == 8
    report = analyzer.analyze_single(duck_data, test_metrics, key.lag(1))
    assert len(report) == 16
