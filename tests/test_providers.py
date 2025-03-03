# import datetime as dt

# import pyarrow as pa
# import pytest
# from returns.result import Success
# from rich.console import Console

# from dqx import specs
# from dqx.analyzer import Analyzer
# from dqx.common import ResultKey
# from dqx.extensions.pyarrow_ds import ArrowDataSource
# from dqx.orm.repositories import InMemoryMetricDB
# from dqx.provider import MetricProvider


# def test_print(commerce_data: pa.Table) -> None:
#     console = Console()
#     console.print(commerce_data.slice(length=3).to_pydict())


# # def test_metric_provider(commerce_data: pa.Table, console: Console) -> None:
#     # ds = ArrowDataSource(commerce_data)
#     # ds.query(ds.cte + " limit 5").show()

#     # db = InMemoryMetricDB()
#     # key = ResultKey(yyyy_mm_dd=dt.date.fromisoformat("2025-01-15"), tags={})

#     # provider = MetricProvider(db)
#     # avg_price = provider.metric(specs.Average("price"))
#     # null_delivered = provider.metric(specs.NullCount("delivered"))
#     # nr_dod = provider.ext.day_over_day(specs.NumRows())

#     # analyzer: Analyzer = Analyzer()
#     # analyzer.analyze_single(ds, pendings, key=key)
#     # analyzer.analyze_single(ds, pendings, key=key.lag(1))
#     # analyzer.persist(db)
#     # assert provider.evaluate(nr_dod, key) == Success(pytest.approx(1.0))
#     # assert provider.evaluate(avg_price, key).unwrap() > 0
#     # assert provider.evaluate(null_delivered, key).unwrap() > 0
