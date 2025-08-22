from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import duckdb
import pyarrow as pa
from pyarrow.dataset import dataset

from dqx.analyzer import Analyzer
from dqx.common import SqlDataSource
from dqx.utils import random_prefix

if TYPE_CHECKING:
    from dqx.common import Analyzer as AnalyzerType

MAX_ARROW_BATCH_SIZE: int = 10_000_000


class ArrowDataSource:
    name: str = "pyarrow"
    analyzer_class: type[AnalyzerType] = Analyzer

    def __init__(self, table: pa.RecordBatch | pa.Table) -> None:
        self._table = table
        self._table_name = random_prefix(k=6)

    @property
    def cte(self) -> str:
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        return duckdb.arrow(self._table).query(
            self._table_name,
            query,
        )


class ArrowBatchDataSource:
    name: str = "pyarrow_batch"
    analyzer_class: type[AnalyzerType] = Analyzer

    def __init__(self, batches: Iterable[pa.RecordBatch | pa.Table]) -> None:
        self._batches = batches

    def arrow_ds(self) -> Iterable[ArrowDataSource]:
        for batch in self._batches:
            yield ArrowDataSource(batch)

    @classmethod
    def from_parquets(
        cls, parquets: Iterable[str], batch_size: int = MAX_ARROW_BATCH_SIZE, filesystem: Any | None = None
    ) -> ArrowBatchDataSource:
        return cls(dataset(parquets, format="parquet", filesystem=filesystem).to_batches(batch_size=batch_size))


class DuckRelationDataSource:
    name: str = "duckdb"
    analyzer_class: type[AnalyzerType] = Analyzer

    def __init__(self, relation: duckdb.DuckDBPyRelation) -> None:
        self._relation = relation
        self._table_name = random_prefix(k=6)

    @property
    def cte(self) -> str:
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        return self._relation.query(self._table_name, query)

    @classmethod
    def from_arrow(cls, table: pa.RecordBatch | pa.Table) -> SqlDataSource:
        return ArrowDataSource(table)
