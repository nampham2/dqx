from __future__ import annotations

from typing import Any, Iterable

import duckdb
import pyarrow as pa
from pyarrow.dataset import dataset

from dqx.analyzer import Analyzer
from dqx.common import random_prefix


class ArrowDataSource:
    name: str = "pyarrow"
    analyzer_class = Analyzer

    def __init__(self, table: pa.Table | pa.RecordBatch) -> None:
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
    name: str = "pyarrow"
    analyzer_class = Analyzer

    def __init__(self, batches: Iterable[pa.RecordBatch | pa.Table]) -> None:
        self._batches = batches

    def arrow_ds(self) -> Iterable[ArrowDataSource]:
        for batch in self._batches:
            yield ArrowDataSource(batch)

    @classmethod
    def from_parquets(
        cls, parquets: Iterable[str], batch_size: int = 10_000_000, filesystem: Any | None = None
    ) -> ArrowBatchDataSource:
        return cls(dataset(parquets, format="parquet", filesystem=filesystem).to_batches(batch_size=batch_size))


class DuckRelationDataSource:
    name: str = "duckdb"
    analyzer_class = Analyzer

    def __init__(self, relation: duckdb.DuckDBPyRelation) -> None:
        self._relation = relation
        self._table_name = random_prefix(k=6)

    @property
    def cte(self) -> str:
        return f"SELECT * FROM {self._table_name}"

    def query(self, query: str) -> duckdb.DuckDBPyRelation:
        return self._relation.query(self._table_name, query)
