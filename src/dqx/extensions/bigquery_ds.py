from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import duckdb
import pyarrow as pa
from pyarrow.dataset import dataset

from dqx.analyzer import Analyzer
from dqx.common import random_prefix


class BigQueryDataSource(ABC):
    name: str = "bigquery"
    analyzer_class = Analyzer

    @abstractmethod
    @property
    def cte(self) -> str: ...

class BQTableDataSource(BigQueryDataSource):
    def __init__(self, ) -> None:
        self.table = table
        self.project = project
        self.dataset = dataset

    @property
    def cte(self) -> str:
        return f"{self.project}.{self.dataset}.{self.table}"