from __future__ import annotations

import itertools
import logging
import multiprocessing
import textwrap
from collections import UserDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TypeVar

import duckdb
import numpy as np
import toolz
from rich.console import Console

from dqx import models
from dqx.common import (
    DQXError,
    DuckBatchDataSource,
    DuckDataSource,
    ResultKey,
)
from dqx.extensions.bigquery_ds import BigQueryDataSource
from dqx.ops import Average, First, Maximum, Minimum, NegativeCount, NullCount, NumRows, SketchOp, SqlOp, Sum, Variance
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

logger = logging.getLogger(__name__)
ColumnName = str
MetricKey = tuple[MetricSpec, ResultKey]

T = TypeVar("T", bound=DuckDataSource)


def translator(op: SqlOp[float]) -> str:
    match op:
        case NumRows():
            return f"CAST(COUNT(*) AS DOUBLE) AS `{op.sql_col}`"
        case Average(column):
            return f"AVG({column}) AS `{op.sql_col}`"
        case Minimum(column):
            return f"MIN({column}) AS `{op.sql_col}`"
        case Maximum(column):
            return f"MAX({column}) AS `{op.sql_col}`"
        case Sum(column):
            return f"SUM({column}) AS `{op.sql_col}`"
        case Variance(column):
            return f"VAR_SAMP({column}) AS `{op.sql_col}`"
        case First(column):
            return f"FIRST_VALUE({column}) OVER(PARTITION BY 1 ORDER BY RAND()) AS `{op.sql_col}`"
        case NullCount(column):
            return f"COUNTIF({column} IS NULL) AS `{op.sql_col}`"
        case NegativeCount(column):
            return f"COUNTIF({column} < 0) AS `{op.sql_col}`"
        case _:
            raise NotImplementedError(f"Unsupported SqlOp: {op}")


def analyze_sketch_ops(ds: T, ops: Sequence[SketchOp], batch_size: int = 100_000) -> None:
    raise NotImplementedError("Sketch ops are not supported in BigQuery!")


def analyze_sql_ops(ds: T, ops: Sequence[SqlOp]) -> None:
    if len(ops) == 0:
        return

    # Deduping the ops
    groups: dict[SqlOp, list[SqlOp]] = toolz.groupby(lambda op: op, ops)
    distinct_ops = list(groups.keys())

    # Constructing the query
    logger.info(f"Analyzing SqlOps: {distinct_ops}")
    sql = textwrap.dedent(f"""\
        WITH source AS ( {ds.cte} )
        SELECT {", ".join(op.sql for op in distinct_ops)} FROM source
        """)

    # Execute the query
    logger.debug(f"DuckDB SQL:\n{sql}")
    result: dict[str, np.ndarray] = ds.query(sql).fetchnumpy()

    # Assign the collected values to the ops
    cols: dict[SqlOp, str] = {op: translator(op) for op in distinct_ops}
    for op in ops:
        op.assign(result[cols[op]][0])


# TODO(npham): Analyze datasources in parallel
class Analyzer:
    """
    The Analyzer class is responsible for analyzing data from DuckDataSource or DuckBatchDataSource
    using specified metrics and generating an AnalysisReport. It supports both single data source
    analysis and batch data source analysis.

    The class is thread-safe and can be used in a multi-threaded environment.
    """
    def analyze()

    def analyze(
        self,
        ds: BigQueryDataSource,
        metrics: Sequence[MetricSpec],
        key: ResultKey,
        threading: bool = False,
    ) -> AnalysisReport:
        if isinstance(ds, DuckBatchDataSource):
            if threading:
                return self._analyze_batches_threaded(ds, metrics, key)
            else:
                return self._analyze_batches(ds, metrics, key)

        if isinstance(ds, BigQueryDataSource):
            return self.analyze_single(ds, metrics, key)

        raise DQXError(f"Unsupported data source: {ds.name}")
        return self._report

    def _setup_duckdb(self) -> None:
        duckdb.execute("SET enable_progress_bar = false")

    def analyze_single(self, ds: DuckDataSource, metrics: Sequence[MetricSpec], key: ResultKey) -> AnalysisReport:
        logger.info(f"Analyzing report with key {key}...")
        self._setup_duckdb()

        if len(metrics) == 0:
            raise DQXError("No metrics provided for analysis!")

        # All ops for the metrics
        all_ops = list(itertools.chain.from_iterable(m.analyzers for m in metrics))
        if len(all_ops) == 0:
            return AnalysisReport()

        # Analyze sql ops
        sql_ops = [op for op in all_ops if isinstance(op, SqlOp)]
        analyze_sql_ops(ds, sql_ops)

        # Analyze sketch ops
        sketch_ops = [op for op in all_ops if isinstance(op, SketchOp)]
        analyze_sketch_ops(ds, sketch_ops)

        # Build the analysis report and merge with the current one
        report = AnalysisReport(data={(metric, key): models.Metric.build(metric, key) for metric in metrics})

        with self._mutex:
            self._report = self._report.merge(report)

        return self._report

    def _analyze_batches(
        self, ds: DuckBatchDataSource, metrics: Sequence[MetricSpec], key: ResultKey
    ) -> AnalysisReport:
        batch_id: int = 0
        for batch_ds in ds.arrow_ds():
            logger.info(f"Analyzing batch #{batch_id} ...")
            self.analyze_single(batch_ds, metrics, key)
            batch_id += 1
        return self._report

    def _analyze_batches_threaded(
        self, ds: DuckBatchDataSource, metrics: Sequence[MetricSpec], key: ResultKey, max_workers: int | None = None
    ) -> AnalysisReport:
        max_workers = max_workers or multiprocessing.cpu_count()
        logger.info(f"Analyzing batches with {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch_ds in ds.arrow_ds():
                future = executor.submit(self.analyze_single, batch_ds, metrics, key)
                futures.append(future)

        for future in futures:
            future.result()

        return self._report

    def _merge_persist(self, db: MetricDB) -> None:
        db_report: AnalysisReport = AnalysisReport()

        for _, metric in self._report.items():
            # Find the metric in DB
            db_metric = db.get(metric.key, metric.spec)
            if db_metric is not None:
                db_report[_] = db_metric.unwrap()

        with self._mutex:
            report = self._report.merge(db_report)

        db.persist(report.values())

    def persist(self, db: MetricDB, overwrite: bool = True) -> None:
        # TODO(npham): Move persist to the analysis report
        if len(self._report) == 0:
            logger.warning("Try to save an empty analysis report!")
            return

        if overwrite:
            logger.info("Overwriting analysis report ...")
            db.persist(self._report.values())
        else:
            logger.info("Merging analysis report ...")
            self._merge_persist(db)
