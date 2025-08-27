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
    BatchSqlDataSource,
    SqlDataSource,
    ResultKey,
)
from dqx.dialect import get_dialect
from dqx.ops import SketchOp, SqlOp
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

logger = logging.getLogger(__name__)
ColumnName = str
MetricKey = tuple[MetricSpec, ResultKey]

T = TypeVar("T", bound=SqlDataSource)


class AnalysisReport(UserDict[MetricKey, models.Metric]):
    def __init__(self, data: dict[MetricKey, models.Metric] = {}) -> None:
        self.data = data

    def merge(self, other: AnalysisReport) -> AnalysisReport:
        merged_report = toolz.merge_with(models.Metric.reduce, self, other)

        return AnalysisReport(data=merged_report)

    def show(self) -> None:
        # TODO(npham): Add more visualization options
        Console().print({v.spec.name: v.value for v in self.values()})


def analyze_sketch_ops(ds: T, ops: Sequence[SketchOp], batch_size: int = 100_000) -> None:
    if len(ops) == 0:
        return

    # Deduping the ops
    groups: dict[SketchOp, list[SketchOp]] = toolz.groupby(lambda op: op, ops)
    distinct_ops = list(groups.keys())

    # Constructing the query
    logger.info(f"Analyzing SketchOps: {distinct_ops}")
    query = textwrap.dedent(f"""\
        WITH source AS ( {ds.cte})
        SELECT {", ".join(op.sketch_column for op in distinct_ops)} FROM source
        """)

    # Fetch the only the required columns into memory by batch
    sketches = {op: op.create() for op in ops}
    batches = ds.query(query).fetch_arrow_reader(
        batch_size=batch_size,
    )

    # Fitting the sketches
    # TODO(npham): Possible optimization by fitting multiple ops per column
    for batch in batches:
        for op in ops:
            sketches[op].fit(batch[op.sketch_column])

    # Assign the collected values to the ops
    for op in ops:
        op.assign(sketches[op])


def analyze_sql_ops(ds: T, ops: Sequence[SqlOp]) -> None:
    if len(ops) == 0:
        return

    # Deduping the ops
    groups: dict[SqlOp, list[SqlOp]] = toolz.groupby(lambda op: op, ops)
    distinct_ops = list(groups.keys())

    # Constructing the query
    logger.info(f"Analyzing SqlOps: {distinct_ops}")
    
    # Require datasource to have a dialect
    if not hasattr(ds, 'dialect'):
        raise DQXError(f"Data source {ds.name} must have a dialect to analyze SQL ops")
    
    # Get the dialect instance from the registry
    dialect_instance = get_dialect(ds.dialect)
    
    # Generate SQL expressions using the dialect
    expressions = [dialect_instance.translate_sql_op(op) for op in distinct_ops]
    sql = dialect_instance.build_cte_query(ds.cte, expressions)

    # Execute the query
    logger.debug(f"SQL Query:\n{sql}")
    result: dict[str, np.ndarray] = ds.query(sql).fetchnumpy()

    # Assign the collected values to the ops
    cols: dict[SqlOp, str] = {op: op.sql_col for op in distinct_ops}
    for op in ops:
        op.assign(result[cols[op]][0])


class Analyzer:
    """
    The Analyzer class is responsible for analyzing data from DuckDataSource or DuckBatchDataSource
    using specified metrics and generating an AnalysisReport. It supports both single data source
    analysis and batch data source analysis.

    The class is thread-safe and can be used in a multi-threaded environment.
    """

    def __init__(self) -> None:
        self._report: AnalysisReport = AnalysisReport()

        # Mutex for updating the report
        self._mutex = Lock()

    @property
    def report(self) -> AnalysisReport:
        return self._report

    def analyze(
        self,
        ds: SqlDataSource | BatchSqlDataSource,
        metrics: Sequence[MetricSpec],
        key: ResultKey,
        threading: bool = False,
    ) -> AnalysisReport:
        if isinstance(ds, BatchSqlDataSource):
            if threading:
                return self._analyze_batches_threaded(ds, metrics, key)
            else:
                return self._analyze_batches(ds, metrics, key)

        if isinstance(ds, SqlDataSource):
            return self.analyze_single(ds, metrics, key)

        raise DQXError(f"Unsupported data source: {ds.name}")

    def _setup_duckdb(self) -> None:
        duckdb.execute("SET enable_progress_bar = false")

    def analyze_single(self, ds: SqlDataSource, metrics: Sequence[MetricSpec], key: ResultKey) -> AnalysisReport:
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
        self, ds: BatchSqlDataSource, metrics: Sequence[MetricSpec], key: ResultKey
    ) -> AnalysisReport:
        batch_id: int = 0
        for batch_ds in ds.batches():
            logger.info(f"Analyzing batch #{batch_id} ...")
            self.analyze_single(batch_ds, metrics, key)
            batch_id += 1
        return self._report

    def _analyze_batches_threaded(
        self, ds: BatchSqlDataSource, metrics: Sequence[MetricSpec], key: ResultKey, max_workers: int | None = None
    ) -> AnalysisReport:
        max_workers = max_workers or multiprocessing.cpu_count()
        logger.info(f"Analyzing batches with {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch_ds in ds.batches():
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
            logger.warning("Try to save an EMPTY analysis report!")
            return

        if overwrite:
            logger.info("Overwriting analysis report ...")
            db.persist(self._report.values())
        else:
            logger.info("Merging analysis report ...")
            self._merge_persist(db)
