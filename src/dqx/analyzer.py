from __future__ import annotations

import datetime
import itertools
import logging
import textwrap
from collections import UserDict
from collections.abc import Sequence
from typing import TypeVar

import duckdb
import numpy as np
import sqlparse
from rich.console import Console

from dqx import models
from dqx.common import (
    DQXError,
    ResultKey,
    SqlDataSource,
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
    def __init__(self, data: dict[MetricKey, models.Metric] | None = None) -> None:
        self.data = data if data is not None else {}

    def merge(self, other: AnalysisReport) -> AnalysisReport:
        """Merge two AnalysisReports, using Metric.reduce for conflicts.

        When the same (metric_spec, result_key) exists in both reports,
        the values are merged using Metric.reduce which applies the
        appropriate state merge operation (e.g., sum for SimpleAdditiveState).

        Args:
            other: Another AnalysisReport to merge with this one

        Returns:
            A new AnalysisReport containing all metrics from both reports
        """
        # Start with a copy of self.data for efficiency
        merged_data = dict(self.data)

        # Merge items from other
        for key, metric in other.items():
            if key in merged_data:
                # Key exists in both: use Metric.reduce to merge
                merged_data[key] = models.Metric.reduce([merged_data[key], metric])
            else:
                # Key only in other: just add it
                merged_data[key] = metric

        return AnalysisReport(data=merged_data)

    def show(self) -> None:
        # TODO(npham): Add more visualization options
        Console().print({v.spec.name: v.value for v in self.values()})

    def persist(self, db: MetricDB, overwrite: bool = True) -> None:
        """Persist the analysis report to the metric database.

        NOTE: This method is NOT thread-safe. If thread safety is required,
        it must be implemented by the caller.

        Args:
            db: MetricDB instance for persistence
            overwrite: If True, overwrite existing metrics. If False, merge with existing.
        """
        if len(self) == 0:  # Changed from self._report
            logger.warning("Try to save an EMPTY analysis report!")
            return

        if overwrite:
            logger.info("Overwriting analysis report ...")
            db.persist(self.values())
        else:
            logger.info("Merging analysis report ...")
            self._merge_persist(db)

    def _merge_persist(self, db: MetricDB) -> None:
        """Merge with existing metrics in the database before persisting.

        NOTE: This method is NOT thread-safe.

        Args:
            db: MetricDB instance for persistence
        """
        db_report = AnalysisReport()

        for key, metric in self.items():  # Changed from self._report.items()
            # Find the metric in DB
            db_metric = db.get(metric.key, metric.spec)
            if db_metric is not None:
                db_report[key] = db_metric.unwrap()

        # Merge and persist
        merged_report = self.merge(db_report)
        db.persist(merged_report.values())


def analyze_sketch_ops(ds: T, ops: Sequence[SketchOp], batch_size: int, nominal_date: datetime.date) -> None:
    if len(ops) == 0:
        return

    # Deduping the ops preserving order of first occurrence
    seen = set()
    distinct_ops = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            distinct_ops.append(op)

    # Constructing the query
    logger.info(f"Analyzing SketchOps: {distinct_ops}")
    query = textwrap.dedent(f"""\
        WITH source AS ( {ds.cte(nominal_date)})
        SELECT {", ".join(op.sketch_column for op in distinct_ops)} FROM source
        """)

    # Fetch the only the required columns into memory by batch
    sketches = {op: op.create() for op in ops}
    batches = ds.query(query, nominal_date).fetch_arrow_reader(
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


def analyze_sql_ops(ds: T, ops: Sequence[SqlOp], nominal_date: datetime.date) -> None:
    if len(ops) == 0:
        return

    # Deduping the ops preserving order of first occurrence
    seen = set()
    distinct_ops = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            distinct_ops.append(op)

    # Constructing the query
    logger.info(f"Analyzing SqlOps: {distinct_ops}")

    # Get the dialect instance from the registry
    dialect_instance = get_dialect(ds.dialect)

    # Generate SQL expressions using the dialect
    expressions = [dialect_instance.translate_sql_op(op) for op in distinct_ops]
    sql = dialect_instance.build_cte_query(ds.cte(nominal_date), expressions)

    # Format SQL for consistent output
    sql = sqlparse.format(
        sql,
        reindent=True,
        keyword_case="upper",
        identifier_case="lower",
        indent_width=2,
        wrap_after=120,
        comma_first=False,
    )

    # Execute the query
    logger.debug(f"SQL Query:\n{sql}")
    result: dict[str, np.ndarray] = ds.query(sql, nominal_date).fetchnumpy()

    # Assign the collected values to the ops
    # Create a mapping from all ops to their sql_col (duplicates will map to same col)
    cols: dict[SqlOp, str] = {}
    for op in ops:
        # Find the corresponding distinct op that has the same value
        for distinct_op in distinct_ops:
            if op == distinct_op:
                cols[op] = distinct_op.sql_col
                break

    # Now assign values to all ops
    for op in ops:
        op.assign(result[cols[op]][0])


class Analyzer:
    """
    The Analyzer class is responsible for analyzing data from SqlDataSource
    using specified metrics and generating an AnalysisReport.

    Note: This class is NOT thread-safe. Thread safety must be handled by callers if needed.
    """

    def __init__(self) -> None:
        # TODO(npham): Remove _report and make the analyzer stateless.
        self._report: AnalysisReport = AnalysisReport()

    @property
    def report(self) -> AnalysisReport:
        return self._report

    def analyze(
        self,
        ds: SqlDataSource,
        metrics: Sequence[MetricSpec],
        key: ResultKey,
    ) -> AnalysisReport:
        """Analyze a data source using specified metrics.

        Args:
            ds: The SQL data source to analyze
            metrics: Sequence of metrics to compute
            key: Result key for the analysis

        Returns:
            AnalysisReport containing computed metrics
        """
        logger.info(f"Analyzing report with key {key}...")
        self._setup_duckdb()

        if len(metrics) == 0:
            raise DQXError("No metrics provided for analysis!")

        # All ops for the metrics
        all_ops = list(itertools.chain.from_iterable(m.analyzers for m in metrics))
        if len(all_ops) == 0:
            return AnalysisReport()

        # Analyze sql ops - pass the date from key
        sql_ops = [op for op in all_ops if isinstance(op, SqlOp)]
        analyze_sql_ops(ds, sql_ops, key.yyyy_mm_dd)

        # Analyze sketch ops - pass the date from key
        sketch_ops = [op for op in all_ops if isinstance(op, SketchOp)]
        analyze_sketch_ops(ds, sketch_ops, batch_size=100_000, nominal_date=key.yyyy_mm_dd)

        # Build the analysis report and merge with the current one
        report = AnalysisReport(data={(metric, key): models.Metric.build(metric, key) for metric in metrics})
        self._report = self._report.merge(report)
        return self._report

    def _setup_duckdb(self) -> None:
        duckdb.execute("SET enable_progress_bar = false")
