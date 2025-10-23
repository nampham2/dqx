from __future__ import annotations

import datetime
import itertools
import logging
from collections import UserDict
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

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
from dqx.ops import SqlOp
from dqx.orm.repositories import MetricDB
from dqx.specs import MetricSpec

logger = logging.getLogger(__name__)
ColumnName = str
MetricKey = tuple[MetricSpec, ResultKey]

T = TypeVar("T", bound=SqlDataSource)

# Note: This design supports future enhancements such as:
# - Parallel batch processing (each batch can be processed independently)
# - Adaptive batch sizing based on query complexity or data volume
# - Custom batching strategies via subclassing or configuration
DEFAULT_BATCH_SIZE = 7  # Maximum dates per SQL query for optimal performance


def _validate_value(value: Any, date_str: str, symbol: str) -> float:
    """Validate a value from SQL query results.

    Args:
        value: The value to validate
        date_str: Date string for error context
        symbol: Symbol/column name for error context

    Returns:
        The validated float value

    Raises:
        DQXError: If value is masked, nan, null, or cannot be converted to float
    """
    # Check for numpy masked value
    if np.ma.is_masked(value):
        raise DQXError(
            f"Masked value encountered for symbol '{symbol}' on date {date_str}. "
            f"This typically means no data was found for the requested date."
        )

    # Check for None/null
    if value is None:
        raise DQXError(f"Null value encountered for symbol '{symbol}' on date {date_str}")

    # Try to convert to float and check for NaN
    try:
        float_value = float(value)
    except (ValueError, TypeError) as e:
        raise DQXError(
            f"Cannot convert value to float for symbol '{symbol}' on date {date_str}. Value: {value!r}, Error: {e}"
        )

    if np.isnan(float_value):
        raise DQXError(f"NaN value encountered for symbol '{symbol}' on date {date_str}")

    return float_value


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
    result: dict[str, np.ndarray] = ds.query(sql).fetchnumpy()

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


def analyze_batch_sql_ops(ds: T, ops_by_key: dict[ResultKey, list[SqlOp]]) -> None:
    """Analyze SQL ops for multiple dates in one query.

    Args:
        ds: Data source
        ops_by_key: Dict mapping ResultKey to list of deduplicated SqlOps

    Raises:
        DQXError: If SQL execution fails
    """
    if not ops_by_key:
        return

    # Get dialect
    dialect_instance = get_dialect(ds.dialect)

    # Build CTE data using dataclass
    from dqx.models import BatchCTEData

    cte_data = [BatchCTEData(key=key, cte_sql=ds.cte(key.yyyy_mm_dd), ops=ops) for key, ops in ops_by_key.items()]

    # Generate and execute SQL
    sql = dialect_instance.build_batch_cte_query(cte_data)

    # Format SQL for readability
    sql = sqlparse.format(
        sql,
        reindent=True,
        keyword_case="upper",
        identifier_case="lower",
        indent_width=2,
        wrap_after=120,
        comma_first=False,
    )

    logger.debug(f"Batch SQL Query:\n{sql}")

    # Execute query - will raise DQXError on failure
    result: dict[str, np.ndarray] = ds.query(sql).fetchnumpy()

    # Parse results - expecting columns: date, symbol, value
    date_col = result["date"]
    symbol_col = result["symbol"]
    value_col = result["value"]

    # Build lookup map
    value_map: dict[tuple[str, str], float] = {}
    for i in range(len(date_col)):
        date_str = date_col[i]
        symbol = symbol_col[i]
        value = value_col[i]
        # Validate before adding to map
        validated_value = _validate_value(value, date_str, symbol)
        value_map[(date_str, symbol)] = validated_value

    # Assign values back to ops
    for key, ops in ops_by_key.items():
        date_str = key.yyyy_mm_dd.isoformat()
        for op in ops:
            value = value_map.get((date_str, op.sql_col))
            if value is not None:
                op.assign(value)


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

        # Build the analysis report and merge with the current one
        report = AnalysisReport(data={(metric, key): models.Metric.build(metric, key) for metric in metrics})
        self._report = self._report.merge(report)
        return self._report

    def _setup_duckdb(self) -> None:
        duckdb.execute("SET enable_progress_bar = false")

    def analyze_batch(
        self,
        ds: SqlDataSource,
        metrics_by_key: Mapping[ResultKey, Sequence[MetricSpec]],
    ) -> AnalysisReport:
        """Analyze multiple dates with different metrics in batch.

        This method processes multiple ResultKeys efficiently by batching SQL
        operations. When the number of keys exceeds DEFAULT_BATCH_SIZE (7),
        the analysis is automatically split into smaller batches to optimize
        query performance and avoid excessively large SQL queries.

        Args:
            ds: The SQL data source to analyze
            metrics_by_key: Dictionary mapping ResultKeys to their metrics

        Returns:
            AnalysisReport containing all computed metrics for all dates

        Raises:
            DQXError: If no metrics provided or SQL execution fails

        Note:
            Large date ranges are automatically processed in batches of
            DEFAULT_BATCH_SIZE to maintain optimal performance. This limit
            can be adjusted by modifying the DEFAULT_BATCH_SIZE constant.
        """
        self._setup_duckdb()

        if not metrics_by_key:
            raise DQXError("No metrics provided for batch analysis!")

        # Log entry point with explicit dates
        dates = sorted([key.yyyy_mm_dd for key in metrics_by_key.keys()])
        if len(dates) <= 4:
            date_strs = [d.isoformat() for d in dates]
            logger.info(f"Analyzing batch of {len(metrics_by_key)} dates: {date_strs}")
        else:
            first_dates = [d.isoformat() for d in dates[:2]]
            last_dates = [d.isoformat() for d in dates[-2:]]
            logger.info(f"Analyzing batch of {len(metrics_by_key)} dates: {first_dates} ... {last_dates}")

        # Log batch processing info for large date ranges
        keys = list(metrics_by_key.keys())
        if len(keys) > DEFAULT_BATCH_SIZE:
            logger.debug(
                f"Processing {len(keys)} dates in batches of {DEFAULT_BATCH_SIZE}. "
                f"Date range: {keys[0].yyyy_mm_dd} to {keys[-1].yyyy_mm_dd}"
            )

        # Create final report at the beginning
        final_report = AnalysisReport()

        # Process in batches if needed
        items = list(metrics_by_key.items())

        for i in range(0, len(items), DEFAULT_BATCH_SIZE):
            batch_items = items[i : i + DEFAULT_BATCH_SIZE]
            batch = dict(batch_items)

            # Log batch boundaries
            if len(keys) > DEFAULT_BATCH_SIZE:
                batch_keys = [key for key, _ in batch_items]
                logger.debug(
                    f"Processing batch {i // DEFAULT_BATCH_SIZE + 1}: "
                    f"{batch_keys[0].yyyy_mm_dd} to {batch_keys[-1].yyyy_mm_dd} "
                    f"({len(batch_keys)} dates)"
                )

            report = self._analyze_batch_internal(ds, batch)
            # Merge directly into final report
            final_report = final_report.merge(report)

        self._report = self._report.merge(final_report)

        # Log result summary
        logger.info(f"Batch analysis complete: {len(final_report)} metrics computed")

        return self._report

    def _analyze_batch_internal(
        self,
        ds: SqlDataSource,
        metrics_by_key: dict[ResultKey, Sequence[MetricSpec]],
    ) -> AnalysisReport:
        """Process a single batch of dates.

        This method handles deduplication of SQL operations while ensuring
        all analyzer instances receive their computed values, even if they
        were deduplicated during SQL execution.

        Args:
            ds: Data source
            metrics_by_key: Batch of dates to process

        Returns:
            AnalysisReport for this batch
        """
        from collections import defaultdict

        # Maps (ResultKey, SqlOp) to all equivalent analyzer instances for that date
        analyzer_equivalence_map: defaultdict[tuple[ResultKey, SqlOp], list[SqlOp]] = defaultdict(list)

        # Phase 1: Collect all analyzers per date and build equivalence mapping
        for key, metrics in metrics_by_key.items():
            if not metrics:
                logger.warning(f"No metrics to analyze for date {key.yyyy_mm_dd}")
            for metric in metrics:
                for analyzer in metric.analyzers:
                    if isinstance(analyzer, SqlOp):
                        # Group by (date, analyzer) - same type on same date are equivalent
                        analyzer_equivalence_map[(key, analyzer)].append(analyzer)

        # Phase 2: Build ops_by_key from analyzer_equivalence_map keys
        ops_by_key: defaultdict[ResultKey, list[SqlOp]] = defaultdict(list)
        for key, analyzer in analyzer_equivalence_map.keys():
            ops_by_key[key].append(analyzer)

        # Log deduplication statistics
        if analyzer_equivalence_map:
            total_ops = sum(len(instances) for instances in analyzer_equivalence_map.values())
            actual_ops = len(analyzer_equivalence_map)
            reduction_pct = (1 - actual_ops / total_ops) * 100 if total_ops > 0 else 0
            logger.info(
                f"Batch deduplication: {actual_ops} unique ops out of {total_ops} total ({reduction_pct:.1f}% reduction)"
            )

        # Phase 3: Execute SQL with deduplicated ops
        if ops_by_key:
            analyze_batch_sql_ops(ds, dict(ops_by_key))

            # Phase 4: Propagate values to all equivalent analyzer instances
            for (key, representative), equivalent_instances in analyzer_equivalence_map.items():
                # Check if representative has a value by trying to get it
                try:
                    value = representative.value()
                    # Propagate to all instances for this specific date
                    for instance in equivalent_instances:
                        instance.assign(value)
                except DQXError:
                    # No value assigned yet, skip propagation
                    pass

        # Phase 5: Build report
        report_data = {}
        for key, metrics in metrics_by_key.items():
            for metric in metrics:
                report_data[(metric, key)] = models.Metric.build(metric, key)

        return AnalysisReport(data=report_data)
