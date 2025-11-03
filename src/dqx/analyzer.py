from __future__ import annotations

import logging
import math
from collections import UserDict, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import sqlparse
from returns.result import Failure, Success

from dqx import models, states
from dqx.cache import MetricCache
from dqx.common import (
    DQXError,
    ExecutionId,
    Metadata,
    MetricKey,
    ResultKey,
    SqlDataSource,
)
from dqx.dialect import get_dialect
from dqx.ops import SqlOp
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider, SymbolicMetric
from dqx.specs import MetricSpec

DEFAULT_BATCH_SIZE = 14  # Maximum dates per analysis SQL query

ColumnName = str
T = TypeVar("T", bound=SqlDataSource)


logger = logging.getLogger(__name__)


def _validate_value(value: Any, date_str: str, symbol: str) -> float:
    """Validate a value from SQL query results.

    Args:
        value: The value to validate
        date_str: Date string for error context
        symbol: Symbol/column name for error context

    Returns:
        The validated float value

    Raises:
        DQXError: If value is null or cannot be converted to float
    """
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

    if math.isnan(float_value):
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

        merged_report = AnalysisReport(data=merged_data)
        return merged_report

    def show(self, symbol_lookup: dict[MetricKey, Any]) -> None:
        from dqx.display import print_analysis_report

        print_analysis_report(self, symbol_lookup)

    def persist(self, db: MetricDB, cache: MetricCache) -> None:
        """Persist the analysis report to the metric database.

        NOTE: This method is NOT thread-safe. If thread safety is required,
        it must be implemented by the caller.

        Args:
            db: MetricDB instance for persistence
            cache: MetricCache instance to warm the cache when persisting
            overwrite: If True, overwrite existing metrics. If False, merge with existing.
        """
        if len(self) == 0:  # Changed from self._report
            logger.warning("Try to save an EMPTY analysis report!")
            return

        # Overwrite metrics in DB
        logger.info("Overwriting analysis report ...")
        cache.put(list(self.values()), mark_dirty=True)
        cache.write_back()


def analyze_sql_ops(ds: T, ops_by_key: dict[ResultKey, list[SqlOp]]) -> None:
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
    from dqx.dialect import BatchCTEData

    cte_data = [BatchCTEData(key=key, cte_sql=ds.cte(key.yyyy_mm_dd), ops=ops) for key, ops in ops_by_key.items()]

    # Generate and execute SQL
    sql = dialect_instance.build_batch_cte_query(cte_data)

    # Format SQL for readability
    sql = sqlparse.format(
        sql,
        reindent_aligned=True,
        keyword_case="upper",
        identifier_case="lower",
        indent_width=2,
        wrap_after=120,
        comma_first=False,
        compact=True,
    )

    if logger.isEnabledFor(logging.DEBUG):
        print(f"SQL Query:\n{sql}")

    # Execute query and process MAP results
    result = ds.query(sql).fetchall()

    # Build date lookup map to ensure proper alignment
    date_to_ops: dict[str, tuple[ResultKey, list[SqlOp]]] = {
        key.yyyy_mm_dd.isoformat(): (key, ops) for key, ops in ops_by_key.items()
    }

    # Process results - expecting (date, values) tuples
    for date_str, values_data in result:
        if date_str not in date_to_ops:
            raise DQXError(f"Unexpected date '{date_str}' in SQL results. Expected dates: {sorted(date_to_ops.keys())}")

        key, ops = date_to_ops[date_str]
        # values_data is array of {key: str, value: float}
        values_map = {item["key"]: item["value"] for item in values_data}

        for op in ops:
            if op.sql_col in values_map:
                # Validate and assign the value
                validated_value = _validate_value(values_map[op.sql_col], date_str, op.sql_col)
                op.assign(validated_value)

    # Check that all expected dates were present in results
    result_dates = {row[0] for row in result}
    expected_dates = set(date_to_ops.keys())
    missing_dates = expected_dates - result_dates
    if missing_dates:
        raise DQXError(f"Missing dates in SQL results: {sorted(missing_dates)}. Got: {sorted(result_dates)}")


class Analyzer:
    """
    The Analyzer class is responsible for analyzing data from SqlDataSource
    using specified metrics and generating an AnalysisReport.

    Note: This class is NOT thread-safe. Thread safety must be handled by callers if needed.
    """

    def __init__(
        self,
        datasources: list[SqlDataSource],
        provider: MetricProvider,
        key: ResultKey,
        execution_id: ExecutionId,
    ) -> None:
        self.datasources = datasources
        self.provider = provider
        self.key = key
        self.execution_id = execution_id

    @property
    def metrics(self) -> list[SymbolicMetric]:
        return self.provider.registry.metrics

    @property
    def db(self) -> MetricDB:
        return self.provider._db

    @property
    def cache(self) -> MetricCache:
        return self.provider._cache

    def _analyze_internal(
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
            analyze_sql_ops(ds, dict(ops_by_key))

            # Phase 4: Propagate values to all equivalent analyzer instances
            for (key, representative), equivalent_instances in analyzer_equivalence_map.items():
                # Check if representative has a value by trying to get it
                try:
                    value = representative.value()
                    # Propagate to all instances for this specific date
                    for instance in equivalent_instances:
                        instance.assign(value)
                except DQXError:
                    raise DQXError(f"Failed to retrieve value for analyzer {representative} on date {key.yyyy_mm_dd}")

        # Phase 5: Build report
        report_data: dict[MetricKey, models.Metric] = {}
        report = AnalysisReport(data=report_data)

        metadata = Metadata(execution_id=self.execution_id)

        for key, metrics in metrics_by_key.items():
            for metric in metrics:
                metric_key = (metric, key, ds.name)
                report_data[metric_key] = models.Metric.build(metric, key, dataset=ds.name, metadata=metadata)

        return report

    def analyze_simple_metrics(
        self,
        ds: SqlDataSource,
        metrics: Mapping[ResultKey, Sequence[MetricSpec]],
    ) -> AnalysisReport:
        """Analyze multiple dates with different metrics in batch.

        This method processes multiple ResultKeys efficiently by batching SQL
        operations. When the number of keys exceeds DEFAULT_BATCH_SIZE (14),
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
        if not metrics:
            raise DQXError("No metrics provided for batch analysis!")

        # Log entry point with explicit dates
        dates = sorted([key.yyyy_mm_dd for key in metrics.keys()])
        date_strs = ", ".join(d.isoformat() for d in dates)
        logger.info(f"Analyzing dataset {ds.name} for {len(metrics)} dates: {date_strs}")

        # Create final report at the beginning
        final_report = AnalysisReport()

        # Process in batches if needed
        items = list(metrics.items())

        for i in range(0, len(items), DEFAULT_BATCH_SIZE):
            batch_items = items[i : i + DEFAULT_BATCH_SIZE]
            batch = dict(batch_items)

            # Log batch boundaries
            batch_keys = [key for key, _ in batch_items]
            logger.info(
                f"Processing batch {i // DEFAULT_BATCH_SIZE + 1}: {', '.join(str(key.yyyy_mm_dd) for key in batch_keys)}"
            )

            report = self._analyze_internal(ds, batch)
            # Merge directly into final report
            final_report = final_report.merge(report)

        # Log result summary
        logger.info(f"Analysis complete: {len(final_report)} metrics computed")
        return final_report

    def analyze_extended_metrics(self) -> AnalysisReport:
        # First sort the metrics topologically for analysis
        self.provider.registry.topological_sort()

        report: AnalysisReport = AnalysisReport()
        metadata = Metadata(execution_id=self.execution_id)

        for sym_metric in self.metrics:
            if sym_metric.metric_spec.is_extended:
                # Calculate effective key with lag
                effective_key = self.key.lag(sym_metric.lag)

                # Extended metrics ALWAYS have a dataset - they inherit from base metrics
                assert sym_metric.dataset is not None, f"Extended metric {sym_metric.name} has no dataset"

                try:
                    result = sym_metric.fn(effective_key)

                    match result:
                        case Success(value):
                            # Build the metric key
                            metric_key = (sym_metric.metric_spec, effective_key, sym_metric.dataset)

                            # Create NonMergeable state with the actual computed value
                            state = states.NonMergeable(value=value, metric_type=sym_metric.metric_spec.metric_type)

                            metric = models.Metric.build(
                                metric=sym_metric.metric_spec,
                                key=effective_key,
                                dataset=sym_metric.dataset,
                                state=state,
                                metadata=metadata,
                            )

                            report[metric_key] = metric
                            # Mark as dirty for batch persistence
                            self.cache.put(metric, mark_dirty=True)

                        case Failure(error):
                            logger.warning(f"Failed to evaluate {sym_metric.name}: {error}")

                except Exception as e:
                    logger.error(f"Error evaluating {sym_metric.name}: {e}", exc_info=True)

        logger.info(f"Evaluated {len(report)} extended metrics")

        # Flush all dirty metrics to DB
        self.cache.write_back()

        return report

    def analyze(self) -> AnalysisReport:
        # Store analysis reports by datasource name
        report: AnalysisReport = AnalysisReport()

        # Group metrics by dataset
        metrics_by_dataset: dict[str, list[SymbolicMetric]] = defaultdict(list)
        for sym_metric in self.metrics:
            assert sym_metric.dataset is not None, f"Metric {sym_metric.name} has no dataset"
            metrics_by_dataset[sym_metric.dataset].append(sym_metric)

        # Phase 1: Analyze simple metrics for each datasource
        for ds in self.datasources:
            # Get all metrics for this dataset
            all_metrics = metrics_by_dataset.get(ds.name, [])

            # Filter to only include simple metrics (not extended)
            relevant_metrics = [sym_metric for sym_metric in all_metrics if not sym_metric.metric_spec.is_extended]

            # Skip if no simple metrics for this dataset
            if not relevant_metrics:
                continue

            # Group metrics by their effective date
            metrics_by_date: dict[ResultKey, list[MetricSpec]] = defaultdict(list)
            for sym_metric in relevant_metrics:
                # Use lag directly instead of key_provider
                effective_key = self.key.lag(sym_metric.lag)
                metrics_by_date[effective_key].append(sym_metric.metric_spec)

            # Analyze each date group separately
            this_report = self.analyze_simple_metrics(ds, metrics_by_date)
            report.update(this_report)

        # Persist simple metrics before evaluating extended metrics
        report.persist(self.db, self.cache)

        # Phase 2: Evaluate extended metrics AFTER all simple metrics are persisted
        logger.info("Evaluating extended metrics...")
        extended_report = self.analyze_extended_metrics()
        report.update(extended_report)

        return report
