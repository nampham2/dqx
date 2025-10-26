"""Data retrieval module for DQX metrics."""

from typing import TYPE_CHECKING, Sequence

from dqx.models import Metric
from dqx.orm.repositories import MetricDB

if TYPE_CHECKING:
    import pyarrow as pa

    from dqx.analyzer import AnalysisReport


def metrics_by_execution_id(db: MetricDB, execution_id: str) -> Sequence[Metric]:
    """
    Retrieve all metrics associated with a specific execution ID.

    This function queries the metric database for all metrics that were
    created during a specific VerificationSuite execution, identified by
    its unique execution ID.

    Args:
        db: The MetricDB instance to query
        execution_id: The UUID string identifying the execution

    Returns:
        Sequence of Metric objects that have the given execution_id in their metadata.
        Returns empty sequence if no metrics are found.

    Example:
        >>> from dqx.orm.repositories import InMemoryMetricDB
        >>> from dqx import data
        >>>
        >>> db = InMemoryMetricDB()
        >>> # After running a VerificationSuite...
        >>> execution_id = suite.execution_id
        >>> metrics = data.metrics_by_execution_id(db, execution_id)
        >>>
        >>> # Process the metrics
        >>> for metric in metrics:
        ...     print(f"{metric.spec.name}: {metric.value}")
    """
    # Import here to avoid circular imports
    from dqx.orm.repositories import Metric as DBMetric

    # Get all metrics and filter by execution_id in Python
    # This is necessary because SQLite doesn't support the JSON operators
    # that work with PostgreSQL
    session = db.new_session()

    # Query all metrics
    all_db_metrics = session.query(DBMetric).all()

    # Filter metrics that have the matching execution_id
    matching_metrics = []
    for db_metric in all_db_metrics:
        if db_metric.meta and db_metric.meta.execution_id == execution_id:
            matching_metrics.append(db_metric.to_model())

    return matching_metrics


def metrics_to_pyarrow_table(metrics: Sequence[Metric], execution_id: str) -> "pa.Table":
    """
    Transform metrics from metrics_by_execution_id to a PyArrow table.

    The table schema matches the display format of print_metrics_by_execution_id
    with columns: Date, Metric Name, Type, Dataset, Value, Tags.

    Args:
        metrics: Sequence of Metric objects from metrics_by_execution_id
        execution_id: The execution ID (included for consistency with display function)

    Returns:
        PyArrow table with metrics data, sorted by date (newest first) then by name
    """
    from datetime import date

    import pyarrow as pa

    # Sort metrics: newest dates first, then alphabetical by name
    sorted_metrics = sorted(metrics, key=lambda m: (-m.key.yyyy_mm_dd.toordinal(), m.spec.name))

    # Build column data
    dates: list[date] = []
    metric_names: list[str] = []
    types: list[str] = []
    datasets: list[str] = []
    values: list[float] = []
    tags: list[str] = []

    for metric in sorted_metrics:
        dates.append(metric.key.yyyy_mm_dd)
        metric_names.append(metric.spec.name)
        types.append(metric.spec.metric_type)
        datasets.append(metric.dataset)
        values.append(metric.value)

        # Format tags
        if metric.key.tags:
            tag_str = ", ".join(f"{k}={v}" for k, v in metric.key.tags.items())
        else:
            tag_str = "-"
        tags.append(tag_str)

    # Create PyArrow table
    return pa.Table.from_pydict(
        {
            "Date": dates,
            "Metric Name": metric_names,
            "Type": types,
            "Dataset": datasets,
            "Value": values,
            "Tags": tags,
        }
    )


def analysis_reports_to_pyarrow_table(reports: dict[str, "AnalysisReport"]) -> "pa.Table":
    """
    Transform analysis reports from VerificationSuite to a PyArrow table.

    The table schema matches the display format of print_analysis_report
    with columns: Date, Metric Name, Symbol, Type, Dataset, Value, Tags.

    Args:
        reports: Dictionary mapping datasource names to their AnalysisReports

    Returns:
        PyArrow table with all metrics from all reports, sorted by date (newest first) then by name
    """
    from datetime import date

    import pyarrow as pa

    from dqx.common import ResultKey
    from dqx.specs import MetricSpec

    # Collect all items from all reports
    all_items: list[tuple[tuple[MetricSpec, ResultKey], Metric, str]] = []

    for ds_name, ds_report in reports.items():
        for metric_key, metric in ds_report.items():
            # metric_key is (MetricSpec, ResultKey)
            symbol = ds_report.symbol_mapping.get(metric_key, "-")
            all_items.append((metric_key, metric, symbol))

    # Sort by date (newest first) then by metric name
    sorted_items = sorted(all_items, key=lambda x: (-x[0][1].yyyy_mm_dd.toordinal(), x[0][0].name))

    # Build column data
    dates: list[date] = []
    metric_names: list[str] = []
    symbols: list[str] = []
    types: list[str] = []
    datasets: list[str] = []
    values: list[float] = []
    tags: list[str] = []

    for (metric_spec, result_key), metric, symbol in sorted_items:
        dates.append(result_key.yyyy_mm_dd)
        metric_names.append(metric_spec.name)
        symbols.append(symbol)
        types.append(metric_spec.metric_type)
        datasets.append(metric.dataset or "-")
        values.append(metric.value)

        # Format tags
        if result_key.tags:
            tag_str = ", ".join(f"{k}={v}" for k, v in result_key.tags.items())
        else:
            tag_str = "-"
        tags.append(tag_str)

    # Create PyArrow table
    return pa.Table.from_pydict(
        {
            "Date": dates,
            "Metric Name": metric_names,
            "Symbol": symbols,
            "Type": types,
            "Dataset": datasets,
            "Value": values,
            "Tags": tags,
        }
    )
