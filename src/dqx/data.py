"""Data retrieval module for DQX metrics."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import pyarrow as pa
import sympy as sp

from dqx.common import ResultKey
from dqx.models import Metric
from dqx.provider import SymbolInfo
from dqx.specs import MetricSpec

if TYPE_CHECKING:
    from dqx.analyzer import AnalysisReport

# Define MetricKey type alias to match analyzer.py
MetricKey = tuple[MetricSpec, ResultKey, str]


@dataclass
class MetricTraceStats:
    """Statistics about discrepancies in a metric trace.

    Attributes:
        total_rows: Total number of rows in the trace
        discrepancy_count: Number of rows with value mismatches (excluding extended metrics)
        discrepancy_rows: List of row indices with discrepancies
        discrepancy_details: List of detailed information for each discrepancy row
    """

    total_rows: int
    discrepancy_count: int
    discrepancy_rows: list[int]
    discrepancy_details: list[dict[str, Any]]


def metrics_to_pyarrow_table(metrics: Sequence[Metric], execution_id: str) -> pa.Table:
    """
    Transform metrics from metrics_by_execution_id to a PyArrow table.

    The table schema matches the display format of print_metrics_by_execution_id
    with columns: date, metric, type, dataset, value, tags.

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
            "date": dates,
            "metric": metric_names,
            "type": types,
            "dataset": datasets,
            "value": values,
            "tags": tags,
        }
    )


def analysis_reports_to_pyarrow_table(report: "AnalysisReport", symbol_lookup: dict[MetricKey, sp.Symbol]) -> pa.Table:
    """
    Transform analysis reports from VerificationSuite to a PyArrow table.

    The table schema matches the display format of print_analysis_report
    with columns: date, metric, symbol, type, dataset, value, tags.

    Args:
        report: AnalysisReport containing metrics from analysis
        symbol_lookup: Dictionary mapping metric keys to their symbolic representations

    Returns:
        PyArrow table with all metrics from all reports, sorted by symbol indices
    """
    import re
    from datetime import date

    import pyarrow as pa

    # Collect all items from all reports
    all_items: list[tuple[MetricKey, Metric, str]] = []

    for metric_key, metric in report.items():
        # metric_key is (MetricSpec, ResultKey, DatasetName)
        symbol = str(symbol_lookup.get(metric_key, "-"))
        all_items.append((metric_key, metric, symbol))

    # Sort by symbol indices (x_1, x_2, ..., x_10, ..., x_20, ...)
    def symbol_sort_key(item: tuple[MetricKey, Metric, str]) -> tuple[int, int, str]:
        symbol = item[2]
        if symbol and symbol != "-":
            # Extract numeric part from x_N pattern
            match = re.match(r"x_(\d+)", symbol)
            if match:
                return (0, int(match.group(1)), "")  # (0, N, "") for symbols
        # For non-symbols, use metric name as secondary sort
        # metric_key[0] is MetricSpec in the 3-tuple
        return (1, 0, item[0][0].name)  # (1, 0, metric_name) for non-symbols

    sorted_items = sorted(all_items, key=symbol_sort_key)

    # Build column data
    dates: list[date] = []
    metric_names: list[str] = []
    symbols: list[str] = []
    types: list[str] = []
    datasets: list[str] = []
    values: list[float] = []
    tags: list[str] = []

    for metric_key, metric, symbol in sorted_items:
        # Unpack the 3-tuple MetricKey
        metric_spec, result_key, dataset_name = metric_key
        dates.append(result_key.yyyy_mm_dd)
        metric_names.append(metric_spec.name)
        symbols.append(symbol)
        types.append(metric_spec.metric_type)
        datasets.append(dataset_name)
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
            "date": dates,
            "metric": metric_names,
            "symbol": symbols,
            "type": types,
            "dataset": datasets,
            "value": values,
            "tags": tags,
        }
    )


def symbols_to_pyarrow_table(symbols: list[SymbolInfo]) -> pa.Table:
    """
    Transform a list of SymbolInfo objects to a PyArrow table.

    The table schema splits the value/error information into separate columns:
    date, symbol, metric, dataset, value, error, tags.

    Args:
        symbols: List of SymbolInfo objects from collect_symbols()

    Returns:
        PyArrow table with symbol data. Value column contains float values
        (null for failures), Error column contains error messages (null for successes).
    """
    from datetime import date

    import pyarrow as pa
    from returns.result import Failure, Success

    # Note: symbols are already sorted by the provider (x_1, x_2, ..., x_10 order)

    # Build column data
    dates: list[date] = []
    symbol_names: list[str] = []
    metrics: list[str] = []
    datasets: list[str] = []
    values: list[float | None] = []
    errors: list[str | None] = []
    tags: list[str] = []

    for symbol in symbols:
        dates.append(symbol.yyyy_mm_dd)
        symbol_names.append(symbol.name)
        metrics.append(symbol.metric)
        datasets.append(symbol.dataset or "-")

        # Split Result into value and error columns
        match symbol.value:
            case Success(value):
                values.append(value)
                errors.append(None)
            case Failure(error):
                values.append(None)
                errors.append(error)

        # Format tags
        if symbol.tags:
            tag_str = ", ".join(f"{k}={v}" for k, v in symbol.tags.items())
        else:
            tag_str = "-"
        tags.append(tag_str)

    # Create PyArrow table with proper types for nullable columns
    # We must explicitly set types to avoid 'null' type when all values are None
    return pa.table(
        [
            pa.array(dates, type=pa.date32()),
            pa.array(symbol_names, type=pa.string()),
            pa.array(metrics, type=pa.string()),
            pa.array(datasets, type=pa.string()),
            pa.array(values, type=pa.float64()),
            pa.array(errors, type=pa.string()),  # Explicit string type even if all None
            pa.array(tags, type=pa.string()),
        ],
        names=["date", "symbol", "metric", "dataset", "value", "error", "tags"],
    )


def metric_trace(
    metrics: Sequence[Metric],
    execution_id: str,
    reports: "AnalysisReport",
    symbols: list[SymbolInfo],
    symbol_lookup: dict[MetricKey, sp.Symbol],
) -> pa.Table:
    """
    Join metrics from DB, analysis reports, and symbols to trace metric values.

    Performs a FULL OUTER JOIN between metrics and analysis reports on (date, metric, dataset),
    followed by a LEFT JOIN with symbols on (date, symbol, metric, dataset) from symbols perspective.

    Args:
        metrics: Sequence of Metric objects from metrics_by_execution_id
        execution_id: The execution ID
        reports: Dictionary mapping datasource names to their AnalysisReports
        symbols: List of SymbolInfo objects from collect_symbols()

    Returns:
        PyArrow table with columns: date, metric, symbol, type, dataset,
        value_db, value_analysis, value_final, error, tags, is_extended
    """
    import pyarrow.compute as pc

    # Build a mapping of metric name to is_extended flag
    is_extended_map: dict[str, bool] = {}

    # Process metrics from DB
    for metric in metrics:
        is_extended_map[metric.spec.name] = metric.spec.is_extended

    # Process metrics from analysis reports
    for metric_key, metric in reports.items():
        # Unpack the 3-tuple MetricKey
        metric_spec, result_key, dataset_name = metric_key
        is_extended_map[metric_spec.name] = metric_spec.is_extended

    # Get individual tables
    metrics_table = metrics_to_pyarrow_table(metrics, execution_id)
    reports_table = analysis_reports_to_pyarrow_table(reports, symbol_lookup)
    symbols_table = symbols_to_pyarrow_table(symbols)

    # Rename value columns to avoid conflicts during joins
    if metrics_table.num_rows > 0:
        metrics_columns = metrics_table.column_names
        metrics_columns[metrics_columns.index("value")] = "value_db"
        metrics_table = metrics_table.rename_columns(metrics_columns)

    if reports_table.num_rows > 0:
        reports_columns = reports_table.column_names
        reports_columns[reports_columns.index("value")] = "value_analysis"
        reports_table = reports_table.rename_columns(reports_columns)

    if symbols_table.num_rows > 0:
        symbols_columns = symbols_table.column_names
        symbols_columns[symbols_columns.index("value")] = "value_final"
        symbols_table = symbols_table.rename_columns(symbols_columns)

    # Handle empty tables
    if metrics_table.num_rows == 0 and reports_table.num_rows == 0 and symbols_table.num_rows == 0:
        # Return empty table with expected schema
        return pa.table(
            {
                "date": pa.array([], type=pa.date32()),
                "metric": pa.array([], type=pa.string()),
                "symbol": pa.array([], type=pa.string()),
                "type": pa.array([], type=pa.string()),
                "dataset": pa.array([], type=pa.string()),
                "value_db": pa.array([], type=pa.float64()),
                "value_analysis": pa.array([], type=pa.float64()),
                "value_final": pa.array([], type=pa.float64()),
                "error": pa.array([], type=pa.string()),
                "tags": pa.array([], type=pa.string()),
                "is_extended": pa.array([], type=pa.bool_()),
            }
        )

    # First join: FULL OUTER JOIN metrics with reports
    if metrics_table.num_rows > 0 and reports_table.num_rows > 0:
        # PyArrow join with multiple keys
        first_join = metrics_table.join(
            reports_table,
            keys=["date", "metric", "dataset"],
            join_type="full outer",
            left_suffix="_left",
            right_suffix="_right",
        )

        # Handle duplicate columns from join
        # Coalesce type and tags columns (take non-null value)
        type_col = pc.coalesce(first_join["type_left"], first_join["type_right"])
        tags_col = pc.coalesce(first_join["tags_left"], first_join["tags_right"])

        # Build intermediate table
        first_join = pa.table(
            {
                "date": first_join["date"],
                "metric": first_join["metric"],
                "dataset": first_join["dataset"],
                "symbol": first_join["symbol"],
                "type": type_col,
                "value_db": first_join["value_db"],
                "value_analysis": first_join["value_analysis"],
                "tags": tags_col,
            }
        )
    elif metrics_table.num_rows > 0:
        # Only metrics data
        first_join = metrics_table.append_column("symbol", pa.array([None] * metrics_table.num_rows, type=pa.string()))
        first_join = first_join.append_column(
            "value_analysis", pa.array([None] * metrics_table.num_rows, type=pa.float64())
        )
    elif reports_table.num_rows > 0:
        # Only reports data
        first_join = reports_table.append_column(
            "value_db", pa.array([None] * reports_table.num_rows, type=pa.float64())
        )
    else:
        # Both empty, use symbols only
        first_join = None

    # Second join: LEFT JOIN from symbols perspective
    if symbols_table.num_rows > 0 and first_join is not None and first_join.num_rows > 0:
        final_join = symbols_table.join(
            first_join,
            keys=["date", "symbol", "metric", "dataset"],
            join_type="left outer",
            left_suffix="_symbols",
            right_suffix="_joined",
        )

        # Handle duplicate columns
        tags_col = pc.coalesce(final_join["tags_symbols"], final_join["tags_joined"])

        # Build final table
        # For columns that might be null, ensure they have the right type
        type_col = (
            final_join["type"]
            if "type" in final_join.column_names
            else pa.array([None] * final_join.num_rows, type=pa.string())
        )
        value_db_col = (
            final_join["value_db"]
            if "value_db" in final_join.column_names
            else pa.array([None] * final_join.num_rows, type=pa.float64())
        )
        value_analysis_col = (
            final_join["value_analysis"]
            if "value_analysis" in final_join.column_names
            else pa.array([None] * final_join.num_rows, type=pa.float64())
        )

        # Look up is_extended flag for each metric
        is_extended_values = []
        metric_names = final_join["metric"].to_pylist()
        for metric_name in metric_names:
            is_extended_values.append(is_extended_map.get(metric_name, False))

        result = pa.table(
            {
                "date": final_join["date"],
                "metric": final_join["metric"],
                "symbol": final_join["symbol"],
                "type": type_col,
                "dataset": final_join["dataset"],
                "value_db": value_db_col,
                "value_analysis": value_analysis_col,
                "value_final": final_join["value_final"],
                "error": final_join["error"],
                "tags": tags_col,
                "is_extended": pa.array(is_extended_values, type=pa.bool_()),
            }
        )
    elif symbols_table.num_rows > 0:
        # Only symbols data
        # Look up is_extended flag for each metric
        is_extended_values = []
        metric_names = symbols_table["metric"].to_pylist()
        for metric_name in metric_names:
            is_extended_values.append(is_extended_map.get(metric_name, False))

        result = pa.table(
            {
                "date": symbols_table["date"],
                "metric": symbols_table["metric"],
                "symbol": symbols_table["symbol"],
                "type": pa.array([None] * symbols_table.num_rows, type=pa.string()),
                "dataset": symbols_table["dataset"],
                "value_db": pa.array([None] * symbols_table.num_rows, type=pa.float64()),
                "value_analysis": pa.array([None] * symbols_table.num_rows, type=pa.float64()),
                "value_final": symbols_table["value_final"],
                "error": symbols_table["error"],
                "tags": symbols_table["tags"],
                "is_extended": pa.array(is_extended_values, type=pa.bool_()),
            }
        )
    else:
        # No symbols, just return first join result
        result = first_join.append_column("value_final", pa.array([None] * first_join.num_rows, type=pa.float64()))
        result = result.append_column("error", pa.array([None] * first_join.num_rows, type=pa.string()))

        # Reorder columns to match expected schema
        # Look up is_extended flag for each metric
        is_extended_values = []
        metric_names = result["metric"].to_pylist()
        for metric_name in metric_names:
            is_extended_values.append(is_extended_map.get(metric_name, False))

        result = pa.table(
            {
                "date": result["date"],
                "metric": result["metric"],
                "symbol": result["symbol"],
                "type": result["type"],
                "dataset": result["dataset"],
                "value_db": result["value_db"],
                "value_analysis": result["value_analysis"],
                "value_final": result["value_final"],
                "error": result["error"],
                "tags": result["tags"],
                "is_extended": pa.array(is_extended_values, type=pa.bool_()),
            }
        )

    return result


def metric_trace_stats(trace_table: pa.Table) -> MetricTraceStats:
    """
    Analyze a metric trace table for discrepancies.

    Identifies rows where values differ between database, analysis,
    and final stages. Extended metrics are excluded from discrepancy
    counts as they are computed metrics.

    Args:
        trace_table: PyArrow table from metric_trace()

    Returns:
        MetricTraceStats object with discrepancy analysis
    """
    # Convert PyArrow table to dict for easier processing
    data = trace_table.to_pydict()
    total_rows = trace_table.num_rows

    discrepancy_rows: list[int] = []
    discrepancy_details: list[dict[str, Any]] = []

    # Process each row
    for i in range(total_rows):
        # Extract values
        value_db = data["value_db"][i]
        value_analysis = data["value_analysis"][i]
        value_final = data["value_final"][i]
        is_extended = data["is_extended"][i] if "is_extended" in data else False

        # Check for discrepancies (only for non-extended metrics)
        if not is_extended:
            has_discrepancy = False
            discrepancy_info = {
                "row_index": i,
                "date": data["date"][i],
                "metric": data["metric"][i],
                "symbol": data["symbol"][i],
                "dataset": data["dataset"][i],
                "value_db": value_db,
                "value_analysis": value_analysis,
                "value_final": value_final,
                "discrepancies": [],
            }

            # Check each pair of values
            if value_db is not None and value_analysis is not None and value_db != value_analysis:
                has_discrepancy = True
                discrepancy_info["discrepancies"].append("value_db != value_analysis")

            if value_db is not None and value_final is not None and value_db != value_final:
                has_discrepancy = True
                discrepancy_info["discrepancies"].append("value_db != value_final")

            if value_analysis is not None and value_final is not None and value_analysis != value_final:
                has_discrepancy = True
                discrepancy_info["discrepancies"].append("value_analysis != value_final")

            if has_discrepancy:
                discrepancy_rows.append(i)
                discrepancy_details.append(discrepancy_info)

    return MetricTraceStats(
        total_rows=total_rows,
        discrepancy_count=len(discrepancy_rows),
        discrepancy_rows=discrepancy_rows,
        discrepancy_details=discrepancy_details,
    )
