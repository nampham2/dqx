"""Data retrieval module for DQX metrics."""

from typing import Sequence

from dqx.models import Metric
from dqx.orm.repositories import MetricDB


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
