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
        Sequence of Metric objects that have the given execution_id in their tags.
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

    # Search for all metrics with the given execution_id tag
    # Using the search method with a filter on the tags JSON field
    db_metrics = db.search(DBMetric.tags.op("->>")("__execution_id") == execution_id)

    return db_metrics
