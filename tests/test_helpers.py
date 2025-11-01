"""Helper utilities for test files."""

from dqx.cache import MetricCache
from dqx.orm.repositories import InMemoryMetricDB


def create_test_cache() -> MetricCache:
    """Create a MetricCache instance for testing with an in-memory database.

    Returns:
        A MetricCache instance suitable for testing
    """
    db = InMemoryMetricDB()
    return MetricCache(db)
