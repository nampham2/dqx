import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from rich.console import Console

from dqx import get_logger
from tests.fixtures.data_fixtures import commerce_data_c1, commerce_data_c2  # noqa: F401


@pytest.fixture(autouse=True)
def run_around_tests() -> Iterator:
    # Use dqx logger with default format
    get_logger(level=logging.DEBUG, force_reconfigure=True)
    print("\n")
    yield


@pytest.fixture(scope="session")
def console() -> Console:
    return Console()


@pytest.fixture
def isolated_dialect_registry() -> Iterator[dict[str, type]]:
    """Provide an isolated dialect registry for tests.

    This fixture creates a fresh copy of the dialect registry for each test,
    preventing test interference when registering dialects.
    """
    # Import here to avoid circular imports
    from dqx.dialect import _DIALECT_REGISTRY, BigQueryDialect, DuckDBDialect

    # Create a fresh registry with only built-in dialects
    clean_registry = {
        "duckdb": DuckDBDialect,
        "bigquery": BigQueryDialect,
    }

    with patch.dict("dqx.dialect._DIALECT_REGISTRY", clean_registry, clear=True):
        yield _DIALECT_REGISTRY
