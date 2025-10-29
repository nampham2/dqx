import logging
import os
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


@pytest.fixture
def bigquery_emulator() -> Iterator[str | None]:
    """Provide BigQuery emulator host if available.

    This fixture checks if the BigQuery emulator is running (either in CI/CD
    or locally) and provides the host URL. Tests using this fixture will
    automatically skip if the emulator is not available.

    Yields:
        str | None: The BigQuery emulator host URL if available, None otherwise.
    """
    emulator_host = os.environ.get("BIGQUERY_EMULATOR_HOST")

    if not emulator_host:
        pytest.skip("BigQuery emulator not available (BIGQUERY_EMULATOR_HOST not set)")

    # Optionally verify the emulator is responsive
    try:
        import requests

        response = requests.get(f"{emulator_host}/discovery/v1/apis/bigquery/v2/rest", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"BigQuery emulator not responding at {emulator_host}")
    except Exception as e:
        pytest.skip(f"Cannot connect to BigQuery emulator: {e}")

    yield emulator_host


@pytest.fixture
def bigquery_project_id() -> str:
    """Provide the BigQuery project ID for testing.

    Returns:
        str: The project ID used for BigQuery emulator testing.
    """
    return "dqx-cicd"
