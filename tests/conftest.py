import logging
from collections.abc import Iterator

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
