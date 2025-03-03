import logging
import sys
from collections.abc import Iterator

import pytest
from rich.console import Console

from tests.fixtures.data_fixtures import commerce_data, commerce_table  # noqa: F401

LOG_FORMAT = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_NAME = "metasearch_core"


def _stream_handler() -> logging.Handler:
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    return handler


@pytest.fixture(autouse=True)
def run_around_tests() -> Iterator:
    # Use a single handler in tests
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = [_stream_handler()]
    print("\n")
    yield


@pytest.fixture(scope="session")
def console() -> Console:
    return Console()
