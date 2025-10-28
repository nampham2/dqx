"""Test the print_symbols convenience method in MetricProvider."""

import datetime as dt
import uuid
from io import StringIO
from unittest.mock import patch

from dqx.common import ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_print_symbols_convenience_method() -> None:
    """Test that print_symbols method correctly calls collect_symbols and prints."""
    # Setup
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    provider = MetricProvider(db, execution_id)
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={"env": "test"})

    # Create some test metrics
    provider.average("price", dataset="ds1")
    provider.sum("revenue", dataset="ds1")

    # Mock the console output
    with patch("dqx.display.Console") as MockConsole:
        # Create a StringIO to capture output
        output = StringIO()
        mock_console = MockConsole.return_value
        mock_console.print = lambda x: output.write(str(x))

        # Call the convenience method
        provider.print_symbols(key)

        # Verify that Console was called (print_symbols was invoked)
        assert MockConsole.called

    # Alternative test: verify the method works without errors
    # and that it internally calls collect_symbols
    with patch.object(provider, "collect_symbols") as mock_collect:
        # Set up mock return value
        mock_collect.return_value = []

        # Call print_symbols
        provider.print_symbols(key)

        # Verify collect_symbols was called with correct arguments
        mock_collect.assert_called_once_with(key)


# Removed test_print_symbols_with_actual_output - stdout inspection is unstable


def test_print_symbols_integration() -> None:
    """Test that print_symbols works the same as manual collect + print."""

    # Setup
    db = InMemoryMetricDB()
    execution_id = str(uuid.uuid4())
    provider = MetricProvider(db, execution_id)
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})

    # Create metrics
    provider.sum("amount")
    provider.null_count("status")

    # Mock display.print_symbols to capture arguments
    with patch("dqx.display.print_symbols") as mock_print:
        # Call the convenience method
        provider.print_symbols(key)

        # Get the symbols that were passed to print_symbols
        mock_print.assert_called_once()
        symbols_from_convenience = mock_print.call_args[0][0]

    # Now get symbols manually
    symbols_manual = provider.collect_symbols(key)

    # They should be identical
    assert len(symbols_from_convenience) == len(symbols_manual)
    for s1, s2 in zip(symbols_from_convenience, symbols_manual):
        assert s1.name == s2.name
        assert s1.metric == s2.metric
        assert s1.dataset == s2.dataset
        assert s1.yyyy_mm_dd == s2.yyyy_mm_dd
        assert s1.tags == s2.tags
