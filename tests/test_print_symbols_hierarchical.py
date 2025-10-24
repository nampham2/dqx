"""Test hierarchical display of symbols in print_symbols."""

import datetime as dt
from io import StringIO
from unittest.mock import patch

from returns.result import Success
from rich.console import Console

from dqx.common import ResultKey, SymbolInfo
from dqx.display import print_symbols
from dqx.orm.repositories import InMemoryMetricDB


def test_print_symbols_with_hierarchical_display() -> None:
    """Test that print_symbols can show parent-child relationships."""
    # GIVEN: A metric provider with extended metrics

    from dqx.api import VerificationSuite, check
    from dqx.datasource import DuckRelationDataSource

    db = InMemoryMetricDB()

    from dqx.api import Context
    from dqx.provider import MetricProvider as MP

    @check(name="Test Check")
    def test_check(mp: MP, ctx: Context) -> None:
        base = mp.maximum("tax")
        dod = mp.ext.day_over_day(base)
        ctx.assert_that(dod).where(name="DoD check").is_positive()

    # Create a suite and run it
    suite = VerificationSuite([test_check], db, "Test Suite")

    # Create a simple data source using Arrow
    import pyarrow as pa

    data = pa.table(
        {
            "tax": [100.0, 200.0, 300.0],
            "yyyy_mm_dd": [
                dt.date(2024, 10, 22),
                dt.date(2024, 10, 23),
                dt.date(2024, 10, 24),
            ],
        }
    )
    ds = DuckRelationDataSource.from_arrow(data)

    key = ResultKey(yyyy_mm_dd=dt.date(2024, 10, 24), tags={})
    suite.run({"test_table": ds}, key, enable_plugins=False)

    # Collect all symbols
    symbol_infos = suite.collect_symbols()

    # WHEN: Printing symbols with show_dependencies=True
    with patch("sys.stdout", new=StringIO()) as fake_out:
        console = Console(file=fake_out, force_terminal=True, width=150)
        with patch("dqx.display.Console", return_value=console):
            print_symbols(symbol_infos, show_dependencies=True)
        output = fake_out.getvalue()

    # THEN: Output should show hierarchical structure
    # The base metric should be shown without indentation
    assert "maximum(tax)" in output
    # The extended metric should be indented (with 2 spaces)
    assert "  day_over_day(maximum(tax))" in output
    # The lag dependency should be indented (with 2 spaces as a child of x_1)
    assert "  lag(1)(x_1)" in output


def test_print_symbols_without_dependencies() -> None:
    """Test that print_symbols works normally without show_dependencies."""
    # GIVEN: Symbol infos
    symbols = [
        SymbolInfo(
            name="x_1",
            metric="maximum(tax)",
            dataset=None,
            value=Success(100.0),
            yyyy_mm_dd=dt.date(2024, 10, 24),
            suite="Test Suite",
            tags={},
        ),
        SymbolInfo(
            name="x_2",
            metric="day_over_day(maximum(tax))",
            dataset=None,
            value=Success(10.0),
            yyyy_mm_dd=dt.date(2024, 10, 24),
            suite="Test Suite",
            tags={},
        ),
    ]

    # WHEN: Printing symbols without show_dependencies
    with patch("sys.stdout", new=StringIO()) as fake_out:
        console = Console(file=fake_out, force_terminal=True, width=150)
        with patch("dqx.display.Console", return_value=console):
            print_symbols(symbols)
        output = fake_out.getvalue()

    # THEN: Output should be flat (no indentation)
    lines = output.strip().split("\n")
    # Check that metrics are not indented
    metric_lines = [line for line in lines if "maximum(tax)" in line]
    for line in metric_lines:
        # The line should not start with spaces (ignoring table borders)
        content_start = line.find("│") + 1 if "│" in line else 0
        if content_start > 0:
            content = line[content_start:].lstrip()
            assert not content.startswith(" " * 2)  # No indentation
