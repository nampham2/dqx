"""Tests to verify all DQL fixture files parse successfully with updated grammar."""

from __future__ import annotations

from pathlib import Path


from dqx.dql import parse_file


class TestDQLFixtureParsing:
    """Test that all DQL fixture files parse successfully."""

    def test_banking_transactions_dql_parses(self) -> None:
        """Test banking_transactions.dql parses with numeric tunables."""
        dql_path = Path(__file__).parent / "banking_transactions.dql"
        suite = parse_file(dql_path)

        assert suite.name == "Banking Transaction Data Quality"
        assert len(suite.tunables) == 5
        # Verify all tunables have numeric bounds
        for tunable in suite.tunables:
            assert tunable.bounds is not None
            assert len(tunable.bounds) == 2

    def test_book_inventory_dql_parses(self) -> None:
        """Test book_inventory.dql parses with numeric tunables."""
        dql_path = Path(__file__).parent / "book_inventory.dql"
        suite = parse_file(dql_path)

        assert suite.name == "Book E-Commerce Inventory Quality"
        assert len(suite.tunables) == 4
        for tunable in suite.tunables:
            assert tunable.bounds is not None
            assert len(tunable.bounds) == 2

    def test_book_orders_dql_parses(self) -> None:
        """Test book_orders.dql parses with numeric tunables."""
        dql_path = Path(__file__).parent / "book_orders.dql"
        suite = parse_file(dql_path)

        assert suite.name == "Book Order Processing Quality"
        assert len(suite.tunables) == 4
        for tunable in suite.tunables:
            assert tunable.bounds is not None
            assert len(tunable.bounds) == 2

    def test_video_streaming_dql_parses(self) -> None:
        """Test video_streaming.dql parses with numeric tunables."""
        dql_path = Path(__file__).parent / "video_streaming.dql"
        suite = parse_file(dql_path)

        assert suite.name == "Video Streaming Data Quality"
        assert len(suite.tunables) == 5
        for tunable in suite.tunables:
            assert tunable.bounds is not None
            assert len(tunable.bounds) == 2

    def test_commerce_suite_dql_parses(self) -> None:
        """Test commerce_suite.dql parses (no tunables)."""
        dql_path = Path(__file__).parent / "commerce_suite.dql"
        suite = parse_file(dql_path)

        assert suite.name == "Simple test suite"
        assert len(suite.tunables) == 0

    def test_metric_collection_dql_parses(self) -> None:
        """Test metric_collection.dql parses (no tunables)."""
        dql_path = Path(__file__).parent / "metric_collection.dql"
        suite = parse_file(dql_path)

        assert suite.name == "Metric Collection Example"
        # May or may not have tunables, just verify it parses

    def test_all_fixture_files_parse_successfully(self) -> None:
        """Regression test: all DQL fixture files should parse without errors."""
        dql_dir = Path(__file__).parent
        dql_files = list(dql_dir.glob("*.dql"))

        assert len(dql_files) > 0, "No DQL files found in fixtures directory"

        for dql_file in dql_files:
            # Should not raise any exception
            suite = parse_file(dql_file)
            assert suite.name is not None
            # Verify all tunables have valid bounds
            for tunable in suite.tunables:
                assert tunable.bounds is not None
                assert len(tunable.bounds) == 2
                # Verify bounds are Expr objects with text attribute
                assert hasattr(tunable.bounds[0], "text")
                assert hasattr(tunable.bounds[1], "text")
                # Verify value is Expr object with text attribute
                assert hasattr(tunable.value, "text")
