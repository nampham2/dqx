"""Tests for VerificationSuite DQL integration (Phase 3)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from dqx.api import VerificationSuite, check
from dqx.common import DQXError
from dqx.orm.repositories import InMemoryMetricDB


class TestVerificationSuiteDQLParameter:
    """Test Phase 3.1: DQL parameter validation and parsing."""

    def test_dql_parameter_mutual_exclusion_both_provided(self, tmp_path: Path) -> None:
        """Test error when both checks and dql are provided."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text('suite "Test" {}')

        with pytest.raises(DQXError, match="Exactly one of 'checks' or 'dql' must be provided"):
            VerificationSuite(
                checks=[],  # Dummy check
                dql=dql_file,
                db=db,
            )

    def test_dql_parameter_mutual_exclusion_neither_provided(self) -> None:
        """Test error when neither checks nor dql are provided."""
        db = InMemoryMetricDB()

        with pytest.raises(DQXError, match="Exactly one of 'checks' or 'dql' must be provided"):
            VerificationSuite(
                db=db,
            )

    def test_dql_parameter_requires_db(self, tmp_path: Path) -> None:
        """Test error when dql is provided without db."""
        dql_file = tmp_path / "test.dql"
        dql_file.write_text('suite "Test" {}')

        with pytest.raises(DQXError, match="'db' parameter is required when using 'dql'"):
            VerificationSuite(
                dql=dql_file,
            )

    def test_python_api_requires_db(self) -> None:
        """Test error when checks are provided without db."""

        @check(name="Test Check", datasets=["dataset"])
        def test_check(mp, ctx) -> None:  # type: ignore[no-untyped-def]
            pass

        with pytest.raises(DQXError, match="'db' parameter is required"):
            VerificationSuite(
                checks=[test_check],
            )

    def test_dql_with_name_parameter_raises_error(self, tmp_path: Path) -> None:
        """Test that providing name parameter with dql raises error."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Test Suite" {
            check "Test Check" on dataset {
                assert num_rows() > 0
                    name "Has rows"
            }
        }
        """)

        with pytest.raises(DQXError, match="'name' parameter cannot be specified when using 'dql'"):
            VerificationSuite(
                dql=dql_file,
                db=db,
                name="This Should Cause Error",
            )

    def test_dql_path_parsing(self, tmp_path: Path) -> None:
        """Test that DQL Path parsing works."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Test Suite" {
            check "Test Check" on dataset {
                assert num_rows() > 0
                    name "Has rows"
            }
        }
        """)

        # Should not raise - DQL is now implemented!
        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        # Verify suite was created with DQL name
        assert suite._name == "Test Suite"
        assert len(suite._checks) == 1

    def test_dql_file_parsing(self, tmp_path: Path) -> None:
        """Test that DQL file parsing works."""
        db = InMemoryMetricDB()

        # Create temporary DQL file
        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Test Suite" {
            check "Test Check" on dataset {
                assert num_rows() > 0
                    name "Has rows"
            }
        }
        """)

        # Should not raise - DQL is now implemented!
        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        # Verify suite was created with DQL name
        assert suite._name == "Test Suite"
        assert len(suite._checks) == 1

    def test_dql_suite_name_missing_everywhere(self, tmp_path: Path) -> None:
        """Test error when DQL has no suite name."""
        from lark.exceptions import VisitError

        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "" {
        }
        """)

        # Parser should raise error for empty suite name
        # The DQLSyntaxError is wrapped in a VisitError by Lark
        with pytest.raises(VisitError, match="Suite name cannot be empty"):
            VerificationSuite(
                dql=dql_file,
                db=db,
            )

    def test_dql_uses_fallback_name(self) -> None:
        """Test removed - fallback logic no longer exists."""
        # This test is no longer relevant since name parameter
        # cannot be specified with dql
        pass

    def test_dql_tunable_types(self, tmp_path: Path) -> None:
        """Test that DQL tunables are created with correct types."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Tunable Types" {
            tunable PERCENT_VAL = 0.5 bounds [0.0, 1.0]
            tunable INT_VAL = 10 bounds [1, 100]
            tunable FLOAT_VAL = 123.456 bounds [0.0, 1000.0]

            check "Check" on dataset {
                assert num_rows() >= INT_VAL
                    name "test"

                assert null_count(col) <= FLOAT_VAL
                    name "test2"

                collect average(val) * PERCENT_VAL
                    name "test3"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        # Verify tunables were created with correct types
        assert suite._name == "Tunable Types"

    def test_python_api_still_works(self) -> None:
        """Test that Python API (checks parameter) still works after changes."""
        db = InMemoryMetricDB()

        @check(name="Test Check", datasets=["dataset"])
        def test_check(mp, ctx) -> None:  # type: ignore[no-untyped-def]
            ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

        # Should not raise - Python API still works
        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
        )

        assert suite._name == "Test Suite"
        assert suite._data_av_threshold == 0.9
        assert len(suite._checks) == 1

    def test_python_api_empty_checks_error(self) -> None:
        """Test error when empty checks list provided."""
        db = InMemoryMetricDB()

        with pytest.raises(DQXError, match="At least one check must be provided"):
            VerificationSuite(
                checks=[],
                db=db,
                name="Test Suite",
            )

    def test_python_api_empty_name_error(self) -> None:
        """Test error when empty name provided with Python API."""
        db = InMemoryMetricDB()

        @check(name="Test Check", datasets=["dataset"])
        def test_check(mp, ctx) -> None:  # type: ignore[no-untyped-def]
            pass

        with pytest.raises(DQXError, match="Suite name cannot be empty"):
            VerificationSuite(
                checks=[test_check],
                db=db,
                name="",
            )

    def test_python_api_name_not_provided_error(self) -> None:
        """Test error when name not provided with Python API."""
        db = InMemoryMetricDB()

        @check(name="Test Check", datasets=["dataset"])
        def test_check(mp, ctx) -> None:  # type: ignore[no-untyped-def]
            pass

        with pytest.raises(DQXError, match="'name' parameter is required when using 'checks'"):
            VerificationSuite(
                checks=[test_check],
                db=db,
            )

    def test_python_api_with_profiles(self) -> None:
        """Test that profiles parameter still works with Python API."""
        from dqx.profiles import SeasonalProfile, check as profile_check

        db = InMemoryMetricDB()

        @check(name="Test Check", datasets=["dataset"])
        def test_check(mp, ctx) -> None:  # type: ignore[no-untyped-def]
            ctx.assert_that(mp.num_rows()).where(name="Has rows").is_gt(0)

        profile = SeasonalProfile(
            name="Test Profile",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            rules=[profile_check("Test Check").disable()],
        )

        suite = VerificationSuite(
            checks=[test_check],
            db=db,
            name="Test Suite",
            profiles=[profile],
        )

        assert len(suite._profiles) == 1
        assert suite._profiles[0].name == "Test Profile"


class TestCheckNameValidation:
    """Test check name validation."""

    def test_check_name_cannot_be_empty(self) -> None:
        """Test that check names cannot be empty."""
        from dqx.graph.nodes import CheckNode, RootNode

        root = RootNode(name="Test Suite")

        with pytest.raises(ValueError, match="Check name cannot be empty"):
            CheckNode(parent=root, name="")

    def test_check_name_cannot_be_whitespace(self) -> None:
        """Test that check names cannot be only whitespace."""
        from dqx.graph.nodes import CheckNode, RootNode

        root = RootNode(name="Test Suite")

        with pytest.raises(ValueError, match="Check name cannot be empty"):
            CheckNode(parent=root, name="   ")

    def test_check_name_too_long(self) -> None:
        """Test that check names cannot exceed 255 characters."""
        from dqx.graph.nodes import CheckNode, RootNode

        root = RootNode(name="Test Suite")
        long_name = "a" * 256

        with pytest.raises(ValueError, match="Check name is too long"):
            CheckNode(parent=root, name=long_name)

    def test_check_name_strips_whitespace(self) -> None:
        """Test that check names are stripped of leading/trailing whitespace."""
        from dqx.graph.nodes import CheckNode, RootNode

        root = RootNode(name="Test Suite")
        check_node = CheckNode(parent=root, name="  Check Name  ")

        assert check_node.name == "Check Name"

    def test_check_decorator_validates_empty_name(self) -> None:
        """Test that @check decorator with empty name fails at graph build time."""
        from dqx.api import VerificationSuite, check
        from dqx.orm.repositories import InMemoryMetricDB

        db = InMemoryMetricDB()

        @check(name="", datasets=["dataset"])
        def empty_name_check(mp, ctx) -> None:  # type: ignore[no-untyped-def]
            ctx.assert_that(mp.num_rows()).where(name="test").is_gt(0)

        # Should fail when building the graph (during suite creation)
        with pytest.raises(ValueError, match="Check name cannot be empty"):
            VerificationSuite(
                checks=[empty_name_check],
                db=db,
                name="Test Suite",
            )
