"""E2E tests for DQL execution through VerificationSuite."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from dqx.api import VerificationSuite
from dqx.orm.repositories import InMemoryMetricDB


class TestDQLVerificationSuiteE2E:
    """End-to-end tests for DQL parameter in VerificationSuite."""

    def test_simple_dql_file(self, tmp_path: Path) -> None:
        """Test that a simple DQL file can be parsed and creates a suite."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Simple Suite" {
            check "Basic Check" on dataset {
                assert num_rows() > 0
                    name "Has rows"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert suite._name == "Simple Suite"
        assert len(suite._checks) == 1

    def test_dql_file_parsing(self, tmp_path: Path) -> None:
        """Test that DQL files can be loaded."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "File Suite" {
            availability_threshold 85%

            check "Check One" on ds1 {
                assert num_rows() >= 10
                    name "Minimum rows"
                    severity P1
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert suite._name == "File Suite"
        assert suite._data_av_threshold == 0.85

    def test_dql_with_multiple_checks(self, tmp_path: Path) -> None:
        """Test DQL with multiple checks."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Multi Check Suite" {
            check "Check A" on ds1 {
                assert num_rows() > 0
                    name "A has rows"
            }

            check "Check B" on ds2 {
                assert num_rows() > 0
                    name "B has rows"
            }

            check "Check C" on ds1, ds2 {
                assert num_rows(dataset=ds1) > 0
                    name "Cross dataset check"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert len(suite._checks) == 3

    @pytest.mark.skip(reason="Tunable discovery from DQL not yet implemented")
    def test_dql_with_tunables(self, tmp_path: Path) -> None:
        """Test that tunables defined in DQL are discovered when used."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Tunable Suite" {
            tunable MIN_ROWS = 10 bounds [1, 100]

            check "Check" on dataset {
                assert num_rows() >= MIN_ROWS
                    name "Has minimum rows"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        # Only tunables that are referenced in expressions are discovered
        assert "MIN_ROWS" in suite._tunables
        assert suite._tunables["MIN_ROWS"].value == 10

    def test_dql_with_profiles_from_api(self, tmp_path: Path) -> None:
        """Test that profiles can be passed alongside DQL."""
        from dqx.profiles import SeasonalProfile, check as profile_check

        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Profile Suite" {
            check "My Check" on dataset {
                assert num_rows() > 0
                    name "Has rows"
            }
        }
        """)

        profile = SeasonalProfile(
            name="Test Profile",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            rules=[profile_check("My Check").disable()],
        )

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
            profiles=[profile],
        )

        assert len(suite._profiles) == 1
        assert suite._profiles[0].name == "Test Profile"

    @pytest.mark.skip(reason="Tunable discovery from DQL not yet implemented")
    def test_dql_with_config_file(self, tmp_path: Path) -> None:
        """Test DQL with YAML config for tunable override."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Config Suite" {
            tunable MAX_VALUE = 50.0 bounds [0.0, 100.0]

            check "Check" on dataset {
                assert num_rows() <= MAX_VALUE
                    name "Row count limit"
            }
        }
        """)

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
tunables:
  MAX_VALUE: 75.0
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
            config=config_file,
        )

        # Config should override DQL tunable value
        assert "MAX_VALUE" in suite._tunables
        assert suite._tunables["MAX_VALUE"].value == 75.0

    def test_dql_assertions_with_all_condition_types(self, tmp_path: Path) -> None:
        """Test DQL with various condition types."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Conditions Suite" {
            check "Various Conditions" on dataset {
                assert num_rows() > 10
                    name "Greater than"

                assert num_rows() >= 5
                    name "Greater or equal"

                assert null_count(col) < 5
                    name "Less than"

                assert null_count(col) <= 10
                    name "Less or equal"

                assert sum(value) == 100
                    name "Equal to"
                    tolerance 0.01

                assert average(value) != 0
                    name "Not equal"

                assert num_rows() between 10 and 100
                    name "Between"

                assert sum(profit) is positive
                    name "Is positive"

                assert sum(loss) is negative
                    name "Is negative"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        # Just verify it parses and builds successfully
        assert suite._name == "Conditions Suite"
        assert len(suite._checks) == 1

    def test_dql_with_tags_and_severity(self, tmp_path: Path) -> None:
        """Test DQL assertions with tags and severity levels."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Tags Suite" {
            check "Tagged Check" on dataset {
                assert num_rows() > 0
                    name "Critical check"
                    severity P0
                    tags [critical, production]

                assert null_count(id) == 0
                    name "Medium priority"
                    severity P2
                    tags [data_quality]
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert len(suite._checks) == 1

    def test_dql_with_annotations(self, tmp_path: Path) -> None:
        """Test DQL with @experimental, @required, and @cost annotations."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Annotations Suite" {
            check "Annotated" on dataset {
                @experimental
                assert num_rows() > 100
                    name "Experimental check"

                @required
                assert null_count(id) == 0
                    name "Required check"

                @cost(false_positive=5, false_negative=10)
                assert average(value) > 50
                    name "Costly check"

                @experimental
                @required
                @cost(false_positive=1, false_negative=100)
                assert sum(amount) > 1000
                    name "All annotations"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert len(suite._checks) == 1

    def test_dql_with_collection_statements(self, tmp_path: Path) -> None:
        """Test DQL with collect statements (noop assertions)."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Collections Suite" {
            check "With Collections" on dataset {
                assert num_rows() > 0
                    name "Regular assertion"

                collect average(value)
                    name "Average value metric"

                collect first(timestamp)
                    name "Latest timestamp"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert len(suite._checks) == 1

    def test_dql_name_takes_precedence_over_parameter(self) -> None:
        """Test removed - name parameter cannot be used with dql."""
        # This test is no longer relevant since name parameter
        # raises error when used with dql
        pass

    def test_dql_availability_threshold_override(self, tmp_path: Path) -> None:
        """Test that DQL availability_threshold overrides parameter."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Threshold Suite" {
            availability_threshold 75%

            check "Check" on dataset {
                assert num_rows() > 0
                    name "test"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
            data_av_threshold=0.9,
        )

        assert suite._data_av_threshold == 0.75

    def test_dql_uses_real_commerce_suite(self) -> None:
        """Test using the real commerce_suite.dql file."""
        db = InMemoryMetricDB()

        dql_file = Path(__file__).parent.parent / "dql" / "commerce_suite.dql"

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert suite._name == "Simple test suite"
        assert suite._data_av_threshold == 0.8
        assert len(suite._checks) == 6  # 6 checks in commerce_suite.dql

    def test_dql_with_count_values(self, tmp_path: Path) -> None:
        """Test DQL with count_values function."""
        db = InMemoryMetricDB()

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "Count Values Suite" {
            check "Count Check" on dataset {
                assert count_values(status, "active") > 0
                    name "Has active records"

                assert count_values(flag, 1) == 0
                    name "No flags set"
            }
        }
        """)

        suite = VerificationSuite(
            dql=dql_file,
            db=db,
        )

        assert len(suite._checks) == 1
