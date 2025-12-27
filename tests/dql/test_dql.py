"""Tests for complex DQL scenarios - Banking, Book E-Commerce, and Video Streaming."""

from pathlib import Path

import pytest

from dqx.dql import (
    DisableRule,
    ScaleRule,
    SetSeverityRule,
    Severity,
    parse,
)


class TestBankingTransactionsDQL:
    """Test banking transaction validation DQL parsing."""

    @pytest.fixture
    def banking_dql(self) -> str:
        return Path(__file__).parent.joinpath("banking_transactions.dql").read_text()

    def test_suite_metadata(self, banking_dql: str) -> None:
        """Test suite-level metadata parsing."""
        result = parse(banking_dql)
        assert result.name == "Banking Transaction Data Quality"
        assert result.availability_threshold == 0.95

    def test_tunable_constants(self, banking_dql: str) -> None:
        """Test tunable constants with bounds for RL optimization."""
        result = parse(banking_dql)
        assert len(result.constants) == 5

        # Find tunable constants
        tunables = [c for c in result.constants if c.tunable]
        assert len(tunables) == 4

        # Check MAX_NULL_RATE tunable
        max_null = next(c for c in result.constants if c.name == "MAX_NULL_RATE")
        assert max_null.tunable is True
        assert max_null.bounds is not None

        # Check exported constant
        exported = next(c for c in result.constants if c.name == "AMOUNT_VARIANCE_THRESHOLD")
        assert exported.export is True
        assert exported.tunable is False

    def test_check_count(self, banking_dql: str) -> None:
        """Test that at least 5 checks are defined."""
        result = parse(banking_dql)
        assert len(result.checks) >= 5

    def test_completeness_check_assertions(self, banking_dql: str) -> None:
        """Test completeness check has proper assertions."""
        result = parse(banking_dql)
        completeness = next(c for c in result.checks if c.name == "Completeness")

        assert completeness.datasets == ("transactions",)
        assert len(completeness.assertions) >= 4

        # Check for @required annotation
        required_assertions = [
            a for a in completeness.assertions if any(ann.name == "required" for ann in a.annotations)
        ]
        assert len(required_assertions) >= 1

        # Check for @cost annotation
        cost_assertions = [a for a in completeness.assertions if any(ann.name == "cost" for ann in a.annotations)]
        assert len(cost_assertions) >= 1
        cost_ann = next(ann for ann in cost_assertions[0].annotations if ann.name == "cost")
        assert "false_positive" in cost_ann.args
        assert "false_negative" in cost_ann.args

    def test_integrity_check_with_sampling(self, banking_dql: str) -> None:
        """Test integrity check includes sampling configuration."""
        result = parse(banking_dql)
        integrity = next(c for c in result.checks if c.name == "Integrity")

        sampled_assertions = [a for a in integrity.assertions if a.sample is not None]
        assert len(sampled_assertions) >= 1

        sample = sampled_assertions[0].sample
        assert sample is not None
        assert sample.is_percentage is True
        assert sample.seed == 42

    def test_multi_dataset_check(self, banking_dql: str) -> None:
        """Test checks spanning multiple datasets."""
        result = parse(banking_dql)
        financial = next(c for c in result.checks if c.name == "Financial Validity")
        assert len(financial.datasets) == 2
        assert "transactions" in financial.datasets
        assert "accounts" in financial.datasets

    def test_reconciliation_with_tolerance(self, banking_dql: str) -> None:
        """Test reconciliation check with tolerance modifier."""
        result = parse(banking_dql)
        recon = next(c for c in result.checks if c.name == "Reconciliation")

        tolerance_assertions = [a for a in recon.assertions if a.tolerance is not None]
        assert len(tolerance_assertions) >= 1
        assert tolerance_assertions[0].tolerance == 0.02

    def test_profiles(self, banking_dql: str) -> None:
        """Test profile definitions with rules."""
        result = parse(banking_dql)
        assert len(result.profiles) == 2

        # End of Month profile
        eom = next(p for p in result.profiles if p.name == "End of Month")
        assert eom.profile_type == "recurring"
        assert len(eom.rules) >= 2

        scale_rules = [r for r in eom.rules if isinstance(r, ScaleRule)]
        assert len(scale_rules) >= 1
        assert scale_rules[0].multiplier == 2.0

        # Holiday Season profile
        holiday = next(p for p in result.profiles if p.name == "Holiday Season")
        assert holiday.profile_type == "holiday"
        disable_rules = [r for r in holiday.rules if isinstance(r, DisableRule)]
        assert len(disable_rules) >= 1

    def test_severity_levels(self, banking_dql: str) -> None:
        """Test various severity levels are used."""
        result = parse(banking_dql)
        all_assertions = [a for c in result.checks for a in c.assertions]

        severities = {a.severity for a in all_assertions}
        assert Severity.P0 in severities
        assert Severity.P1 in severities
        assert Severity.P2 in severities

    def test_experimental_annotation(self, banking_dql: str) -> None:
        """Test @experimental annotation parsing."""
        result = parse(banking_dql)
        volume = next(c for c in result.checks if c.name == "Volume")

        experimental = [a for a in volume.assertions if any(ann.name == "experimental" for ann in a.annotations)]
        assert len(experimental) >= 1

    def test_between_condition(self, banking_dql: str) -> None:
        """Test between condition in assertions."""
        result = parse(banking_dql)
        volume = next(c for c in result.checks if c.name == "Volume")

        between_assertions = [a for a in volume.assertions if a.condition == "between"]
        assert len(between_assertions) >= 1

    def test_is_not_none_condition(self, banking_dql: str) -> None:
        """Test 'is not None' condition."""
        result = parse(banking_dql)
        financial = next(c for c in result.checks if c.name == "Financial Validity")

        not_none_assertions = [a for a in financial.assertions if a.keyword == "not None"]
        assert len(not_none_assertions) >= 1

    def test_duplicate_count_with_list(self, banking_dql: str) -> None:
        """Test duplicate_count with list of columns."""
        result = parse(banking_dql)
        integrity = next(c for c in result.checks if c.name == "Integrity")

        dup_assertions = [a for a in integrity.assertions if "duplicate_count" in a.expr.text]
        assert len(dup_assertions) >= 2

    def test_lag_parameter(self, banking_dql: str) -> None:
        """Test lag parameter in metrics."""
        result = parse(banking_dql)
        volume = next(c for c in result.checks if c.name == "Volume")

        lag_assertions = [a for a in volume.assertions if "lag=" in a.expr.text]
        assert len(lag_assertions) >= 1

    def test_n_parameter_stddev(self, banking_dql: str) -> None:
        """Test n parameter for stddev function."""
        result = parse(banking_dql)
        volume = next(c for c in result.checks if c.name == "Volume")

        n_assertions = [a for a in volume.assertions if "n=" in a.expr.text]
        assert len(n_assertions) >= 1

    def test_dataset_parameter(self, banking_dql: str) -> None:
        """Test dataset parameter in cross-dataset checks."""
        result = parse(banking_dql)
        recon = next(c for c in result.checks if c.name == "Reconciliation")

        dataset_assertions = [a for a in recon.assertions if "dataset=" in a.expr.text]
        assert len(dataset_assertions) >= 2

    def test_date_function_in_profile(self, banking_dql: str) -> None:
        """Test date function expressions in profiles."""
        result = parse(banking_dql)
        eom = next(p for p in result.profiles if p.name == "End of Month")

        assert "last_day_of_month" in str(eom.from_date.value)

    def test_set_severity_rule(self, banking_dql: str) -> None:
        """Test set severity rule in profile."""
        result = parse(banking_dql)
        eom = next(p for p in result.profiles if p.name == "End of Month")

        set_rules = [r for r in eom.rules if isinstance(r, SetSeverityRule)]
        assert len(set_rules) >= 1
        assert set_rules[0].severity == Severity.P3


class TestBookInventoryDQL:
    """Test book e-commerce inventory validation DQL parsing."""

    @pytest.fixture
    def inventory_dql(self) -> str:
        return Path(__file__).parent.joinpath("book_inventory.dql").read_text()

    def test_suite_metadata(self, inventory_dql: str) -> None:
        """Test suite-level metadata."""
        result = parse(inventory_dql)
        assert result.name == "Book E-Commerce Inventory Quality"
        assert result.availability_threshold == 0.9

    def test_check_count(self, inventory_dql: str) -> None:
        """Test minimum check count."""
        result = parse(inventory_dql)
        assert len(result.checks) >= 5

    def test_total_assertions(self, inventory_dql: str) -> None:
        """Test total assertion count meets requirement."""
        result = parse(inventory_dql)
        total = sum(len(c.assertions) for c in result.checks)
        assert total >= 7

    def test_catalog_completeness(self, inventory_dql: str) -> None:
        """Test catalog completeness check."""
        result = parse(inventory_dql)
        catalog = next(c for c in result.checks if c.name == "Catalog Completeness")

        assert len(catalog.assertions) >= 5
        assert catalog.datasets == ("books",)

        # Check tags
        all_tags: set[str] = set()
        for a in catalog.assertions:
            all_tags.update(a.tags)
        assert "completeness" in all_tags

    def test_inventory_health_count_values(self, inventory_dql: str) -> None:
        """Test count_values function usage."""
        result = parse(inventory_dql)
        health = next(c for c in result.checks if c.name == "Inventory Health")

        # Should have count_values assertion
        count_vals = [a for a in health.assertions if "count_values" in a.expr.text]
        assert len(count_vals) >= 1

    def test_pricing_validity_between(self, inventory_dql: str) -> None:
        """Test between condition in pricing check."""
        result = parse(inventory_dql)
        pricing = next(c for c in result.checks if c.name == "Pricing Validity")

        between_assertions = [a for a in pricing.assertions if a.condition == "between"]
        assert len(between_assertions) >= 1

    def test_sample_with_rows(self, inventory_dql: str) -> None:
        """Test sampling by row count."""
        result = parse(inventory_dql)
        uniqueness = next(c for c in result.checks if c.name == "Data Uniqueness")

        sampled = [a for a in uniqueness.assertions if a.sample is not None]
        assert len(sampled) >= 1
        sample = sampled[0].sample
        assert sample is not None
        assert sample.is_percentage is False
        assert sample.value == 5000
        assert sample.seed == 123

    def test_cross_table_consistency(self, inventory_dql: str) -> None:
        """Test cross-table consistency check."""
        result = parse(inventory_dql)
        consistency = next(c for c in result.checks if c.name == "Cross-Table Consistency")

        assert len(consistency.datasets) == 3
        assert "books" in consistency.datasets
        assert "inventory" in consistency.datasets
        assert "orders" in consistency.datasets

    def test_is_positive_condition(self, inventory_dql: str) -> None:
        """Test 'is positive' condition."""
        result = parse(inventory_dql)
        consistency = next(c for c in result.checks if c.name == "Cross-Table Consistency")

        positive_assertions = [a for a in consistency.assertions if a.keyword == "positive"]
        assert len(positive_assertions) >= 1

    def test_is_not_none_with_order_by(self, inventory_dql: str) -> None:
        """Test 'is not None' with order_by parameter."""
        result = parse(inventory_dql)
        consistency = next(c for c in result.checks if c.name == "Cross-Table Consistency")

        order_by_assertions = [
            a for a in consistency.assertions if "order_by=" in a.expr.text and a.keyword == "not None"
        ]
        assert len(order_by_assertions) >= 1

    def test_profiles_with_date_functions(self, inventory_dql: str) -> None:
        """Test profiles with date function expressions."""
        result = parse(inventory_dql)
        assert len(result.profiles) >= 2

        black_friday = next(p for p in result.profiles if p.name == "Black Friday")
        assert "nth_weekday" in str(black_friday.from_date.value)

        # Check disable assertion in check
        disable_rules = [r for r in black_friday.rules if isinstance(r, DisableRule)]
        assertion_disables = [r for r in disable_rules if r.target_type == "assertion"]
        assert len(assertion_disables) >= 1
        assert assertion_disables[0].in_check is not None

    def test_temporal_quality_check(self, inventory_dql: str) -> None:
        """Test temporal quality check."""
        result = parse(inventory_dql)
        temporal = next(c for c in result.checks if c.name == "Temporal Quality")

        assert len(temporal.assertions) >= 2
        assert temporal.datasets == ("books",)

    def test_unique_count_metric(self, inventory_dql: str) -> None:
        """Test unique_count metric usage."""
        result = parse(inventory_dql)
        uniqueness = next(c for c in result.checks if c.name == "Data Uniqueness")

        unique_assertions = [a for a in uniqueness.assertions if "unique_count" in a.expr.text]
        assert len(unique_assertions) >= 2

    def test_negative_count_metric(self, inventory_dql: str) -> None:
        """Test negative_count metric usage."""
        result = parse(inventory_dql)
        health = next(c for c in result.checks if c.name == "Inventory Health")

        negative_assertions = [a for a in health.assertions if "negative_count" in a.expr.text]
        assert len(negative_assertions) >= 1

    def test_minimum_maximum_metrics(self, inventory_dql: str) -> None:
        """Test minimum and maximum metrics."""
        result = parse(inventory_dql)
        pricing = next(c for c in result.checks if c.name == "Pricing Validity")

        min_assertions = [a for a in pricing.assertions if "minimum" in a.expr.text]
        max_assertions = [a for a in pricing.assertions if "maximum" in a.expr.text]
        assert len(min_assertions) >= 1
        assert len(max_assertions) >= 1


class TestVideoStreamingDQL:
    """Test video streaming data validation DQL parsing."""

    @pytest.fixture
    def streaming_dql(self) -> str:
        return Path(__file__).parent.joinpath("video_streaming.dql").read_text()

    def test_suite_metadata(self, streaming_dql: str) -> None:
        """Test suite metadata with default availability_threshold."""
        result = parse(streaming_dql)
        assert result.name == "Video Streaming Data Quality"
        # availability_threshold not specified, should be None (defaults to 90% at runtime)
        assert result.availability_threshold is None

    def test_tunable_count(self, streaming_dql: str) -> None:
        """Test tunable constants for RL."""
        result = parse(streaming_dql)
        tunables = [c for c in result.constants if c.tunable]
        assert len(tunables) >= 4

    def test_check_count(self, streaming_dql: str) -> None:
        """Test minimum check count."""
        result = parse(streaming_dql)
        assert len(result.checks) >= 5

    def test_catalog_completeness_assertions(self, streaming_dql: str) -> None:
        """Test catalog completeness check."""
        result = parse(streaming_dql)
        catalog = next(c for c in result.checks if c.name == "Catalog Completeness")
        assert len(catalog.assertions) >= 5

    def test_viewing_activity_lag_and_n_params(self, streaming_dql: str) -> None:
        """Test lag and n parameters in functions."""
        result = parse(streaming_dql)
        activity = next(c for c in result.checks if c.name == "Viewing Activity")

        # Check for lag parameter
        lag_assertions = [a for a in activity.assertions if "lag=" in a.expr.text]
        assert len(lag_assertions) >= 1

        # Check for n parameter (stddev)
        n_assertions = [a for a in activity.assertions if "n=" in a.expr.text]
        assert len(n_assertions) >= 1

    def test_user_engagement_tolerance(self, streaming_dql: str) -> None:
        """Test tolerance modifier usage."""
        result = parse(streaming_dql)
        engagement = next(c for c in result.checks if c.name == "User Engagement")

        tolerance_assertions = [a for a in engagement.assertions if a.tolerance is not None]
        assert len(tolerance_assertions) >= 1

    def test_streaming_quality_count_values_string(self, streaming_dql: str) -> None:
        """Test count_values with string values."""
        result = parse(streaming_dql)
        quality = next(c for c in result.checks if c.name == "Streaming Quality")

        count_vals = [a for a in quality.assertions if "count_values" in a.expr.text]
        assert len(count_vals) >= 2

    def test_content_integrity_duplicate_count(self, streaming_dql: str) -> None:
        """Test duplicate_count with list columns."""
        result = parse(streaming_dql)
        integrity = next(c for c in result.checks if c.name == "Content Integrity")

        dup_assertions = [a for a in integrity.assertions if "duplicate_count" in a.expr.text]
        assert len(dup_assertions) >= 2

    def test_cross_dataset_with_order_by(self, streaming_dql: str) -> None:
        """Test order_by parameter in first() function."""
        result = parse(streaming_dql)
        cross = next(c for c in result.checks if c.name == "Cross-Dataset Consistency")

        order_by_assertions = [a for a in cross.assertions if "order_by=" in a.expr.text]
        assert len(order_by_assertions) >= 1

    def test_multiple_profiles(self, streaming_dql: str) -> None:
        """Test multiple profile definitions."""
        result = parse(streaming_dql)
        assert len(result.profiles) >= 3

        # Check different profile types
        recurring_profiles = [p for p in result.profiles if p.profile_type == "recurring"]
        holiday_profiles = [p for p in result.profiles if p.profile_type == "holiday"]
        assert len(recurring_profiles) >= 1
        assert len(holiday_profiles) >= 2

    def test_set_severity_rule(self, streaming_dql: str) -> None:
        """Test set severity rule in profiles."""
        result = parse(streaming_dql)

        set_severity_rules = []
        for profile in result.profiles:
            for rule in profile.rules:
                if isinstance(rule, SetSeverityRule):
                    set_severity_rules.append(rule)

        assert len(set_severity_rules) >= 1
        assert any(r.severity == Severity.P3 for r in set_severity_rules)

    def test_variance_metric(self, streaming_dql: str) -> None:
        """Test variance metric usage."""
        result = parse(streaming_dql)
        integrity = next(c for c in result.checks if c.name == "Content Integrity")

        variance_assertions = [a for a in integrity.assertions if "variance" in a.expr.text]
        assert len(variance_assertions) >= 1

    def test_average_metric(self, streaming_dql: str) -> None:
        """Test average metric usage."""
        result = parse(streaming_dql)
        quality = next(c for c in result.checks if c.name == "Streaming Quality")

        avg_assertions = [a for a in quality.assertions if "average" in a.expr.text]
        assert len(avg_assertions) >= 1

    def test_sum_metric_with_is_positive(self, streaming_dql: str) -> None:
        """Test sum metric with is positive condition."""
        result = parse(streaming_dql)
        cross = next(c for c in result.checks if c.name == "Cross-Dataset Consistency")

        sum_positive = [a for a in cross.assertions if "sum" in a.expr.text and a.keyword == "positive"]
        assert len(sum_positive) >= 1

    def test_date_offset_in_profile(self, streaming_dql: str) -> None:
        """Test date offset expressions in profiles."""
        result = parse(streaming_dql)
        new_release = next(p for p in result.profiles if p.name == "New Release Week")

        assert new_release.to_date.offset == 7

    def test_disable_assertion_in_check(self, streaming_dql: str) -> None:
        """Test disable assertion targeting specific check."""
        result = parse(streaming_dql)
        holiday = next(p for p in result.profiles if p.name == "Holiday Binge Season")

        disable_rules = [r for r in holiday.rules if isinstance(r, DisableRule)]
        assertion_disables = [r for r in disable_rules if r.target_type == "assertion"]
        assert len(assertion_disables) >= 1
        assert assertion_disables[0].in_check == "Viewing Activity"


class TestAllDQLScenariosIntegration:
    """Integration tests across all DQL scenarios."""

    @pytest.fixture
    def all_dqls(self) -> list[str]:
        base_path = Path(__file__).parent
        return [
            base_path.joinpath("banking_transactions.dql").read_text(),
            base_path.joinpath("book_inventory.dql").read_text(),
            base_path.joinpath("video_streaming.dql").read_text(),
        ]

    def test_total_assertions_minimum(self, all_dqls: list[str]) -> None:
        """Test that total assertions across all suites meets minimum of 20."""
        total_assertions = 0
        for dql in all_dqls:
            result = parse(dql)
            total_assertions += sum(len(c.assertions) for c in result.checks)

        assert total_assertions >= 20, f"Expected at least 20 assertions, got {total_assertions}"

    def test_all_suites_parse_successfully(self, all_dqls: list[str]) -> None:
        """Test all DQL files parse without errors."""
        for dql in all_dqls:
            result = parse(dql)
            assert result is not None
            assert result.name is not None
            assert len(result.checks) >= 5

    def test_feature_coverage(self, all_dqls: list[str]) -> None:
        """Test comprehensive feature coverage across all suites."""
        all_features = {
            "tunable_constants": False,
            "export_constants": False,
            "required_annotation": False,
            "experimental_annotation": False,
            "cost_annotation": False,
            "sample_percent": False,
            "sample_rows": False,
            "tolerance": False,
            "between_condition": False,
            "is_positive": False,
            "is_not_none": False,
            "lag_parameter": False,
            "n_parameter": False,
            "order_by_parameter": False,
            "dataset_parameter": False,
            "duplicate_count_list": False,
            "scale_rule": False,
            "disable_rule": False,
            "set_severity_rule": False,
            "date_function": False,
        }

        for dql in all_dqls:
            result = parse(dql)

            # Check constants
            for const in result.constants:
                if const.tunable:
                    all_features["tunable_constants"] = True
                if const.export:
                    all_features["export_constants"] = True

            # Check assertions
            for check in result.checks:
                for assertion in check.assertions:
                    for ann in assertion.annotations:
                        if ann.name == "required":
                            all_features["required_annotation"] = True
                        if ann.name == "experimental":
                            all_features["experimental_annotation"] = True
                        if ann.name == "cost":
                            all_features["cost_annotation"] = True

                    if assertion.sample:
                        if assertion.sample.is_percentage:
                            all_features["sample_percent"] = True
                        else:
                            all_features["sample_rows"] = True

                    if assertion.tolerance:
                        all_features["tolerance"] = True

                    if assertion.condition == "between":
                        all_features["between_condition"] = True

                    if assertion.keyword == "positive":
                        all_features["is_positive"] = True
                    if assertion.keyword == "not None":
                        all_features["is_not_none"] = True

                    expr_text = assertion.expr.text
                    if "lag=" in expr_text:
                        all_features["lag_parameter"] = True
                    if "n=" in expr_text:
                        all_features["n_parameter"] = True
                    if "order_by=" in expr_text:
                        all_features["order_by_parameter"] = True
                    if "dataset=" in expr_text:
                        all_features["dataset_parameter"] = True
                    if "duplicate_count([" in expr_text:
                        all_features["duplicate_count_list"] = True

            # Check profiles
            for profile in result.profiles:
                date_funcs = ["nth_weekday", "last_day_of_month", "today"]
                if any(func in str(profile.from_date.value) for func in date_funcs):
                    all_features["date_function"] = True

                for rule in profile.rules:
                    if isinstance(rule, ScaleRule):
                        all_features["scale_rule"] = True
                    if isinstance(rule, DisableRule):
                        all_features["disable_rule"] = True
                    if isinstance(rule, SetSeverityRule):
                        all_features["set_severity_rule"] = True

        # Assert all features are covered
        missing_features = [f for f, covered in all_features.items() if not covered]
        assert not missing_features, f"Missing feature coverage: {missing_features}"

    def test_all_severity_levels_used(self, all_dqls: list[str]) -> None:
        """Test that all severity levels are used across the suites."""
        all_severities = set()
        for dql in all_dqls:
            result = parse(dql)
            for check in result.checks:
                for assertion in check.assertions:
                    all_severities.add(assertion.severity)

        assert Severity.P0 in all_severities, "P0 severity not used"
        assert Severity.P1 in all_severities, "P1 severity not used"
        assert Severity.P2 in all_severities, "P2 severity not used"
        assert Severity.P3 in all_severities, "P3 severity not used"

    def test_tags_diversity(self, all_dqls: list[str]) -> None:
        """Test that a variety of tags are used."""
        all_tags: set[str] = set()
        for dql in all_dqls:
            result = parse(dql)
            for check in result.checks:
                for assertion in check.assertions:
                    all_tags.update(assertion.tags)

        # Expect at least 10 different tags
        assert len(all_tags) >= 10, f"Expected at least 10 different tags, got {len(all_tags)}"

        # Check for essential tag categories
        expected_tag_patterns = ["completeness", "integrity", "volume", "quality", "validity"]
        for pattern in expected_tag_patterns:
            assert any(pattern in tag for tag in all_tags), f"No tag containing '{pattern}'"

    def test_multi_dataset_checks_present(self, all_dqls: list[str]) -> None:
        """Test that multi-dataset checks are present in each suite."""
        for dql in all_dqls:
            result = parse(dql)
            multi_dataset_checks = [c for c in result.checks if len(c.datasets) > 1]
            assert len(multi_dataset_checks) >= 1, f"No multi-dataset checks in {result.name}"

    def test_profile_types_coverage(self, all_dqls: list[str]) -> None:
        """Test that both profile types (holiday and recurring) are used."""
        all_profile_types = set()
        for dql in all_dqls:
            result = parse(dql)
            for profile in result.profiles:
                all_profile_types.add(profile.profile_type)

        assert "holiday" in all_profile_types, "No holiday profiles found"
        assert "recurring" in all_profile_types, "No recurring profiles found"

    def test_check_naming_convention(self, all_dqls: list[str]) -> None:
        """Test that assertions follow naming convention (dataset.check.assertion)."""
        named_assertions = 0
        convention_compliant = 0

        for dql in all_dqls:
            result = parse(dql)
            for check in result.checks:
                for assertion in check.assertions:
                    if assertion.name:
                        named_assertions += 1
                        # Check if name contains dots (convention: dataset.check.assertion)
                        if "." in assertion.name:
                            convention_compliant += 1

        # At least 80% should follow naming convention
        compliance_rate = convention_compliant / named_assertions if named_assertions > 0 else 0
        assert compliance_rate >= 0.8, f"Only {compliance_rate:.0%} assertions follow naming convention"

    def test_metrics_diversity(self, all_dqls: list[str]) -> None:
        """Test that a variety of metrics are used."""
        all_metrics = set()
        metric_patterns = [
            "num_rows",
            "null_count",
            "average",
            "sum",
            "minimum",
            "maximum",
            "variance",
            "unique_count",
            "duplicate_count",
            "negative_count",
            "count_values",
            "first",
            "day_over_day",
            "stddev",
        ]

        for dql in all_dqls:
            result = parse(dql)
            for check in result.checks:
                for assertion in check.assertions:
                    for pattern in metric_patterns:
                        if pattern in assertion.expr.text:
                            all_metrics.add(pattern)

        # Expect at least 10 different metrics
        assert len(all_metrics) >= 10, f"Expected at least 10 different metrics, got {len(all_metrics)}"


class TestBookOrdersDQLWithImport:
    """Test book orders DQL that imports from book_inventory.dql."""

    @pytest.fixture
    def orders_dql(self) -> str:
        return Path(__file__).parent.joinpath("book_orders.dql").read_text()

    def test_suite_metadata(self, orders_dql: str) -> None:
        """Test suite-level metadata."""
        result = parse(orders_dql)
        assert result.name == "Book Order Processing Quality"
        assert result.availability_threshold == 0.92

    def test_import_statement(self, orders_dql: str) -> None:
        """Test import statement is parsed correctly."""
        result = parse(orders_dql)
        assert len(result.imports) == 1
        assert result.imports[0].path == "book_inventory.dql"
        assert result.imports[0].alias is None
        assert result.imports[0].names is None

    def test_check_count(self, orders_dql: str) -> None:
        """Test minimum check count."""
        result = parse(orders_dql)
        assert len(result.checks) >= 5

    def test_uses_imported_constant(self, orders_dql: str) -> None:
        """Test that assertion references imported PRICE_VOLATILITY_THRESHOLD constant."""
        result = parse(orders_dql)
        pricing = next(c for c in result.checks if c.name == "Order Pricing")

        # Find assertion that uses the imported constant in threshold
        imported_const_assertions = [
            a for a in pricing.assertions if a.threshold and "PRICE_VOLATILITY_THRESHOLD" in a.threshold.text
        ]
        assert len(imported_const_assertions) >= 1

    def test_local_constants(self, orders_dql: str) -> None:
        """Test local constants are defined correctly."""
        result = parse(orders_dql)

        # Should have local constants plus imported ones don't count in constants list
        local_constants = {c.name for c in result.constants}
        assert "MIN_DAILY_ORDERS" in local_constants
        assert "MAX_REFUND_RATE" in local_constants
        assert "ORDER_VALUE_THRESHOLD" in local_constants

        # Check tunable
        min_orders = next(c for c in result.constants if c.name == "MIN_DAILY_ORDERS")
        assert min_orders.tunable is True

        # Check export
        order_value = next(c for c in result.constants if c.name == "ORDER_VALUE_THRESHOLD")
        assert order_value.export is True

    def test_order_completeness_check(self, orders_dql: str) -> None:
        """Test order completeness check."""
        result = parse(orders_dql)
        completeness = next(c for c in result.checks if c.name == "Order Completeness")

        assert completeness.datasets == ("orders",)
        assert len(completeness.assertions) >= 4

        # Check @required annotation
        required = [a for a in completeness.assertions if any(ann.name == "required" for ann in a.annotations)]
        assert len(required) >= 1

    def test_order_volume_experimental(self, orders_dql: str) -> None:
        """Test experimental annotation in order volume check."""
        result = parse(orders_dql)
        volume = next(c for c in result.checks if c.name == "Order Volume")

        experimental = [a for a in volume.assertions if any(ann.name == "experimental" for ann in a.annotations)]
        assert len(experimental) >= 1

    def test_order_integrity_sampling(self, orders_dql: str) -> None:
        """Test sampling in order integrity check."""
        result = parse(orders_dql)
        integrity = next(c for c in result.checks if c.name == "Order Integrity")

        sampled = [a for a in integrity.assertions if a.sample is not None]
        assert len(sampled) >= 1
        sample = sampled[0].sample
        assert sample is not None
        assert sample.seed == 42

    def test_cross_dataset_check(self, orders_dql: str) -> None:
        """Test cross-dataset check with orders and books."""
        result = parse(orders_dql)
        consistency = next(c for c in result.checks if c.name == "Order-Book Consistency")

        assert len(consistency.datasets) == 2
        assert "orders" in consistency.datasets
        assert "books" in consistency.datasets

        # Check @cost annotation
        cost_assertions = [a for a in consistency.assertions if any(ann.name == "cost" for ann in a.annotations)]
        assert len(cost_assertions) >= 1

    def test_profile(self, orders_dql: str) -> None:
        """Test profile definition."""
        result = parse(orders_dql)
        assert len(result.profiles) >= 1

        holiday = next(p for p in result.profiles if p.name == "Holiday Shopping")
        assert holiday.profile_type == "holiday"

        scale_rules = [r for r in holiday.rules if isinstance(r, ScaleRule)]
        assert len(scale_rules) >= 2

    def test_total_assertions(self, orders_dql: str) -> None:
        """Test total assertion count."""
        result = parse(orders_dql)
        total = sum(len(c.assertions) for c in result.checks)
        assert total >= 14
