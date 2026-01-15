"""Tests for DQL parser."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from dqx.dql import (
    DisableRule,
    ScaleRule,
    SetSeverityRule,
    Severity,
    Suite,
    parse,
)
from dqx.dql.errors import DQLSyntaxError


class TestBasicSuite:
    """Test basic suite parsing."""

    def test_minimal_suite(self) -> None:
        source = """
        suite "Test Suite" {
        }
        """
        result = parse(source)
        assert isinstance(result, Suite)
        assert result.name == "Test Suite"
        assert result.checks == ()
        assert result.profiles == ()
        assert result.tunables == ()

    def test_suite_with_availability_threshold(self) -> None:
        source = """
        suite "Test Suite" {
            availability_threshold 80%
        }
        """
        result = parse(source)
        assert result.availability_threshold == 0.8


class TestChecks:
    """Test check parsing."""

    def test_simple_check(self) -> None:
        source = """
        suite "Test" {
            check "Completeness" on orders {
                assert num_rows() > 0
            }
        }
        """
        result = parse(source)
        assert len(result.checks) == 1
        check = result.checks[0]
        assert check.name == "Completeness"
        assert check.datasets == ("orders",)
        assert len(check.assertions) == 1

    def test_check_multiple_datasets(self) -> None:
        source = """
        suite "Test" {
            check "Cross-Dataset" on orders, returns {
                assert num_rows() > 0
            }
        }
        """
        result = parse(source)
        check = result.checks[0]
        assert check.datasets == ("orders", "returns")

    def test_escaped_dataset_name(self) -> None:
        source = """
        suite "Test" {
            check "Test" on `from` {
                assert num_rows() > 0
            }
        }
        """
        result = parse(source)
        check = result.checks[0]
        assert check.datasets == ("from",)


class TestAssertions:
    """Test assertion parsing."""

    def test_comparison_operators(self) -> None:
        ops = [">", ">=", "<", "<=", "==", "!="]
        for op in ops:
            source = f"""
            suite "Test" {{
                check "Test" on t {{
                    assert num_rows() {op} 100
                }}
            }}
            """
            result = parse(source)
            assertion = result.checks[0].assertions[0]
            assert assertion.condition == op

    def test_between_condition(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price) between 10 and 500
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.condition == "between"
        assert assertion.threshold is not None
        assert assertion.threshold_upper is not None

    def test_is_positive(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert sum(amount) is positive
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.condition == "is"
        assert assertion.keyword == "positive"

    def test_is_negative(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert sum(loss) is negative
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.keyword == "negative"

    def test_assertion_with_name(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0
                    name "At least one row"
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.name == "At least one row"

    def test_assertion_with_severity(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0 severity P0
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.severity == Severity.P0

    def test_assertion_with_tolerance(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price) == 100 tolerance 0.05
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.tolerance == 0.05

    def test_assertion_with_tags(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0 tags [volume, critical]
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.tags == ("volume", "critical")

    # Sampling support removed - no longer part of DQL spec
    # def test_assertion_with_sample_percent(self) -> None:
    # def test_assertion_with_sample_rows(self) -> None:
    # def test_assertion_with_sample_and_seed(self) -> None:

    def test_assertion_all_modifiers(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price) > 0
                    name "Price positive"
                    severity P0
                    tags [pricing, critical]
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.name == "Price positive"
        assert assertion.severity == Severity.P0
        assert assertion.tags == ("pricing", "critical")


class TestCollections:
    """Test collection statement parsing."""

    def test_simple_collect(self) -> None:
        """Test basic collect statement."""
        source = """
        suite "Test" {
            check "Metrics" on orders {
                collect num_rows()
                    name "row_count"
            }
        }
        """
        result = parse(source)
        assert len(result.checks) == 1
        check = result.checks[0]
        assert len(check.assertions) == 1

        from dqx.dql import Collection

        statement = check.assertions[0]
        assert isinstance(statement, Collection)
        assert statement.name == "row_count"
        assert statement.expr.text == "num_rows()"
        assert statement.severity == Severity.P1  # Default

    def test_collect_with_modifiers(self) -> None:
        """Test collect with severity and tags."""
        source = """
        suite "Test" {
            check "Metrics" on orders {
                collect average(price)
                    name "avg_price"
                    severity P0
                    tags [pricing, metrics]
            }
        }
        """
        result = parse(source)
        from dqx.dql import Collection

        collection = result.checks[0].assertions[0]
        assert isinstance(collection, Collection)
        assert collection.name == "avg_price"
        assert collection.severity == Severity.P0
        assert collection.tags == ("pricing", "metrics")

    def test_collect_with_annotations(self) -> None:
        """Test collect with @experimental and @cost."""
        source = """
        suite "Test" {
            check "Metrics" on orders {
                @experimental
                @cost(fp=1, fn=100)
                collect day_over_day(sum(amount))
                    name "dod_revenue"
                    tags [trends]
            }
        }
        """
        result = parse(source)
        from dqx.dql import Collection

        collection = result.checks[0].assertions[0]
        assert isinstance(collection, Collection)
        assert len(collection.annotations) == 2
        assert collection.annotations[0].name == "experimental"
        assert collection.annotations[1].name == "cost"
        assert collection.annotations[1].args == {"fp": 1, "fn": 100}

    def test_collect_complex_expression(self) -> None:
        """Test collect with complex metric expression."""
        source = """
        suite "Test" {
            check "Metrics" on orders {
                collect null_count(email) / num_rows()
                    name "email_null_rate"
            }
        }
        """
        result = parse(source)
        from dqx.dql import Collection

        collection = result.checks[0].assertions[0]
        assert isinstance(collection, Collection)
        assert "null_count(email)" in collection.expr.text
        assert "num_rows()" in collection.expr.text

    def test_mixed_assert_and_collect(self) -> None:
        """Test check with both assertions and collections."""
        source = """
        suite "Test" {
            check "Quality" on orders {
                assert num_rows() > 1000
                    name "min_rows"
                
                collect average(price)
                    name "avg_price"
                
                assert null_count(email) == 0
                    name "email_not_null"
                
                collect day_over_day(sum(amount))
                    name "dod_revenue"
            }
        }
        """
        result = parse(source)
        check = result.checks[0]
        assert len(check.assertions) == 4

        from dqx.dql import Assertion, Collection

        assert isinstance(check.assertions[0], Assertion)
        assert isinstance(check.assertions[1], Collection)
        assert isinstance(check.assertions[2], Assertion)
        assert isinstance(check.assertions[3], Collection)

    def test_collect_without_name_fails(self) -> None:
        """Test that collect without name raises error."""
        source = """
        suite "Test" {
            check "Metrics" on data {
                collect num_rows()
            }
        }
        """
        # Should raise DQLSyntaxError (wrapped in VisitError by Lark)
        with pytest.raises(Exception, match="must have a 'name'"):
            parse(source)


class TestAnnotations:
    """Test annotation parsing."""

    def test_experimental_annotation(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                @experimental
                assert num_rows() > 0
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert len(assertion.annotations) == 1
        assert assertion.annotations[0].name == "experimental"

    def test_required_annotation(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                @required
                assert num_rows() > 0
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.annotations[0].name == "required"

    def test_cost_annotation(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                @cost(false_positive=1, false_negative=100)
                assert num_rows() > 0
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        ann = assertion.annotations[0]
        assert ann.name == "cost"
        assert ann.args["false_positive"] == 1
        assert ann.args["false_negative"] == 100

    def test_multiple_annotations(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                @experimental
                @cost(false_positive=1, false_negative=50)
                assert num_rows() > 0
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert len(assertion.annotations) == 2


class TestExpressions:
    """Test expression parsing."""

    def test_arithmetic_expression(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert null_count(email) / num_rows() < 0.05
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert "null_count" in assertion.expr.text
        assert "/" in assertion.expr.text

    def test_function_with_lag(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price, lag=1) > 0
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert "lag=1" in assertion.expr.text

    def test_function_with_dataset(self) -> None:
        source = """
        suite "Test" {
            check "Test" on orders, returns {
                assert num_rows(dataset=returns) / num_rows(dataset=orders) < 0.15
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert "dataset=returns" in assertion.expr.text
        assert "dataset=orders" in assertion.expr.text

    def test_nested_function_calls(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert abs(day_over_day(average(price))) < 0.1
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert "abs" in assertion.expr.text
        assert "day_over_day" in assertion.expr.text

    def test_percent_literal(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert null_count(email) / num_rows() < 5%
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        # 5% is converted to 0.05
        assert assertion.threshold is not None
        assert "0.05" in assertion.threshold.text


class TestTunables:
    """Test tunable parsing."""

    def test_simple_tunable(self) -> None:
        source = """
        suite "Test" {
            tunable MAX_NULL_RATE = 0.05 bounds [0.0, 0.2]
        }
        """
        result = parse(source)
        assert len(result.tunables) == 1
        tunable = result.tunables[0]
        assert tunable.name == "MAX_NULL_RATE"
        assert "0.05" in tunable.value.text
        assert "0.0" in tunable.bounds[0].text
        assert "0.2" in tunable.bounds[1].text

    def test_percent_tunable(self) -> None:
        source = """
        suite "Test" {
            tunable MAX_NULL_RATE = 5% bounds [0%, 20%]
        }
        """
        result = parse(source)
        tunable = result.tunables[0]
        assert "0.05" in tunable.value.text
        assert "0.0" in tunable.bounds[0].text
        assert "0.2" in tunable.bounds[1].text

    def test_tunable_with_bounds(self) -> None:
        source = """
        suite "Test" {
            tunable NULL_THRESHOLD = 5% bounds [0%, 20%]
        }
        """
        result = parse(source)
        tunable = result.tunables[0]
        assert tunable.bounds is not None
        assert len(tunable.bounds) == 2
        assert "0.0" in tunable.bounds[0].text
        assert "0.2" in tunable.bounds[1].text


class TestProfiles:
    """Test profile parsing."""

    def test_simple_profile(self) -> None:
        source = """
        suite "Test" {
            profile "Black Friday" {
                type holiday
                from 2024-11-29
                to 2024-12-02
            }
        }
        """
        result = parse(source)
        assert len(result.profiles) == 1
        profile = result.profiles[0]
        assert profile.name == "Black Friday"
        assert profile.profile_type == "holiday"
        assert profile.from_date.value == date(2024, 11, 29)
        assert profile.to_date.value == date(2024, 12, 2)

    def test_profile_with_disable(self) -> None:
        source = """
        suite "Test" {
            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2025-01-05

                disable check "Volume"
            }
        }
        """
        result = parse(source)
        profile = result.profiles[0]
        assert len(profile.rules) == 1
        rule = profile.rules[0]
        assert isinstance(rule, DisableRule)
        assert rule.target_type == "check"
        assert rule.target_name == "Volume"

    def test_profile_with_disable_assertion(self) -> None:
        source = """
        suite "Test" {
            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2025-01-05

                disable assertion "Row count" in "Volume"
            }
        }
        """
        result = parse(source)
        rule = result.profiles[0].rules[0]
        assert isinstance(rule, DisableRule)
        assert rule.target_type == "assertion"
        assert rule.target_name == "Row count"
        assert rule.in_check == "Volume"

    def test_profile_with_scale(self) -> None:
        source = """
        suite "Test" {
            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2025-01-05

                scale tag "volume" by 2.0x
            }
        }
        """
        result = parse(source)
        rule = result.profiles[0].rules[0]
        assert isinstance(rule, ScaleRule)
        assert rule.selector_type == "tag"
        assert rule.selector_name == "volume"
        assert rule.multiplier == 2.0

    def test_profile_with_set_severity(self) -> None:
        source = """
        suite "Test" {
            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2025-01-05

                set tag "non-critical" severity P3
            }
        }
        """
        result = parse(source)
        rule = result.profiles[0].rules[0]
        assert isinstance(rule, SetSeverityRule)
        assert rule.selector_type == "tag"
        assert rule.selector_name == "non-critical"
        assert rule.severity == Severity.P3

    def test_profile_with_date_function(self) -> None:
        source = """
        suite "Test" {
            profile "Thanksgiving" {
                type holiday
                from nth_weekday(november, thursday, 4)
                to nth_weekday(november, thursday, 4) + 3
            }
        }
        """
        result = parse(source)
        profile = result.profiles[0]
        assert "nth_weekday" in str(profile.from_date.value)
        assert profile.to_date.offset == 3


class TestCompleteExample:
    """Test parsing the complete example from the spec."""

    def test_complete_example(self) -> None:
        source = """
        suite "E-Commerce Data Quality" {
            availability_threshold 80%

            tunable MAX_NULL_RATE = 5% bounds [0%, 20%]
            tunable MIN_ORDERS = 1000 bounds [100, 10000]

            check "Completeness" on orders {
                assert null_count(customer_id) == 0
                    name "No null customer IDs"
                    severity P0

                assert null_count(email) / num_rows() < MAX_NULL_RATE
                    name "Email null rate below threshold"
            }

            check "Volume" on orders {
                assert num_rows() >= MIN_ORDERS
                    name "At least minimum orders"
                    tags [volume]

                assert day_over_day(num_rows()) between 0.5 and 2.0
                    name "Day-over-day stable"
                    tags [volume, trend]
            }

            profile "Black Friday" {
                type holiday
                from 2024-11-29
                to 2024-12-02

                scale tag "volume" by 3.0x
            }

            profile "Christmas" {
                type holiday
                from 2024-12-20
                to 2025-01-05

                disable check "Volume"
                set tag "trend" severity P3
            }
        }
        """
        result = parse(source)

        # Suite level
        assert result.name == "E-Commerce Data Quality"
        assert result.availability_threshold == 0.8
        assert len(result.tunables) == 2
        assert len(result.checks) == 2
        assert len(result.profiles) == 2

        # Checks
        completeness = result.checks[0]
        assert completeness.name == "Completeness"
        assert len(completeness.assertions) == 2

        volume = result.checks[1]
        assert volume.name == "Volume"
        assert len(volume.assertions) == 2

        # Profiles
        black_friday = result.profiles[0]
        assert black_friday.name == "Black Friday"
        assert len(black_friday.rules) == 1

        christmas = result.profiles[1]
        assert christmas.name == "Christmas"
        assert len(christmas.rules) == 2


class TestSyntaxErrors:
    """Test syntax error handling."""

    def test_missing_suite_name(self) -> None:
        source = """
        suite {
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(source)

    def test_missing_check_name(self) -> None:
        source = """
        suite "Test" {
            check on orders {
                assert num_rows() > 0
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(source)

    def test_invalid_severity(self) -> None:
        source = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > 0 severity P5
            }
        }
        """
        with pytest.raises(DQLSyntaxError):
            parse(source)

    def test_error_includes_location(self) -> None:
        source = """suite "Test" {
            check "Test" on {
                assert num_rows() > 0
            }
        }
        """
        with pytest.raises(DQLSyntaxError) as exc_info:
            parse(source, filename="test.dql")
        error = exc_info.value
        assert error.loc is not None
        assert error.loc.filename == "test.dql"


class TestComments:
    """Test comment handling."""

    def test_line_comment(self) -> None:
        source = """
        suite "Test" {
            # This is a comment
            check "Test" on orders {
                assert num_rows() > 0  # inline comment
            }
        }
        """
        result = parse(source)
        assert len(result.checks) == 1


class TestCoverageEdgeCases:
    """Tests for edge cases to achieve 100% coverage."""

    def test_negation_expression(self) -> None:
        """Cover negation expression parsing."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert -average(price) < 0
            }
        }
        """
        result = parse(source)
        assert "-" in result.checks[0].assertions[0].expr.text

    def test_none_literal(self) -> None:
        """Cover coalesce function with default value."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert coalesce(average(price), 0) > 0
            }
        }
        """
        result = parse(source)
        assert "coalesce" in result.checks[0].assertions[0].expr.text

    def test_list_arg_in_function(self) -> None:
        """Cover list_arg() - lines 199-203, 222."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert duplicate_count([id, date]) == 0
            }
        }
        """
        result = parse(source)
        assert "[id, date]" in result.checks[0].assertions[0].expr.text

    def test_order_by_arg(self) -> None:
        """Cover order_by argument parsing."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert first(timestamp, order_by=price) > 0
            }
        }
        """
        result = parse(source)
        assert "order_by" in result.checks[0].assertions[0].expr.text

    def test_date_func_no_args(self) -> None:
        """Cover date function with no arguments."""
        source = """
        suite "Test" {
            profile "Month End" {
                type recurring
                from last_day_of_month()
                to last_day_of_month()
            }
        }
        """
        result = parse(source)
        assert "last_day_of_month" in str(result.profiles[0].from_date.value)

    def test_date_subtraction(self) -> None:
        """Cover date_sub() - lines 469-471."""
        source = """
        suite "Test" {
            profile "Month End" {
                type recurring
                from last_day_of_month() - 2
                to last_day_of_month()
            }
        }
        """
        result = parse(source)
        assert result.profiles[0].from_date.offset == -2

    def test_scale_check_selector(self) -> None:
        """Cover sel_check() and selector() - lines 506, 509."""
        source = """
        suite "Test" {
            profile "Holiday" {
                type holiday
                from 2024-12-20
                to 2024-12-31
                scale check "Volume" by 2.0x
            }
        }
        """
        result = parse(source)
        rule = result.profiles[0].rules[0]
        assert isinstance(rule, ScaleRule)
        assert rule.selector_type == "check"
        assert rule.selector_name == "Volume"

    def test_unexpected_character_error(self) -> None:
        """Cover UnexpectedCharacters handling - lines 621-624."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() > $invalid
            }
        }
        """
        with pytest.raises(DQLSyntaxError) as exc_info:
            parse(source)
        assert "Unexpected character" in str(exc_info.value)

    def test_between_with_multiplication(self) -> None:
        """Cover between_bound with multiple terms - lines 251-259."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price) between 10 * 2 and 100 * 5
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.condition == "between"
        assert assertion.threshold is not None
        assert "*" in assertion.threshold.text

    def test_complex_arithmetic_expression(self) -> None:
        """Cover multi-term expr and term - lines 137-149, 162-165."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert sum(a) + sum(b) - sum(c) > 0
                assert sum(x) * 2 / 3 > 0
            }
        }
        """
        result = parse(source)
        expr1 = result.checks[0].assertions[0].expr.text
        expr2 = result.checks[0].assertions[1].expr.text
        assert "+" in expr1
        assert "-" in expr1
        assert "*" in expr2
        assert "/" in expr2


class TestParseFile:
    """Test parse_file function."""

    def test_parse_file(self, tmp_path: Path) -> None:
        """Cover parse_file() - lines 655-657."""
        from dqx.dql import parse_file

        dql_file = tmp_path / "test.dql"
        dql_file.write_text("""
        suite "File Test" {
            check "Basic" on orders {
                assert num_rows() > 0
            }
        }
        """)

        result = parse_file(dql_file)
        assert result.name == "File Test"
        assert len(result.checks) == 1


class TestErrorFormatting:
    """Test error message formatting."""

    def test_dql_error_with_location(self) -> None:
        """Cover DQLError._format_message with loc - lines 17-19."""
        from dqx.dql.errors import DQLError
        from dqx.dql import SourceLocation

        loc = SourceLocation(line=10, column=5, filename="test.dql")
        error = DQLError("Test error", loc=loc)
        assert "test.dql:10:5" in str(error)
        assert "Test error" in str(error)

    def test_dql_syntax_error_without_location(self) -> None:
        """Cover DQLSyntaxError formatting without location."""
        from dqx.dql.errors import DQLSyntaxError

        error = DQLSyntaxError("Test syntax error", loc=None)
        assert "error: Test syntax error" in str(error)

    def test_dql_error_without_location(self) -> None:
        """Cover DQLError formatting without location."""
        from dqx.dql.errors import DQLError

        error = DQLError("Test error", loc=None)
        assert str(error) == "Test error"


class TestAdditionalEdgeCases:
    """Additional edge cases for full coverage."""

    def test_between_term_with_number(self) -> None:
        """Cover between_term with int/float - lines 242, 245."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price) between 10 and 500
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.threshold is not None
        assert assertion.threshold_upper is not None
        assert "10" in assertion.threshold.text
        assert "500" in assertion.threshold_upper.text

    def test_between_term_with_identifier(self) -> None:
        """Cover between bounds with identifier constants."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price) between MIN_VAL and MAX_VAL
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.threshold is not None
        assert assertion.threshold_upper is not None
        assert "MIN_VAL" in assertion.threshold.text
        assert "MAX_VAL" in assertion.threshold_upper.text

    # Sampling support removed - no longer part of DQL spec
    # def test_sample_value_rule(self) -> None:

    def test_named_arg_passthrough(self) -> None:
        """Cover named argument parsing."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price, lag=7) > 0
            }
        }
        """
        result = parse(source)
        assert "lag=7" in result.checks[0].assertions[0].expr.text

    def test_condition_comparison_passthrough(self) -> None:
        """Cover condition with comparison operators."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert num_rows() != 0
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.condition == "!="

    def test_make_loc_no_meta(self) -> None:
        """Cover _make_loc when tree has no meta."""
        from dqx.dql.parser import _make_loc

        class FakeTree:
            pass

        result = _make_loc(FakeTree())
        assert result is None

        class FakeTreeWithEmptyMeta:
            meta = None

        result = _make_loc(FakeTreeWithEmptyMeta())
        assert result is None

    def test_date_expr_string_fallback(self) -> None:
        """Cover date expression with function call."""
        # This is covered by the date function tests
        source = """
        suite "Test" {
            profile "Test" {
                type recurring
                from last_day_of_month()
                to last_day_of_month()
            }
        }
        """
        result = parse(source)
        # The date func returns a DateExpr which is passed through date_expr
        assert result.profiles[0].from_date is not None

    def test_factor_with_parenthesized_expr(self) -> None:
        """Cover parenthesized expression parsing."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert (sum(a) + sum(b)) * 2 > 0
            }
        }
        """
        result = parse(source)
        # Parenthesized expression creates nested Expr
        assert "sum(a)" in result.checks[0].assertions[0].expr.text

    def test_between_with_function_call_bounds(self) -> None:
        """Cover between bounds with function calls."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert average(price) between min_val() and max_val()
            }
        }
        """
        result = parse(source)
        assertion = result.checks[0].assertions[0]
        assert assertion.condition == "between"
        assert assertion.threshold is not None
        assert assertion.threshold_upper is not None
        assert "min_val()" in assertion.threshold.text
        assert "max_val()" in assertion.threshold_upper.text

    def test_date_func_no_args_direct(self) -> None:
        """Cover date function returning value with empty parens."""
        source = """
        suite "Test" {
            profile "End of Month" {
                type recurring
                from today()
                to today()
            }
        }
        """
        result = parse(source)
        assert "today()" in str(result.profiles[0].from_date.value)

    def test_n_parameter(self) -> None:
        """Test 'n' parameter for stddev() and similar functions."""
        source = """
        suite "Test" {
            check "Test" on t {
                assert stddev(average(price), n=7) < 0.5
            }
        }
        """
        result = parse(source)
        assert "n=7" in result.checks[0].assertions[0].expr.text
