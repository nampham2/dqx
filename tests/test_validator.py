import sympy as sp

from dqx import specs
from dqx.common import SymbolicValidator
from dqx.graph.base import BaseNode
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.validator import BaseValidator, SuiteValidator, ValidationIssue, ValidationReport


def test_validation_issue_creation() -> None:
    """Test creating a validation issue."""
    issue = ValidationIssue(rule="test_rule", message="Something went wrong", node_path=["root", "check1"])

    assert issue.rule == "test_rule"
    assert issue.message == "Something went wrong"
    assert issue.node_path == ["root", "check1"]


def test_validation_report_empty() -> None:
    """Test empty validation report."""
    report = ValidationReport()

    assert not report.has_errors()
    assert not report.has_warnings()
    assert len(report.errors) == 0
    assert len(report.warnings) == 0


def test_validation_report_add_issues() -> None:
    """Test adding issues to report."""
    report = ValidationReport()

    # Add an error
    report.add_error(ValidationIssue(rule="duplicate_check", message="Duplicate found", node_path=["root"]))

    # Add a warning
    report.add_warning(ValidationIssue(rule="empty_check", message="Empty check found", node_path=["root", "check1"]))

    assert report.has_errors()
    assert report.has_warnings()
    assert len(report.errors) == 1
    assert len(report.warnings) == 1


def test_validation_report_string_format() -> None:
    """Test report string formatting."""
    report = ValidationReport()

    report.add_error(
        ValidationIssue(
            rule="duplicate_check", message="Duplicate check name: 'test_check'", node_path=["root", "check:test_check"]
        )
    )

    report.add_warning(
        ValidationIssue(rule="empty_check", message="Check 'test' has no assertions", node_path=["root", "check:test"])
    )

    report_str = str(report)
    assert "ERROR" in report_str
    assert "WARNING" in report_str
    assert "Duplicate check name: 'test_check'" in report_str
    assert "Check 'test' has no assertions" in report_str


def test_validation_report_string_format_no_issues() -> None:
    """Test report string formatting when there are no issues."""
    report = ValidationReport()

    report_str = str(report)
    assert "No validation issues found" in report_str
    assert "ERROR" not in report_str
    assert "WARNING" not in report_str


def test_validation_report_to_dict() -> None:
    """Test structured output of validation report."""
    report = ValidationReport()

    error = ValidationIssue(
        rule="duplicate_check", message="Duplicate check name: 'test'", node_path=["root", "check:test"]
    )
    warning = ValidationIssue(
        rule="empty_check", message="Check 'empty' has no assertions", node_path=["root", "check:empty"]
    )

    report.add_error(error)
    report.add_warning(warning)

    # Test structured output
    result = report.to_dict()

    assert result["summary"]["error_count"] == 1
    assert result["summary"]["warning_count"] == 1
    assert len(result["errors"]) == 1
    assert len(result["warnings"]) == 1

    # Check error structure
    assert result["errors"][0]["rule"] == "duplicate_check"
    assert result["errors"][0]["message"] == "Duplicate check name: 'test'"
    assert result["errors"][0]["node_path"] == ["root", "check:test"]


def test_suite_validator_clean_suite() -> None:
    """Test validator with a clean suite (no issues)."""
    root = RootNode("clean_suite")
    check = root.add_check("Good Check")
    positive_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(sp.Symbol("x"), name="X is positive", validator=positive_validator)

    graph = Graph(root)
    validator = SuiteValidator()

    # Create a provider for validation
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    report = validator.validate(graph, provider)
    assert not report.has_errors()
    assert not report.has_warnings()


def test_suite_validator_duplicate_check_names() -> None:
    """Test validator detects duplicate check names."""
    root = RootNode("suite")
    root.add_check("Duplicate")
    root.add_check("Duplicate")
    root.add_check("Unique")

    graph = Graph(root)
    validator = SuiteValidator()

    # Create a provider for validation
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    report = validator.validate(graph, provider)
    assert report.has_errors()
    assert "Duplicate check name" in str(report)


def test_suite_validator_empty_checks() -> None:
    """Test validator detects empty checks."""
    root = RootNode("suite")

    # Empty check
    root.add_check("Empty Check")

    # Normal check
    normal = root.add_check("Normal Check")
    test_validator = SymbolicValidator("not None", lambda x: x is not None)
    normal.add_assertion(sp.Symbol("x"), name="Test", validator=test_validator)

    graph = Graph(root)
    validator = SuiteValidator()

    # Create a provider for validation
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    report = validator.validate(graph, provider)
    assert report.has_warnings()
    assert "Empty Check" in str(report)


def test_suite_validator_duplicate_assertion_names() -> None:
    """Test validator detects duplicate assertion names within a check."""
    root = RootNode("suite")
    check = root.add_check("Test Check")
    validator_x = SymbolicValidator("> 0", lambda x: x > 0)
    validator_y = SymbolicValidator("< 100", lambda x: x < 100)
    validator_z = SymbolicValidator("!= 0", lambda x: x != 0)
    check.add_assertion(sp.Symbol("x"), name="Same", validator=validator_x)
    check.add_assertion(sp.Symbol("y"), name="Same", validator=validator_y)
    check.add_assertion(sp.Symbol("z"), name="Different", validator=validator_z)

    graph = Graph(root)
    validator = SuiteValidator()

    # Create a provider for validation
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    report = validator.validate(graph, provider)
    assert report.has_errors()
    assert "Same" in str(report)
    assert "Test Check" in str(report)


def test_suite_validator_performance() -> None:
    """Test that validation completes quickly for large suites."""
    import time

    root = RootNode("large_suite")

    # Create a provider for validation
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create a large suite
    for i in range(100):
        check = root.add_check(f"Check_{i}")
        for j in range(10):
            # Create metrics through provider so they're registered
            metric = provider.metric(specs.Average(f"column_{i}_{j}"))

            # Create validator with closure over j value
            max_value = j + 1

            def validator_fn(x: float, mv: int = max_value) -> bool:
                return bool(0 <= x <= mv)

            range_validator = SymbolicValidator(f"in [0, {max_value}]", validator_fn)
            check.add_assertion(metric, name=f"Assert_{j}", validator=range_validator)

    graph = Graph(root)
    validator = SuiteValidator()

    start = time.time()
    report = validator.validate(graph, provider)
    duration = time.time() - start

    # Should complete quickly even for large suites
    assert duration < 1.0  # 1 second should be plenty

    # Should have no issues
    assert not report.has_errors()
    assert not report.has_warnings()


def test_base_validator_get_issues() -> None:
    """Test BaseValidator get_issues method."""

    class TestValidator(BaseValidator):
        """Simple test validator."""

        name = "test_validator"
        is_error = True

        def process_node(self, node: BaseNode) -> None:
            """Process a node and add a test issue."""
            if hasattr(node, "name") and node.name == "problematic":
                self._issues.append(
                    ValidationIssue(rule=self.name, message="Found problematic node", node_path=["test", "path"])
                )

    # Create validator instance
    validator = TestValidator()

    # Initially should have no issues
    assert len(validator.get_issues()) == 0

    # Create a mock node and process it
    from unittest.mock import Mock

    problem_node = Mock()
    problem_node.name = "problematic"
    validator.process_node(problem_node)

    # Should now have one issue
    issues = validator.get_issues()
    assert len(issues) == 1
    assert issues[0].rule == "test_validator"
    assert issues[0].message == "Found problematic node"
    assert issues[0].node_path == ["test", "path"]

    # Test with non-problematic node
    good_node = Mock()
    good_node.name = "good"
    validator.process_node(good_node)

    # Should still have only one issue
    assert len(validator.get_issues()) == 1

    # Test reset
    validator.reset()
    assert len(validator.get_issues()) == 0


def test_unused_symbol_validator_detects_unused() -> None:
    """Test validator detects unused symbols."""
    # Setup
    root = RootNode("suite")
    check = root.add_check("Test Check")

    # Create provider and define symbols
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define metrics but don't use them all
    x1 = provider.average("revenue")  # This will be unused  # noqa: F841
    x2 = provider.sum("orders")  # This will be used

    # Only use x2 in assertion
    validator_fn = SymbolicValidator("> 100", lambda x: x > 100)
    check.add_assertion(x2 > 100, name="Orders check", validator=validator_fn)

    # Validate
    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Assert
    assert report.has_warnings()
    warnings = report.warnings
    unused_warnings = [w for w in warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 1
    assert "x_1 ← average(revenue)" in unused_warnings[0].message
    assert "x_2" not in str(report)  # x_2 is used, shouldn't warn


def test_unused_symbol_validator_no_warnings_when_all_used() -> None:
    """Test no warnings when all symbols are used."""
    root = RootNode("suite")
    check = root.add_check("Test Check")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define symbols and use ALL of them
    x1 = provider.average("revenue")
    x2 = provider.sum("orders")

    validator_fn1 = SymbolicValidator("> 100", lambda x: x > 100)
    validator_fn2 = SymbolicValidator("> 50", lambda x: x > 50)

    check.add_assertion(x1 > 50, name="Revenue check", validator=validator_fn1)
    check.add_assertion(x2 > 100, name="Orders check", validator=validator_fn2)

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should have no unused symbol warnings
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 0


def test_unused_symbol_validator_complex_expressions() -> None:
    """Test validator correctly identifies symbols in complex expressions."""
    root = RootNode("suite")
    check = root.add_check("Complex Check")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define multiple symbols
    x1 = provider.average("revenue")
    x2 = provider.sum("orders")
    x3 = provider.minimum("price")
    x4 = provider.maximum("price")
    x5 = provider.variance("latency")  # noqa: F841
    x6 = provider.null_count("user_id")  # noqa: F841

    # Use complex expressions
    validator_fn = SymbolicValidator("complex", lambda x: x > 0)

    # x1, x2, x3, x4 are used
    check.add_assertion(x1 + x2 > 100, name="Sum check", validator=validator_fn)
    check.add_assertion(x3 * 2 + x4 / x1 > 50, name="Complex calc", validator=validator_fn)

    # x5, x6 are not used

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should warn about x5 and x6
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 2

    warning_messages = [w.message for w in unused_warnings]
    assert any("x_5 ← variance(latency)" in msg for msg in warning_messages)
    assert any("x_6 ← null_count(user_id)" in msg for msg in warning_messages)


def test_unused_symbol_validator_extended_metrics() -> None:
    """Test validator works with extended metrics."""
    root = RootNode("suite")
    check = root.add_check("Extended Check")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define regular and extended metrics
    # Note: When creating extended metrics, the base metrics are also registered as symbols
    x1 = provider.ext.day_over_day(provider.sum("revenue"))
    x2 = provider.ext.stddev(provider.average("latency"), lag=1, n=7)
    x3 = provider.average("conversion_rate")  # Regular metric, unused  # noqa: F841

    # Only use extended metrics (x1 and x2)
    # The base metrics (sum("revenue") and average("latency")) are registered but not directly used
    # In the current implementation, they will be warned as unused since extended metrics
    # create lag versions (required_metrics) rather than marking the original as a parent
    validator_fn = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(x1 > 0.1, name="DoD check", validator=validator_fn)
    check.add_assertion(x2 < 100, name="Stddev check", validator=validator_fn)

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should warn about:
    # - average(conversion_rate) - explicitly created but unused
    # - average(latency) - created as base for stddev but not directly used
    # Note: sum(revenue) is in required_metrics of day_over_day so it won't be warned
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 2

    warning_messages = [w.message for w in unused_warnings]
    # Check that the unused symbols are warned about
    assert any("average(conversion_rate)" in msg for msg in warning_messages)
    assert any("average(latency)" in msg for msg in warning_messages)
    # sum(revenue) should NOT be warned - it's in required_metrics of day_over_day
    assert not any("sum(revenue)" in msg for msg in warning_messages)


def test_unused_symbol_validator_no_symbols_defined() -> None:
    """Test validator handles case with no symbols defined."""
    root = RootNode("suite")
    check = root.add_check("Literal Check")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create assertions using only literals
    validator_fn = SymbolicValidator("= 42", lambda x: x == 42)
    check.add_assertion(sp.Integer(42), name="Literal assertion", validator=validator_fn)

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should produce no unused symbol warnings
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 0


def test_unused_symbol_validator_empty_suite() -> None:
    """Test validator handles empty suite gracefully."""
    root = RootNode("empty_suite")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should produce no warnings
    assert not report.has_warnings()
    assert not report.has_errors()


def test_unused_symbol_validator_multiple_unused() -> None:
    """Test validator reports multiple unused symbols."""
    root = RootNode("suite")
    check = root.add_check("Multi Check")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define 5 symbols
    x1 = provider.average("metric1")  # noqa: F841
    x2 = provider.sum("metric2")
    x3 = provider.minimum("metric3")  # noqa: F841
    x4 = provider.maximum("metric4")
    x5 = provider.num_rows()  # noqa: F841

    # Use only 2 symbols
    validator_fn = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(x2 > 100, name="Sum check", validator=validator_fn)
    check.add_assertion(x4 < 1000, name="Max check", validator=validator_fn)

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should get 3 warnings
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 3

    warning_messages = [w.message for w in unused_warnings]
    assert any("x_1 ← average(metric1)" in msg for msg in warning_messages)
    assert any("x_3 ← minimum(metric3)" in msg for msg in warning_messages)
    assert any("x_5 ← num_rows()" in msg for msg in warning_messages)


def test_unused_symbol_validator_symbol_reused() -> None:
    """Test symbol used in multiple assertions is not warned."""
    root = RootNode("suite")
    check1 = root.add_check("Check 1")
    check2 = root.add_check("Check 2")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define symbols
    x1 = provider.average("revenue")
    x2 = provider.sum("orders")

    # Use x1 in multiple assertions across different checks
    validator_fn = SymbolicValidator("> 0", lambda x: x > 0)
    check1.add_assertion(x1 > 100, name="Revenue > 100", validator=validator_fn)
    check1.add_assertion(x1 < 1000, name="Revenue < 1000", validator=validator_fn)
    check2.add_assertion(x1 + x2 > 500, name="Combined check", validator=validator_fn)

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should not warn about x1 or x2
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 0


def test_suite_validator_with_unused_symbols() -> None:
    """Test full validation including unused symbols with other issues."""
    root = RootNode("suite")

    # Create duplicate check names (error)
    check1 = root.add_check("Duplicate")
    check2 = root.add_check("Duplicate")  # noqa: F841

    # Create empty check (warning)
    empty_check = root.add_check("Empty Check")  # noqa: F841

    # Create provider and define unused symbols
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    x1 = provider.average("unused_metric")  # Unused  # noqa: F841
    x2 = provider.sum("used_metric")  # Will be used

    # Add assertion to one of the duplicate checks
    validator_fn = SymbolicValidator("> 0", lambda x: x > 0)
    check1.add_assertion(x2 > 100, name="Used assertion", validator=validator_fn)

    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Should have:
    # - 1 error (duplicate check names)
    # - 3 warnings (2 empty checks + 1 unused symbol)
    # Note: check2 is also empty since we only added assertion to check1
    assert report.has_errors()
    assert report.has_warnings()
    assert len(report.errors) == 1
    assert len(report.warnings) == 3

    # Check specific issues
    assert any("Duplicate check name" in e.message for e in report.errors)
    assert sum(1 for w in report.warnings if "has no assertions" in w.message) == 2  # Two empty checks
    assert any("x_1 ← average(unused_metric)" in w.message for w in report.warnings)


def test_unused_symbol_validator_reset() -> None:
    """Test validator reset clears state properly."""
    root = RootNode("suite")
    check = root.add_check("Test Check")  # noqa: F841

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define unused symbol
    x1 = provider.average("metric1")  # noqa: F841

    graph = Graph(root)
    validator = SuiteValidator()

    # First validation
    report1 = validator.validate(graph, provider)
    unused_warnings1 = [w for w in report1.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings1) == 1

    # Reset and validate again
    # The validator should be reset internally by SuiteValidator
    report2 = validator.validate(graph, provider)
    unused_warnings2 = [w for w in report2.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings2) == 1  # Should still find the same issue

    # Warnings should be identical
    assert unused_warnings1[0].message == unused_warnings2[0].message


def test_unused_symbol_validator_performance() -> None:
    """Test validator performance with many symbols."""
    import time

    root = RootNode("large_suite")
    check = root.add_check("Large Check")

    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create 1000 symbols
    symbols = []
    for i in range(1000):
        sym = provider.metric(specs.Sum(f"metric_{i}"))
        symbols.append(sym)

    # Use 500 of them
    validator_fn = SymbolicValidator("> 0", lambda x: x > 0)
    for i in range(0, 1000, 2):  # Use every other symbol
        check.add_assertion(symbols[i] > 0, name=f"Assert_{i}", validator=validator_fn)

    graph = Graph(root)
    validator = SuiteValidator()

    start = time.time()
    report = validator.validate(graph, provider)
    duration = time.time() - start

    # Should complete quickly
    assert duration < 1.0  # 1000ms should be plenty

    # Should have 500 unused symbol warnings
    unused_warnings = [w for w in report.warnings if w.rule == "unused_symbols"]
    assert len(unused_warnings) == 500
