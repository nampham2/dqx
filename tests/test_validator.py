import sympy as sp

from dqx.common import SymbolicValidator
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.validator import SuiteValidator, ValidationIssue, ValidationReport


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

    # Create a large suite
    for i in range(100):
        check = root.add_check(f"Check_{i}")
        for j in range(10):
            # Create validator with closure over j value
            max_value = j + 1

            def validator_fn(x: float, mv: int = max_value) -> bool:
                return bool(0 <= x <= mv)

            range_validator = SymbolicValidator(f"in [0, {max_value}]", validator_fn)
            check.add_assertion(sp.Symbol(f"x_{i}_{j}"), name=f"Assert_{j}", validator=range_validator)

    graph = Graph(root)
    validator = SuiteValidator()

    # Create a provider for validation
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    start = time.time()
    report = validator.validate(graph, provider)
    duration = time.time() - start

    # Should complete quickly even for large suites
    assert duration < 0.5  # 500ms should be plenty

    # Should have no issues
    assert not report.has_errors()
    assert not report.has_warnings()
