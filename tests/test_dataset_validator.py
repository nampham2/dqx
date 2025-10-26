"""Tests for DatasetValidator."""

from dqx.common import SymbolicValidator
from dqx.graph.nodes import RootNode
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.validator import DatasetValidator


def test_dataset_validator_detects_mismatch() -> None:
    """Test that DatasetValidator catches dataset mismatches."""
    # Arrange: Create a real provider
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create a graph with dataset mismatch
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production", "staging"])

    # Create a symbol with "testing" dataset - mismatch!
    symbol = provider.average("price", dataset="testing")

    # Add assertion using the symbol
    price_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(symbol, name="avg price > 0", validator=price_validator)

    # Act: Run the validator
    validator = DatasetValidator(provider)
    # Process only the assertion node
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have one error
    issues = validator.get_issues()
    assert len(issues) == 1
    assert issues[0].rule == "dataset_mismatch"
    assert "testing" in issues[0].message
    assert "production" in issues[0].message
    assert "staging" in issues[0].message


def test_dataset_validator_allows_valid_configuration() -> None:
    """Test that DatasetValidator allows matching datasets."""
    # Arrange: Create a real provider
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create a graph with matching datasets
    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production", "staging"])

    # Symbol has dataset that IS in check's datasets
    symbol = provider.average("price", dataset="production")

    price_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(symbol, name="avg price > 0", validator=price_validator)

    # Act: Run the validator
    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have no errors
    issues = validator.get_issues()
    assert len(issues) == 0


def test_dataset_validator_skips_when_no_datasets_specified() -> None:
    """Test that validator skips validation when check has no datasets."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    # Check has no datasets specified
    check = root.add_check("price_check")

    symbol = provider.average("price", dataset="testing")
    price_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(symbol, name="avg price > 0", validator=price_validator)

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have no errors since check doesn't specify datasets
    assert len(validator.get_issues()) == 0


def test_dataset_validator_errors_on_ambiguous_none_dataset() -> None:
    """Test that validator errors when symbol has no dataset but check has multiple."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    # Check has multiple datasets
    check = root.add_check("price_check", datasets=["production", "staging"])

    # Symbol has no dataset - ambiguous!
    symbol = provider.average("price", dataset=None)
    price_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(symbol, name="avg price > 0", validator=price_validator)

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have error about ambiguity
    issues = validator.get_issues()
    assert len(issues) == 1
    assert "no dataset specified" in issues[0].message
    assert "multiple datasets" in issues[0].message
    assert "Unable to determine" in issues[0].message


def test_dataset_validator_allows_none_dataset_with_single_check_dataset() -> None:
    """Test that validator allows None dataset when check has single dataset."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    # Check has single dataset
    check = root.add_check("price_check", datasets=["production"])

    # Symbol has no dataset - OK, will be imputed
    symbol = provider.average("price", dataset=None)
    price_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(symbol, name="avg price > 0", validator=price_validator)

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have no errors
    assert len(validator.get_issues()) == 0


def test_dataset_validator_handles_multiple_symbols() -> None:
    """Test validator with multiple symbols in one assertion."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    check = root.add_check("price_check", datasets=["production"])

    # Create multiple symbols - one invalid, one valid
    valid_symbol = provider.average("price", dataset="production")
    invalid_symbol = provider.average("cost", dataset="testing")

    # Assertion with expression using both symbols
    combined_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(valid_symbol + invalid_symbol, name="combined metric", validator=combined_validator)

    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Should have one error for the invalid symbol
    issues = validator.get_issues()
    assert len(issues) == 1
    assert "average(cost)" in issues[0].message
    assert "testing" in issues[0].message


def test_dataset_validator_detects_parent_child_mismatch() -> None:
    """Test that DatasetValidator detects mismatches between parent and child symbols."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    check = root.add_check("revenue_check", datasets=["staging"])

    # Create base metric with dataset
    base_symbol = provider.sum("revenue", dataset="production")

    # Create derived metric (day_over_day) with different dataset - this is the parent now!
    dod_symbol = provider.ext.day_over_day(base_symbol, dataset="staging")

    # Add assertion using the derived symbol (parent)
    revenue_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(dod_symbol, name="revenue day-over-day > 0", validator=revenue_validator)

    # Act: Run the validator
    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have 2 errors about dependent dataset mismatch
    # One for sum(revenue) and one for lag(1) metric
    issues = validator.get_issues()
    assert len(issues) == 2

    # Check both errors have the right content
    for issue in issues:
        assert issue.rule == "dataset_mismatch"
        assert "Required metric" in issue.message
        assert "staging" in issue.message
        assert "production" in issue.message

    # Check we have errors for both base and lag metrics
    error_messages = [issue.message for issue in issues]
    assert any("sum(revenue)" in msg for msg in error_messages)
    assert any("lag(1)" in msg for msg in error_messages)


def test_dataset_validator_allows_consistent_parent_child_datasets() -> None:
    """Test that DatasetValidator allows parent and child with same datasets."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    check = root.add_check("revenue_check", datasets=["production"])

    # Create base metric with dataset
    base_symbol = provider.sum("revenue", dataset="production")

    # Create derived metric (parent) with same dataset - this is OK
    dod_symbol = provider.ext.day_over_day(base_symbol, dataset="production")

    # Add assertion using the derived symbol
    revenue_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(dod_symbol, name="revenue day-over-day > 0", validator=revenue_validator)

    # Act: Run the validator
    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have no errors
    issues = validator.get_issues()
    assert len(issues) == 0


def test_dataset_validator_allows_child_without_dataset() -> None:
    """Test that DatasetValidator allows child symbols without datasets (will be imputed)."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    check = root.add_check("revenue_check", datasets=["production"])

    # Create base metric without dataset
    base_symbol = provider.sum("revenue", dataset=None)

    # Create derived metric (parent) with dataset - OK, child will be imputed
    dod_symbol = provider.ext.day_over_day(base_symbol, dataset="production")

    # Add assertion using the derived symbol
    revenue_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(dod_symbol, name="revenue day-over-day > 0", validator=revenue_validator)

    # Act: Run the validator
    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have no errors
    issues = validator.get_issues()
    assert len(issues) == 0


def test_dataset_validator_checks_multiple_children() -> None:
    """Test that DatasetValidator checks all children of a symbol."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    root = RootNode("test_suite")
    check = root.add_check("revenue_check", datasets=["production"])

    # Create base metric with dataset that differs from derived metrics
    base_symbol = provider.sum("revenue", dataset="testing")

    # Create multiple derived metrics (parents) with same dataset
    dod_symbol = provider.ext.day_over_day(base_symbol, dataset="production")
    wow_symbol = provider.ext.week_over_week(base_symbol, dataset="production")

    # Add assertions using both derived symbols
    revenue_validator = SymbolicValidator("> 0", lambda x: x > 0)
    check.add_assertion(dod_symbol, name="revenue dod > 0", validator=revenue_validator)
    check.add_assertion(wow_symbol, name="revenue wow > 0", validator=revenue_validator)

    # Act: Run the validator
    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have 4 errors total
    # - 2 for day_over_day (base metric + lag(1) metric)
    # - 2 for week_over_week (base metric + lag(7) metric)
    issues = validator.get_issues()
    assert len(issues) == 4

    # Check error messages contain expected content
    error_messages = [issue.message for issue in issues]
    assert all("Required metric" in msg for msg in error_messages)
    assert all("testing" in msg for msg in error_messages)
    assert all("production" in msg for msg in error_messages)

    # Check we have errors for both base and lag metrics
    assert any("sum(revenue)" in msg for msg in error_messages)
    assert any("lag(1)" in msg for msg in error_messages)
    assert any("lag(7)" in msg for msg in error_messages)
