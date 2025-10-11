"""Tests for DatasetValidator."""

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
    check.add_assertion(symbol, name="avg price > 0")

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

    check.add_assertion(symbol, name="avg price > 0")

    # Act: Run the validator
    validator = DatasetValidator(provider)
    for child in check.children:
        validator.process_node(child)

    # Assert: Should have no errors
    issues = validator.get_issues()
    assert len(issues) == 0
