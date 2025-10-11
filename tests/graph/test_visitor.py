"""Tests for visitor classes in the graph module."""

from unittest.mock import Mock

import pytest
import sympy as sp

from dqx.common import DQXError
from dqx.graph.nodes import RootNode
from dqx.graph.visitors import DatasetImputationVisitor
from dqx.provider import MetricProvider, SymbolicMetric


class TestDatasetImputationVisitor:
    """Test suite for DatasetImputationVisitor."""

    def test_propagates_datasets_from_root_to_check(self) -> None:
        """When CheckNode has no datasets, it inherits from parent's datasets."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check")  # No datasets specified

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)

        # Act
        visitor.visit(root)  # Visit root first
        visitor.visit(check)

        # Assert
        assert check.datasets == ["prod", "staging"]

    def test_preserves_existing_check_datasets(self) -> None:
        """When CheckNode has datasets, they are preserved if valid."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["prod"])

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)

        # Act
        visitor.visit(root)  # Visit root first
        visitor.visit(check)

        # Assert
        assert check.datasets == ["prod"]  # Preserved

    def test_error_on_invalid_check_dataset(self) -> None:
        """When CheckNode specifies dataset not in available, collect error."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["invalid_dataset"])

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)

        # Act
        visitor.visit(root)  # Visit root first to populate datasets
        visitor.visit(check)

        # Assert
        assert visitor.has_errors()
        errors = visitor.get_errors()
        assert len(errors) == 1
        assert "invalid_dataset" in errors[0]
        assert "test_check" in errors[0]

    def test_imputation_is_idempotent(self) -> None:
        """Running imputation twice produces the same result."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check")
        assertion = check.add_assertion(actual=sp.Symbol("x_1"), name="test_assertion")

        # Create mock provider with symbolic metric
        provider = Mock(spec=MetricProvider)
        symbolic_metric = Mock(spec=SymbolicMetric)
        symbolic_metric.name = "x_1"
        symbolic_metric.dataset = None
        provider.get_symbol.return_value = symbolic_metric

        visitor = DatasetImputationVisitor(["prod"], provider=provider)

        # Act - First imputation
        visitor.visit(root)
        visitor.visit(check)
        visitor.visit(assertion)
        first_check_datasets = check.datasets.copy()
        first_metric_dataset = symbolic_metric.dataset

        # Act - Second imputation
        visitor2 = DatasetImputationVisitor(["prod"], provider=provider)
        visitor2.visit(root)
        visitor2.visit(check)
        visitor2.visit(assertion)

        # Assert
        assert check.datasets == first_check_datasets
        assert symbolic_metric.dataset == first_metric_dataset

    def test_preserves_explicitly_set_datasets(self) -> None:
        """Datasets explicitly set on SymbolicMetric are preserved."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["prod", "staging"])
        assertion = check.add_assertion(actual=sp.Symbol("x_1"), name="test_assertion")

        # Create mock provider with symbolic metric that has explicit dataset
        provider = Mock(spec=MetricProvider)
        symbolic_metric = Mock(spec=SymbolicMetric)
        symbolic_metric.name = "x_1"
        symbolic_metric.dataset = "prod"  # Explicitly set
        provider.get_symbol.return_value = symbolic_metric

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=provider)

        # Act
        visitor.visit(assertion)

        # Assert
        assert symbolic_metric.dataset == "prod"  # Should be preserved
        assert not visitor.has_errors()

    def test_error_on_symbolic_metric_dataset_mismatch(self) -> None:
        """Error when SymbolicMetric requires dataset not in check."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["staging"])
        assertion = check.add_assertion(actual=sp.Symbol("x_1"), name="test_assertion")

        # Create mock provider with symbolic metric requiring different dataset
        provider = Mock(spec=MetricProvider)
        symbolic_metric = Mock(spec=SymbolicMetric)
        symbolic_metric.name = "x_1"
        symbolic_metric.dataset = "prod"  # Requires prod but check only has staging
        provider.get_symbol.return_value = symbolic_metric

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=provider)

        # Act
        visitor.visit(assertion)

        # Assert
        assert visitor.has_errors()
        errors = visitor.get_errors()
        assert len(errors) == 1
        assert "x_1" in errors[0]
        assert "prod" in errors[0]
        assert "staging" in errors[0]

    def test_error_on_ambiguous_dataset_imputation(self) -> None:
        """Error when metric has no dataset but check has multiple."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["prod", "staging"])
        assertion = check.add_assertion(actual=sp.Symbol("x_1"), name="test_assertion")

        # Create mock provider with symbolic metric without dataset
        provider = Mock(spec=MetricProvider)
        symbolic_metric = Mock(spec=SymbolicMetric)
        symbolic_metric.name = "x_1"
        symbolic_metric.dataset = None  # No dataset specified
        provider.get_symbol.return_value = symbolic_metric

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=provider)

        # Act
        visitor.visit(assertion)

        # Assert
        assert visitor.has_errors()
        errors = visitor.get_errors()
        assert len(errors) == 1
        assert "x_1" in errors[0]
        assert "multiple datasets" in errors[0]

    def test_successful_single_dataset_imputation(self) -> None:
        """Successfully impute dataset when check has single dataset."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["prod"])
        assertion = check.add_assertion(actual=sp.Symbol("x_1"), name="test_assertion")

        # Create mock provider with symbolic metric without dataset
        provider = Mock(spec=MetricProvider)
        symbolic_metric = Mock(spec=SymbolicMetric)
        symbolic_metric.name = "x_1"
        symbolic_metric.dataset = None  # Will be imputed
        provider.get_symbol.return_value = symbolic_metric

        visitor = DatasetImputationVisitor(["prod"], provider=provider)

        # Act
        visitor.visit(assertion)

        # Assert
        assert symbolic_metric.dataset == "prod"  # Successfully imputed
        assert not visitor.has_errors()

    def test_error_on_empty_available_datasets(self) -> None:
        """Initialization fails with empty available datasets."""
        # Act & Assert
        with pytest.raises(DQXError, match="At least one dataset must be provided"):
            DatasetImputationVisitor([], provider=None)

    def test_visit_root_node_sets_datasets(self) -> None:
        """Visiting RootNode sets its datasets from available datasets."""
        # Arrange
        root = RootNode("test_suite")
        visitor = DatasetImputationVisitor(["prod"], provider=None)

        # Act
        visitor.visit(root)

        # Assert
        assert not visitor.has_errors()
        assert root.datasets == ["prod"]  # RootNode now has datasets set

    def test_multiple_errors_are_collected(self) -> None:
        """Multiple validation errors are collected and reported together."""
        # Arrange
        root = RootNode("test_suite")

        # First check with invalid dataset
        check1 = root.add_check("check1", datasets=["invalid1"])

        # Second check with another invalid dataset
        check2 = root.add_check("check2", datasets=["invalid2"])

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=None)

        # Act
        visitor.visit(root)  # Visit root first
        visitor.visit(check1)
        visitor.visit(check2)

        # Assert
        assert visitor.has_errors()
        errors = visitor.get_errors()
        assert len(errors) == 2
        assert any("check1" in err and "invalid1" in err for err in errors)
        assert any("check2" in err and "invalid2" in err for err in errors)

    def test_root_node_receives_available_datasets(self) -> None:
        """RootNode should be populated with available datasets when visited."""
        # Arrange
        root = RootNode("test_suite")
        visitor = DatasetImputationVisitor(["prod", "staging", "dev"], provider=None)

        # Act
        visitor.visit(root)

        # Assert
        assert root.datasets == ["prod", "staging", "dev"]

    def test_root_node_datasets_are_copied_not_referenced(self) -> None:
        """RootNode should get a copy of datasets, not a reference."""
        # Arrange
        available = ["prod", "staging"]
        root = RootNode("test_suite")
        visitor = DatasetImputationVisitor(available, provider=None)

        # Act
        visitor.visit(root)
        available.append("dev")  # Modify original list

        # Assert
        assert root.datasets == ["prod", "staging"]  # Should not include "dev"

    def test_check_validates_against_parent_not_available(self) -> None:
        """CheckNode should validate against parent datasets, not available."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["dev"])

        # Manually set root datasets to simulate a filtered scenario
        root.datasets = ["prod", "staging"]  # "dev" is not included

        visitor = DatasetImputationVisitor(["prod", "staging", "dev"], provider=None)

        # Act - Don't visit root (it already has datasets set)
        visitor.visit(check)

        # Assert
        assert visitor.has_errors()
        errors = visitor.get_errors()
        assert any("parent datasets" in err for err in errors)
        assert any("dev" in err for err in errors)

    def test_hierarchical_flow_root_to_check_to_assertion(self) -> None:
        """Test complete hierarchical flow from root to assertion."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check")  # No datasets
        assertion = check.add_assertion(actual=sp.Symbol("x_1"), name="test")

        # Mock provider
        provider = Mock(spec=MetricProvider)
        metric = Mock(spec=SymbolicMetric)
        metric.name = "x_1"
        metric.dataset = None
        provider.get_symbol.return_value = metric

        visitor = DatasetImputationVisitor(["prod"], provider=provider)

        # Act - Visit in hierarchical order
        visitor.visit(root)  # Sets root.datasets = ["prod"]
        visitor.visit(check)  # Sets check.datasets = ["prod"] from parent
        visitor.visit(assertion)  # Imputes metric.dataset = "prod"

        # Assert
        assert root.datasets == ["prod"]
        assert check.datasets == ["prod"]
        assert metric.dataset == "prod"
        assert not visitor.has_errors()

    def test_error_summary_formatting(self) -> None:
        """Error summary is properly formatted."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check", datasets=["invalid"])

        visitor = DatasetImputationVisitor(["prod"], provider=None)

        # Act
        visitor.visit(root)  # Visit root first
        visitor.visit(check)

        # Assert
        summary = visitor.get_error_summary()
        assert "Dataset validation failed" in summary
        assert "1 error(s)" in summary
        assert "invalid" in summary

    def test_no_errors_returns_empty_summary(self) -> None:
        """When no errors, get_error_summary returns empty string."""
        # Arrange
        visitor = DatasetImputationVisitor(["prod"], provider=None)

        # Assert
        assert visitor.get_error_summary() == ""
        assert visitor.get_errors() == []
        assert not visitor.has_errors()


class TestDatasetImputationIntegration:
    """Integration tests for dataset imputation across the full graph."""

    def test_full_dataset_propagation_flow(self) -> None:
        """Test dataset propagation through entire graph."""
        # Arrange
        root = RootNode("test_suite")
        check = root.add_check("test_check")  # No datasets
        assertion = check.add_assertion(actual=sp.Symbol("x_1") + sp.Symbol("x_2"), name="sum_assertion")

        # Create mock provider with two symbolic metrics
        provider = Mock(spec=MetricProvider)

        metric1 = Mock(spec=SymbolicMetric)
        metric1.name = "x_1"
        metric1.dataset = None

        metric2 = Mock(spec=SymbolicMetric)
        metric2.name = "x_2"
        metric2.dataset = None

        provider.get_symbol.side_effect = lambda s: metric1 if str(s) == "x_1" else metric2

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=provider)

        # Act - Visit all nodes in order
        visitor.visit(root)
        visitor.visit(check)
        visitor.visit(assertion)

        # Assert
        assert check.datasets == ["prod", "staging"]  # Inherited from available
        # Cannot impute dataset when check has multiple datasets
        assert metric1.dataset is None  # Cannot be imputed
        assert metric2.dataset is None  # Cannot be imputed
        # Verify the errors
        assert visitor.has_errors()
        assert len(visitor.get_errors()) == 2  # One for each symbol
        errors = visitor.get_errors()
        assert all("multiple datasets" in error for error in errors)

    def test_dataset_validation_errors_surface_correctly(self) -> None:
        """Test that validation errors are properly reported."""
        # Arrange
        root = RootNode("test_suite")

        # Check with invalid dataset
        check1 = root.add_check("check1", datasets=["invalid"])
        assertion1 = check1.add_assertion(actual=sp.Symbol("x_1"), name="assertion1")

        # Check with valid dataset but mismatched metric
        check2 = root.add_check("check2", datasets=["prod"])
        assertion2 = check2.add_assertion(actual=sp.Symbol("x_2"), name="assertion2")

        # Mock provider
        provider = Mock(spec=MetricProvider)

        metric1 = Mock(spec=SymbolicMetric)
        metric1.name = "x_1"
        metric1.dataset = None

        metric2 = Mock(spec=SymbolicMetric)
        metric2.name = "x_2"
        metric2.dataset = "staging"  # Requires staging but check2 only has prod

        provider.get_symbol.side_effect = lambda s: metric1 if str(s) == "x_1" else metric2

        visitor = DatasetImputationVisitor(["prod", "staging"], provider=provider)

        # Act - Process entire graph
        visitor.visit(root)
        visitor.visit(check1)
        visitor.visit(assertion1)  # Won't process due to check1 error
        visitor.visit(check2)
        visitor.visit(assertion2)

        # Assert
        assert visitor.has_errors()
        errors = visitor.get_errors()
        assert len(errors) == 2

        # Verify specific errors
        assert any("check1" in err and "invalid" in err for err in errors)
        assert any("x_2" in err and "staging" in err for err in errors)
