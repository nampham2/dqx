"""Additional tests to improve coverage for validator.py."""

import pytest

from dqx.graph.base import BaseNode
from dqx.graph.nodes import RootNode
from dqx.validator import BaseValidator, CompositeValidationVisitor, DuplicateCheckNameValidator, ValidationIssue


def test_base_validator_abstract_method() -> None:
    """Test that BaseValidator.process_node is abstract and must be implemented."""

    # Try to create a validator that doesn't implement process_node
    # This should fail since BaseValidator is abstract
    class IncompleteValidator(BaseValidator):
        name = "incomplete"
        is_error = True
        # Intentionally not implementing process_node

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # This should raise TypeError
        IncompleteValidator()  # type: ignore[abstract]

    # Now test that we can call the base method directly to cover the 'pass' statement
    # We need a concrete implementation first
    class ConcreteValidator(BaseValidator):
        name = "concrete"
        is_error = True

        def process_node(self, node: BaseNode) -> None:
            # Override with our own implementation
            # But we can still call super() to test the base 'pass'
            super().process_node(node)  # type: ignore[safe-super]

    validator = ConcreteValidator()
    from unittest.mock import Mock

    node = Mock()
    validator.process_node(node)  # This covers the base class 'pass' statement

    # Should have no issues since the base implementation does nothing
    assert len(validator.get_issues()) == 0


def test_duplicate_check_name_validator_reset() -> None:
    """Test reset method of DuplicateCheckNameValidator."""
    validator = DuplicateCheckNameValidator()

    # Create some test nodes
    root = RootNode("test_suite")
    check1 = root.add_check("duplicate")
    check2 = root.add_check("duplicate")
    check3 = root.add_check("unique")

    # Process nodes
    validator.process_node(root)
    validator.process_node(check1)
    validator.process_node(check2)
    validator.process_node(check3)

    # Finalize to generate issues
    validator.finalize()

    # Should have found duplicate
    issues = validator.get_issues()
    assert len(issues) == 1
    assert "duplicate" in issues[0].message

    # Test reset
    validator.reset()

    # After reset, should have no issues
    assert len(validator.get_issues()) == 0

    # Internal state should also be cleared
    # Process the same nodes again
    validator.process_node(check1)
    validator.process_node(check2)

    # Before finalize, should have no issues yet
    assert len(validator.get_issues()) == 0

    # After finalize, should find the duplicate again
    validator.finalize()
    assert len(validator.get_issues()) == 1


@pytest.mark.asyncio
async def test_composite_validation_visitor_async() -> None:
    """Test async visit method of CompositeValidationVisitor."""

    class TestValidator(BaseValidator):
        """Simple test validator for testing."""

        name = "test"
        is_error = True

        def process_node(self, node: BaseNode) -> None:
            """Process nodes and track them."""
            if hasattr(node, "name") and node.name == "test_check":
                self._issues.append(ValidationIssue(rule=self.name, message="Found test check", node_path=["test"]))

    # Create composite visitor with test validator
    validators = [TestValidator()]
    visitor = CompositeValidationVisitor(validators)  # type: ignore[arg-type]

    # Create test node
    root = RootNode("test_suite")
    check = root.add_check("test_check")

    # Test async visit
    await visitor.visit_async(root)
    await visitor.visit_async(check)

    # Get issues - should find the test check
    issues = visitor.get_all_issues()
    assert len(issues["errors"]) == 1
    assert "Found test check" in issues["errors"][0].message


def test_composite_validation_visitor_reset() -> None:
    """Test reset method of CompositeValidationVisitor."""

    class TestValidator(BaseValidator):
        """Test validator that counts nodes."""

        name = "counter"
        is_error = False

        def __init__(self) -> None:
            super().__init__()
            self.node_count = 0

        def process_node(self, node: BaseNode) -> None:
            """Count nodes."""
            self.node_count += 1
            if self.node_count > 1:
                self._issues.append(
                    ValidationIssue(rule=self.name, message=f"Processed {self.node_count} nodes", node_path=["test"])
                )

        def reset(self) -> None:
            """Reset counter."""
            super().reset()
            self.node_count = 0

    # Create composite visitor
    validator = TestValidator()
    visitor = CompositeValidationVisitor([validator])

    # Process some nodes
    root = RootNode("suite1")
    check1 = root.add_check("check1")
    check2 = root.add_check("check2")

    visitor.visit(root)
    visitor.visit(check1)
    visitor.visit(check2)

    # Should have issues from processing multiple nodes
    issues = visitor.get_all_issues()
    assert len(issues["warnings"]) == 2  # 2 warnings: for 2nd and 3rd nodes

    # Test reset
    visitor.reset()

    # After reset, both visitor and validator should be reset
    assert validator.node_count == 0
    assert len(validator.get_issues()) == 0
    # The visitor's _nodes list should also be empty
    assert len(visitor._nodes) == 0

    # Now test that we can use it again after reset
    # Create a fresh validator to ensure clean state
    validator2 = TestValidator()
    visitor2 = CompositeValidationVisitor([validator2])

    # Process just two nodes - should generate one warning
    root2 = RootNode("suite2")
    check_new = root2.add_check("new_check")

    visitor2.visit(root2)  # First node, no warning
    visitor2.visit(check_new)  # Second node, should generate warning

    issues2 = visitor2.get_all_issues()
    assert len(issues2["warnings"]) == 1
    assert "Processed 2 nodes" in issues2["warnings"][0].message


def test_composite_visitor_with_finalizing_validators() -> None:
    """Test CompositeValidationVisitor with validators that have finalize methods."""

    class FinalizingValidator(BaseValidator):
        """Validator that uses finalize method."""

        name = "finalizer"
        is_error = True

        def __init__(self) -> None:
            super().__init__()
            self.nodes_seen = 0

        def process_node(self, node: BaseNode) -> None:
            """Just count nodes."""
            self.nodes_seen += 1

        def finalize(self) -> None:
            """Generate issue based on total count."""
            if self.nodes_seen > 2:
                self._issues.append(
                    ValidationIssue(rule=self.name, message=f"Too many nodes: {self.nodes_seen}", node_path=["root"])
                )

    # Create visitor with finalizing validator
    validator = FinalizingValidator()
    visitor = CompositeValidationVisitor([validator])

    # Process several nodes
    root = RootNode("suite")
    check1 = root.add_check("check1")
    check2 = root.add_check("check2")
    check3 = root.add_check("check3")

    visitor.visit(root)
    visitor.visit(check1)
    visitor.visit(check2)
    visitor.visit(check3)

    # Get issues - finalize should have been called
    issues = visitor.get_all_issues()
    assert len(issues["errors"]) == 1
    assert "Too many nodes: 4" in issues["errors"][0].message


def test_composite_visitor_mixed_error_warning_validators() -> None:
    """Test CompositeValidationVisitor correctly categorizes errors vs warnings."""

    class ErrorValidator(BaseValidator):
        """Validator that produces errors."""

        name = "error_validator"
        is_error = True

        def process_node(self, node: BaseNode) -> None:
            if hasattr(node, "name"):
                self._issues.append(
                    ValidationIssue(rule=self.name, message=f"Error for {node.name}", node_path=["error"])
                )

    class WarningValidator(BaseValidator):
        """Validator that produces warnings."""

        name = "warning_validator"
        is_error = False

        def process_node(self, node: BaseNode) -> None:
            if hasattr(node, "name"):
                self._issues.append(
                    ValidationIssue(rule=self.name, message=f"Warning for {node.name}", node_path=["warning"])
                )

    # Create visitor with both types
    visitor = CompositeValidationVisitor([ErrorValidator(), WarningValidator()])

    # Process a node
    root = RootNode("test")
    visitor.visit(root)

    # Get issues
    issues = visitor.get_all_issues()

    # Should have one error and one warning
    assert len(issues["errors"]) == 1
    assert len(issues["warnings"]) == 1
    assert "Error for test" in issues["errors"][0].message
    assert "Warning for test" in issues["warnings"][0].message
