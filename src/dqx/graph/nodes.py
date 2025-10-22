from __future__ import annotations

import sympy as sp
from returns.result import Result

from dqx.common import AssertionStatus, EvaluationFailure, SeverityLevel, SymbolicValidator
from dqx.graph.base import BaseNode, CompositeNode


class RootNode(CompositeNode[None, "CheckNode"]):
    """Root node of the verification graph hierarchy.

    The RootNode serves as the top-level container in the data quality verification
    graph structure. It manages a collection of CheckNodes and orchestrates
    graph traversal operations.

    Key responsibilities:
        - Maintains the graph structure with CheckNodes as direct children
        - Provides traversal methods to iterate over checks and assertions
        - Accesses symbol table through the provided Context instance

    The RootNode follows a dependency injection pattern where it receives a Context
    instance rather than owning the SymbolTable directly. This design promotes
    better separation of concerns and avoids circular dependencies.

    Attributes:
        name: Human-readable name identifying this verification suite
        datasets: List of dataset names available in this suite (populated during imputation)
        _context: Reference to the Context instance that owns the symbol table

    Examples:
        >>> root = RootNode("data_quality_suite")
        >>>
        >>> # Add checks to the root using factory method
        >>> check = root.add_check("completeness_check")
        >>>
        >>> # Add assertions to the check
        >>> assertion = check.add_assertion(sp.Symbol("x"), name="x > 0")
    """

    def __init__(self, name: str) -> None:
        """Initialize a root node.

        Args:
            name: Human-readable name for the verification suite
        """
        super().__init__(parent=None)  # Root always has None parent
        self.name = name
        self.datasets: list[str] = []

    def add_check(
        self,
        name: str,
        datasets: list[str] | None = None,
    ) -> CheckNode:
        """Factory method to create and add a check node.

        This ensures the check has the correct parent type.

        Args:
            name: Name for the check
            datasets: Optional list of datasets this check applies to

        Returns:
            The newly created CheckNode
        """
        check = CheckNode(parent=self, name=name, datasets=datasets)
        self.add_child(check)
        return check


class CheckNode(CompositeNode["RootNode", "AssertionNode"]):
    """Node representing a data quality check.

    Parent type is RootNode (never None).
    Child type is AssertionNode.
    """

    def __init__(
        self,
        parent: RootNode,
        name: str,
        datasets: list[str] | None = None,
    ) -> None:
        """Initialize a check node.

        Args:
            parent: The RootNode parent (required)
            name: Name for the check
            datasets: Optional list of datasets this check applies to
        """
        super().__init__(parent)
        self.name = name
        self.datasets = datasets or []

    def add_assertion(
        self,
        actual: sp.Expr,
        name: str,
        validator: SymbolicValidator,
        severity: SeverityLevel = "P1",
    ) -> AssertionNode:
        """Factory method to create and add an assertion node.

        This ensures the assertion has the correct parent type.

        Args:
            actual: The symbolic expression to evaluate
            name: Optional human-readable description
            validator: Validation function
            severity: Severity level for failures

        Returns:
            The newly created AssertionNode
        """
        assertion = AssertionNode(parent=self, actual=actual, name=name, validator=validator, severity=severity)
        self.add_child(assertion)
        return assertion


class AssertionNode(BaseNode["CheckNode"]):
    """Node representing an assertion to be evaluated.

    Parent type is CheckNode (never None).
    AssertionNodes are leaf nodes and cannot have children.
    """

    def __init__(
        self,
        parent: CheckNode,
        actual: sp.Expr,
        name: str,
        validator: SymbolicValidator,
        severity: SeverityLevel = "P1",
    ) -> None:
        """Initialize an assertion node.

        Args:
            parent: The CheckNode parent (required)
            actual: The symbolic expression to evaluate
            name: Optional human-readable description
            validator: Validation function
            severity: Severity level for failures
        """
        super().__init__(parent)
        self.actual = actual
        self.name = name
        self.severity = severity
        self.validator = validator
        # Stores the computed metric result
        self._metric: Result[float, list[EvaluationFailure]]
        # Stores whether the assertion passes validation
        self._result: AssertionStatus

    def is_leaf(self) -> bool:
        return True
