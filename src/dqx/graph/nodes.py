from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp
from returns.result import Result

from dqx.common import AssertionStatus, EvaluationFailure, SeverityLevel, SymbolicValidator
from dqx.graph.base import BaseNode, CompositeNode

if TYPE_CHECKING:
    from dqx.tunables import Tunable


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
        tags: frozenset[str] | None = None,
        experimental: bool = False,
        required: bool = False,
        cost_fp: float | None = None,
        cost_fn: float | None = None,
        tunables: dict[str, Tunable[Any]] | None = None,
    ) -> AssertionNode:
        """
        Create and attach an AssertionNode as a child of this CheckNode.

        Args:
                actual (sp.Expr): Symbolic expression to evaluate.
                name (str): Human-readable description for the assertion.
                validator (SymbolicValidator): Function that validates the evaluated expression.
                severity (SeverityLevel): Severity level to assign on failure (default "P1").
                tags (frozenset[str] | None): Tags used for selecting or grouping assertions.
                experimental (bool): Mark the assertion as algorithm-proposed (default False).
                required (bool): Mark the assertion as non-removable by algorithms (default False).
                cost_fp (float | None): Cost assigned to a false positive for reward computations.
                cost_fn (float | None): Cost assigned to a false negative for reward computations.
                tunables (dict[str, Tunable[Any]] | None): Optional mapping of tunable names to Tunable objects
                        used in validator closures (for tunables not in symbolic expressions).

        Returns:
                AssertionNode: The newly created and attached assertion node.
        """
        assertion = AssertionNode(
            parent=self,
            actual=actual,
            name=name,
            validator=validator,
            severity=severity,
            tags=tags,
            experimental=experimental,
            required=required,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            tunables=tunables,
        )
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
        tags: frozenset[str] | None = None,
        experimental: bool = False,
        required: bool = False,
        cost_fp: float | None = None,
        cost_fn: float | None = None,
        tunables: dict[str, Tunable[Any]] | None = None,
    ) -> None:
        """
        Create an AssertionNode that encapsulates a symbolic expression, its validator, and assertion metadata.

        Args:
            parent: The parent CheckNode that contains this assertion.
            actual: The symbolic expression to evaluate for this assertion.
            name: Human-readable description of the assertion.
            validator: Function that validates the evaluated metric against expectations.
            severity: Severity level to assign when the assertion fails.
            tags: Optional set of tags used to select or group assertions.
            experimental: If True, marks the assertion as proposed by automated algorithms.
            required: If True, marks the assertion as non-removable by automated algorithms.
            cost_fp: Optional cost to attribute to false positives (used for reward/cost calculations).
            cost_fn: Optional cost to attribute to false negatives (used for reward/cost calculations).
            tunables: Optional mapping of tunable names to Tunable objects used in validator closures
                     (for tunables not in symbolic expressions).
        """
        super().__init__(parent)
        self.actual = actual
        self.name = name
        self.severity = severity
        self.validator = validator
        self.tags: frozenset[str] = tags or frozenset()
        self.experimental = experimental
        self.required = required
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.tunables: dict[str, Tunable[Any]] = tunables or {}
        # Stores the computed metric result
        self._metric: Result[float, list[EvaluationFailure]]
        # Stores whether the assertion passes validation
        self._result: AssertionStatus
        # Stores the effective severity (after profile override)
        self._effective_severity: SeverityLevel | None = None

    def is_leaf(self) -> bool:
        return True
