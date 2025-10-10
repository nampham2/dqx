from __future__ import annotations

import sympy as sp
from returns.result import Result

from dqx.common import SeverityLevel, SymbolicValidator
from dqx.graph.base import BaseNode, CompositeNode
from dqx.provider import SymbolicMetric


class RootNode(CompositeNode["CheckNode"]):
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
        _context: Reference to the Context instance that owns the symbol table

    Examples:
        >>> from dqx.api import Context
        >>> context = Context()
        >>> root = RootNode("data_quality_suite", context)
        >>>
        >>> # Add checks to the root
        >>> check = CheckNode("completeness_check")
        >>> root.add_child(check)
        >>>
        >>> # Traverse all assertions
        >>> for assertion in root.assertions():
        ...     result = assertion.evaluate()
    """

    def __init__(self, name: str) -> None:
        """Initialize a root node with the given name and context.

        Creates a new RootNode instance that will serve as the top of the
        verification graph hierarchy. The node starts with an empty collection
        of children that can be populated with CheckNode instances.

        Args:
            name: Human-readable name for the verification suite. This should
                be descriptive of the suite's purpose, e.g., "user_data_quality"
                or "transaction_validation".
            context: The Context instance that provides access to the symbol table
                and other shared resources. This context is passed down to child
                nodes to ensure consistent access to shared state.

        Examples:
            >>> from dqx.api import Context
            >>> context = Context()
            >>> root = RootNode("my_quality_checks", context)
            >>> print(root.name)
            my_quality_checks
        """
        super().__init__()
        self.name = name

    def exists(self, child: "CheckNode") -> bool:
        """Check if a specific CheckNode exists as a direct child.

        Tests whether the given CheckNode instance is present in this root's
        immediate children. This is a shallow check that only looks at direct
        children, not nested assertions within checks.

        Args:
            child: The CheckNode instance to search for. Must be an exact
                object reference match, not just a node with the same name.

        Returns:
            True if the exact CheckNode instance is a direct child of this
            root, False otherwise.

        Examples:
            >>> root = RootNode("suite", context)
            >>> check = CheckNode("my_check")
            >>> root.add_child(check)
            >>> assert root.exists(check) == True
            >>>
            >>> other_check = CheckNode("other_check")
            >>> assert root.exists(other_check) == False
        """
        return child in self.children


class CheckNode(CompositeNode["AssertionNode"]):
    """
    Node representing a data quality check.

    CheckNode manages a collection of AssertionNode children and derives
    its state from their evaluation results.
    """

    def __init__(
        self,
        name: str,
        tags: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        """
        Initialize a check node.

        Args:
            name: Name for the check (either user-provided or function name)
            tags: Optional tags for categorizing the check
            datasets: Optional list of datasets this check applies to
        """
        super().__init__()
        self.name = name
        self.tags = tags or []
        self.datasets = datasets or []


class AssertionNode(BaseNode):
    """
    Node representing an assertion to be evaluated.

    AssertionNodes are leaf nodes and cannot have children.
    """

    def __init__(
        self,
        actual: sp.Expr,
        name: str | None = None,
        severity: SeverityLevel = "P1",
        validator: SymbolicValidator | None = None,
    ) -> None:
        """
        Initialize an assertion node.

        Args:
            actual: The symbolic expression to evaluate
            name: Optional human-readable description
            severity: Severity level for failures (P0, P1, P2, P3). Defaults to "P1".
            validator: Optional validation function to apply
        """
        super().__init__()
        self.actual = actual
        self.name = name
        self.severity = severity
        self.validator = validator
        self._value: Result[float, dict[SymbolicMetric | sp.Expr, str]]

    def is_leaf(self) -> bool:
        return True
