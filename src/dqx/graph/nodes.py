from __future__ import annotations
from returns.result import Result

import sympy as sp
from dqx.common import DQXError, SeverityLevel, SymbolicValidator
from dqx.graph.base import BaseNode, CompositeNode
from dqx.provider import SymbolicMetric


class RootNode(CompositeNode["CheckNode"]):
    """Root node of the verification graph hierarchy.

    The RootNode serves as the top-level container in the data quality verification
    graph structure. It manages a collection of CheckNodes and orchestrates dataset
    propagation and graph traversal operations.

    Key responsibilities:
        - Maintains the graph structure with CheckNodes as direct children
        - Provides traversal methods to iterate over checks and assertions
        - Propagates dataset information through the graph hierarchy
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
        >>> # Propagate datasets
        >>> root.impute_datasets(['dataset1', 'dataset2'])
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

    def impute_datasets(self, datasets: list[str]) -> None:
        """Propagate dataset information through the graph hierarchy.

        Distributes dataset availability information to all checks and their
        assertions in the graph. Each check validates whether it can run with
        the provided datasets, and propagates this information to its assertions.

        This method is typically called after datasets are loaded and before
        evaluation begins. It ensures that all nodes in the graph are aware
        of which datasets are available for computation.

        The propagation follows these steps:
            1. Validate that at least one dataset is provided
            2. Call impute_datasets on each child check
            3. Each check validates its dataset requirements
            4. Each check propagates to its child assertions

        Args:
            datasets: List of dataset names that are available for computation.
                Must contain at least one dataset name. These names should
                correspond to actual loaded datasets in the system.

        Raises:
            DQXError: If the datasets list is empty.

        Examples:
            >>> root = RootNode("suite", context)
            >>> # After adding checks and assertions...
            >>>
            >>> # Single dataset
            >>> root.impute_datasets(['production_data'])
            >>>
            >>> # Multiple datasets
            >>> root.impute_datasets(['train_data', 'test_data', 'validation_data'])
            >>>
            >>> # This will raise an error
            >>> try:
            ...     root.impute_datasets([])
            ... except DQXError as e:
            ...     print(f"Error: {e}")
        """
        if not datasets:
            raise DQXError("At least one dataset must be provided!")

        # Propagate datasets to checks
        for check in self.children:
            check.impute_datasets(datasets)


class CheckNode(CompositeNode["AssertionNode"]):
    """
    Node representing a data quality check.

    CheckNode manages a collection of AssertionNode children and derives
    its state from their evaluation results. It no longer tracks symbols
    directly - instead, symbols are managed by the individual assertions.
    """

    def __init__(
        self,
        name: str,
        tags: list[str] | None = None,
        label: str | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        """
        Initialize a check node.

        Args:
            name: Unique identifier for the check
            tags: Optional tags for categorizing the check
            label: Optional human-readable label
            datasets: Optional list of datasets this check applies to
        """
        super().__init__()
        self.name = name
        self.tags = tags or []
        self.label = label
        self.datasets = datasets or []

    def node_name(self) -> str:
        """Get the display name of the node."""
        return self.label or self.name

    def impute_datasets(self, datasets: list[str]) -> None:
        """Validate and set datasets for this check."""
        if not datasets:
            raise DQXError("At least one dataset must be provided!")

        # No datasets set yet, so impute with provided datasets
        if len(self.datasets) == 0:
            self.datasets = datasets
        elif any(ds not in datasets for ds in self.datasets):
            # Validate existing datasets
            # self._value = Some(f"The check {self.node_name()} requires datasets {self.datasets} but got {datasets}")
            # In case of error, do not propagate datasets to children
            return

        # Always propagate the datasets to assertions
        for child in self.children:
            child.impute_datasets(self.datasets)


class AssertionNode(BaseNode):
    """
    Node representing an assertion to be evaluated.

    AssertionNodes are leaf nodes and cannot have children.
    They implement SymbolStateObserver to monitor the state of symbols
    used in their expressions.
    """

    def __init__(
        self,
        actual: sp.Expr,
        label: str | None = None,
        severity: SeverityLevel | None = None,
        validator: SymbolicValidator | None = None,
    ) -> None:
        """
        Initialize an assertion node.

        Args:
            actual: The symbolic expression to evaluate
            label: Optional human-readable description
            severity: Optional severity level for failures
            validator: Optional validation function to apply
        """
        super().__init__()
        self.actual = actual
        self.label = label
        self.severity = severity
        self.datasets: list[str] = []
        self.validator = validator
        self._value: Result[float, dict[SymbolicMetric | sp.Expr, str]]

    def impute_datasets(self, datasets: list[str]) -> None:
        """Validate and set datasets for this assertion."""
        if not datasets:
            raise DQXError("At least one dataset must be provided!")

        if not self.datasets:
            self.datasets = datasets
        else:
            if any(ds not in datasets for ds in self.datasets):
                raise DQXError(
                    f"The assertion {str(self.actual) or self.label} requires datasets {self.datasets} but got {datasets}"
                )

    def is_leaf(self) -> bool:
        return True
