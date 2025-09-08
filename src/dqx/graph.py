from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Generic, Protocol, TypeVar, overload, TYPE_CHECKING

import sympy as sp
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from dqx.common import DQXError, SeverityLevel, SymbolicValidator
from dqx.symbol_table import SymbolTable

# Type definitions
TChild = TypeVar("TChild", bound="BaseNode")
TNode = TypeVar("TNode", bound="BaseNode")


def aggregate_children_status(children: list["AssertionNode"]) -> Maybe[str]:
    """
    Aggregate validation status from assertion children.

    Returns:
        Nothing: All assertions passed or some are still pending
        Some(str): Error message if any assertion failed
    """
    failure_messages = []

    for child in children:
        if hasattr(child, "_value") and isinstance(child._value, Some):
            result = child._value.unwrap()
            if isinstance(result, Failure):
                # Collect failure message
                if hasattr(child, "label") and child.label:
                    failure_messages.append(f"{child.label}: {result.failure()}")
                else:
                    failure_messages.append(result.failure())

    # If any failures found, return aggregated error message
    if failure_messages:
        if len(failure_messages) == 1:
            return Some(failure_messages[0])
        else:
            return Some(f"Multiple failures: {'; '.join(failure_messages)}")

    # Otherwise return Nothing (success or pending)
    return Nothing


if TYPE_CHECKING:
    from dqx.api import Context  # Avoid circular import


# Base Node Classes
class BaseNode:
    """Base class for all nodes in the graph.

    This class provides the fundamental functionality for nodes in the graph,
    including parent-child relationships and graph traversal capabilities.
    Nodes that cannot have children (like AssertionNode) inherit directly
    from this class, while nodes that can have children inherit from
    CompositeNode which extends this class with child management functionality.
    """

    def __init__(self) -> None:
        """Initialize a base node with no parent and no cached root."""
        self._root: RootNode | None = None
        self.parent: BaseNode | None = None

    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.parent is None

    @property
    def root(self) -> RootNode:
        """Get the root node of the graph."""
        if self._root:
            return self._root

        current: BaseNode = self
        while current.parent:
            current = current.parent

        # Verify that we found an actual RootNode
        if not isinstance(current, RootNode):
            raise RuntimeError("Node is not attached to a graph")

        self._root = current
        return self._root

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        raise NotImplementedError("Subclasses must implement format_display")

    def accept(self, visitor: NodeVisitor) -> Any:
        """Accept a visitor for traversal."""
        return visitor.visit(self)


class CompositeNode(BaseNode, Generic[TChild]):
    """Base class for nodes that can have children.

    CompositeNode implements the Composite pattern, allowing nodes to contain
    and manage child nodes. This creates a tree structure where composite nodes
    can have zero or more children of a specific type.

    The class is generic over TChild, which must be a subtype of BaseNode,
    allowing for type-safe child management while maintaining flexibility
    in the specific types of children a composite can contain.

    Attributes:
        children: A list of child nodes of type TChild. Initialized as an
            empty list and can be populated using add_child method.
        _parent: Reference to the parent node, None for root nodes.
    """

    def __init__(self) -> None:
        """Initialize a composite node with an empty children list."""
        super().__init__()
        self.children: list[TChild] = []  # Instance attribute, not class attribute

    def add_child(self, child: TChild) -> None:
        """Add a child node to this composite.

        Appends the given child node to the end of the children list and
        sets this node as the child's parent.

        Args:
            child: The child node to add. Must be an instance of TChild,
                which is constrained to be a subtype of BaseNode.
        """
        self.children.append(child)

        # Set parent reference
        if hasattr(child, "parent"):
            child.parent = self

    def remove_child(self, child: TChild) -> None:
        """Remove a child node from this composite.

        Removes the first occurrence of the specified child from the children list
        and clears the child's parent reference.

        Args:
            child: The child node to remove. Must be an existing child
                in the children list.

        Raises:
            ValueError: If the child is not in the children list.
        """
        self.children.remove(child)
        # Clear parent reference
        if hasattr(child, "_parent"):
            child._parent = None  # type: ignore[attr-defined]

    def get_children(self) -> list[TChild]:
        """Get all children of this composite node.

        Returns:
            A list containing all child nodes. The list is a direct reference
            to the internal children list, not a copy. Modifications to the
            returned list will affect the composite's children.
        """
        return self.children

    def accept(self, visitor: NodeVisitor) -> Any:
        """Accept a visitor for the visitor pattern.

        This method is part of the visitor pattern implementation, allowing
        external operations to be performed on the node hierarchy without
        modifying the node classes themselves.

        Args:
            visitor: A NodeVisitor instance that will process this node.
                The visitor's visit method will be called with this node
                as an argument.

        Returns:
            The result of the visitor's visit method. The return type
            depends on the specific visitor implementation.
        """
        return visitor.visit(self)


# Visitor Pattern
class NodeVisitor(Protocol):
    """Protocol for visitor pattern."""

    def visit(self, node: BaseNode) -> Any:
        """Visit a node."""
        ...


class GraphTraverser:
    """Concrete visitor implementation for traversing the node graph.

    GraphTraverser implements the visitor pattern to traverse a graph of nodes,
    collecting nodes that match an optional type filter. It performs a depth-first
    traversal of the node hierarchy, visiting each node and recursively traversing
    into composite nodes to visit their children.

    The traverser can be configured to collect all nodes or only nodes of a
    specific type, making it useful for operations like finding all assertion
    nodes or any other specific node type within the graph.

    Attributes:
        filter_type: Optional type filter. When set, only nodes that are instances
            of this type will be collected during traversal. If None, all nodes
            are collected.
        results: List of nodes collected during traversal. Nodes are added in the
            order they are visited (depth-first traversal order).
    """

    def __init__(self, filter_type: type[BaseNode] | None = None):
        """Initialize a GraphTraverser with an optional type filter.

        Args:
            filter_type: Optional type to filter nodes during traversal. If provided,
                only nodes that are instances of this type will be collected in
                the results. If None, all visited nodes will be collected.
        """
        self.filter_type = filter_type
        self.results: list[BaseNode] = []

    def visit(self, node: BaseNode) -> None:
        """Visit a node and potentially traverse its children.

        This method implements the visitor pattern's visit operation. It checks
        if the current node matches the filter criteria and adds it to the results
        if it does. For composite nodes, it recursively visits all children to
        ensure complete traversal of the node hierarchy.

        The traversal is depth-first, meaning a node is visited before its children,
        and children are visited in the order they appear in the composite's
        children list.

        Args:
            node: The node to visit. Must implement the BaseNode protocol.
                If the node is a CompositeNode, its children will also be
                visited recursively.
        """
        if self.filter_type is None or isinstance(node, self.filter_type):
            self.results.append(node)

        # Continue traversal for composite nodes
        if isinstance(node, CompositeNode):
            for child in node.get_children():
                child.accept(self)


# Concrete Node Implementations
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

    def __init__(self, name: str, context: Context) -> None:
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
        self._context = context

    @property
    def symbol_table(self) -> SymbolTable:
        """Access the symbol table from the context.

        Provides read-only access to the SymbolTable instance managed by the
        Context. The symbol table contains all registered symbols and their
        metadata, computation functions, and current values.

        Returns:
            The SymbolTable instance from the context. Since context is required
            during initialization, this property always returns a valid SymbolTable.

        Examples:
            >>> root = RootNode("suite", context)
            >>> table = root.symbol_table
            >>> # Can now query symbols
            >>> entry = table.get(some_symbol)
            >>> if entry:
            ...     print(f"Symbol state: {entry.state}")
        """
        return self._context.symbol_table

    def format_display(self) -> str:
        """Generate a formatted string representation for display purposes.

        Creates a human-readable representation of the root node and its
        entire graph hierarchy using the RootNodeFormatter. This method is
        typically used for debugging, logging, or presenting the graph
        structure to users.

        The formatter is imported locally to avoid circular import issues
        between the graph and display modules.

        Returns:
            A formatted string showing the root node's name and hierarchical
            structure of all child checks and assertions.

        Examples:
            >>> root = RootNode("quality_suite", context)
            >>> # Add some checks...
            >>> print(root.format_display())
            RootNode: quality_suite
              CheckNode: completeness_check
                AssertionNode: non_null_count > 0
        """
        # Imported here to avoid circular import issues
        from dqx.display import RootNodeFormatter

        return RootNodeFormatter().format(self)

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

    @overload
    def traverse(self, filter_type: None = None) -> Iterator[BaseNode]: ...

    @overload
    def traverse(self, filter_type: type[TNode]) -> Iterator[TNode]: ...

    def traverse(self, filter_type: type[BaseNode] | None = None) -> Iterator[BaseNode]:
        """Traverse the graph with optional type filtering.

        Performs a depth-first traversal of the entire graph hierarchy starting
        from this root node. Can optionally filter results to only include nodes
        of a specific type.

        This method uses the visitor pattern with GraphTraverser to collect nodes.
        The traversal order is depth-first: each node is visited before its children,
        and children are visited in the order they were added.

        Args:
            filter_type: Optional type to filter results. If provided, only nodes
                that are instances of this type will be yielded. If None, all
                nodes in the graph are yielded. Common filters include CheckNode
                and AssertionNode.

        Yields:
            BaseNode instances matching the filter criteria. If filter_type is
            specified, the yielded nodes are guaranteed to be instances of that type.

        Examples:
            >>> root = RootNode("suite", context)
            >>> # Add some checks and assertions...
            >>>
            >>> # Traverse all nodes
            >>> for node in root.traverse():
            ...     print(node.format_display())
            >>>
            >>> # Traverse only CheckNodes
            >>> for check in root.traverse(CheckNode):
            ...     print(f"Check: {check.name}")
        """
        traverser = GraphTraverser(filter_type)
        self.accept(traverser)
        yield from traverser.results

    def assertions(self) -> Iterator[AssertionNode]:
        """Iterate over all assertion nodes in the graph.

        Convenience method that traverses the entire graph and yields only
        AssertionNode instances. This includes assertions nested within all
        checks at any depth in the hierarchy.

        The assertions are yielded in depth-first order, meaning assertions
        from earlier checks are yielded before assertions from later checks.

        Yields:
            AssertionNode instances representing all assertions in the graph.

        Examples:
            >>> root = RootNode("suite", context)
            >>> # After building the graph...
            >>>
            >>> # Count total assertions
            >>> total = sum(1 for _ in root.assertions())
            >>> print(f"Total assertions: {total}")
            >>>
            >>> # Evaluate all assertions
            >>> for assertion in root.assertions():
            ...     result = assertion.evaluate()
            ...     if isinstance(result, Failure):
            ...         print(f"Failed: {assertion.label}")
        """
        return self.traverse(AssertionNode)

    def checks(self) -> Iterator[CheckNode]:
        """Iterate over all check nodes in the graph.

        Convenience method that returns an iterator over all CheckNode instances
        that are direct children of this root. This does not include nested
        checks if any exist deeper in the hierarchy.

        Yields:
            CheckNode instances that are direct children of this root.

        Examples:
            >>> root = RootNode("suite", context)
            >>> check1 = CheckNode("check1")
            >>> check2 = CheckNode("check2")
            >>> root.add_child(check1)
            >>> root.add_child(check2)
            >>>
            >>> # List all checks
            >>> for check in root.checks():
            ...     print(f"- {check.name}: {len(check.children)} assertions")
        """
        return self.traverse(CheckNode)

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
        self._value: Maybe[str] = Nothing

    @property
    def context(self) -> Any:
        """
        Get context from the root node.

        Returns:
            The Context instance from the root node.
        """
        return self.root._context

    @property
    def symbol_table(self) -> SymbolTable:
        """
        Get symbol table from context.

        Returns:
            The SymbolTable from the context accessed through the root node.
        """
        return self.context.symbol_table

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        from dqx.display import CheckNodeFormatter

        return CheckNodeFormatter().format(self)

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
            self._value = Some(f"The check {self.node_name()} requires datasets {self.datasets} but got {datasets}")
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
        self._value: Maybe[Result[float, str]] = Nothing

    @property
    def context(self) -> Any:
        """
        Get context from the root node.

        Returns:
            The Context instance from the root node.
        """
        return self.root._context

    @property
    def symbol_table(self) -> SymbolTable:
        """
        Get symbol table from context.

        Returns:
            The SymbolTable from the context accessed through the root node.
        """
        return self.context.symbol_table

    def set_label(self, label: str) -> None:
        self.label = label

    def set_severity(self, severity: SeverityLevel) -> None:
        self.severity = severity

    def set_validator(self, validator: SymbolicValidator) -> None:
        self.validator = validator

    def impute_datasets(self, datasets: list[str]) -> None:
        """Validate and set datasets for this assertion."""
        if not datasets:
            raise DQXError("At least one dataset must be provided!")

        if not self.datasets:
            self.datasets = datasets
        else:
            if any(ds not in datasets for ds in self.datasets):
                self._value = Some(
                    Failure(
                        f"The assertion {str(self.actual) or self.label} requires datasets {self.datasets} but got {datasets}"
                    )
                )
                return

    def mark_as_failure(self, message: str) -> None:
        """Mark this assertion as failed with a message."""
        self._value = Some(Failure(message))

    def _find_parent_check(self) -> CheckNode:
        """Find the parent CheckNode by traversing up from the assertion."""
        # Use the parent property to get the direct parent
        if isinstance(self.parent, CheckNode):
            return self.parent  # type: ignore[return-value]

        # This should not happen if the graph is well-formed
        raise RuntimeError("AssertionNode is not attached to a CheckNode")

    def evaluate(self) -> Result[Any, str]:
        """
        Evaluate the assertion expression using SymbolTable.

        Returns:
            Result containing the evaluated value or error message.
        """
        import math
        
        # Get symbol values from SymbolTable
        symbol_table = self.symbol_table

        symbol_values = {}
        failed_symbols = []
        missing_symbols = []

        # Check all free symbols in the expression
        for symbol in self.actual.free_symbols:
            entry = symbol_table.get(symbol)
            
            if entry is None:
                missing_symbols.append(f"{symbol} (not found)")
                continue

            # Check if symbol has a value
            if entry.value is not None:
                if isinstance(entry.value, Success):
                    symbol_values[symbol] = entry.value.unwrap()
                else:
                    failed_symbols.append(str(symbol))
            else:
                # Symbol is pending - not yet evaluated
                missing_symbols.append(f"{symbol} (pending)")

        # Check for failed symbols
        if failed_symbols:
            self._value = Some(Failure(f"Symbol dependencies failed: {', '.join(failed_symbols)}"))
            return self._value.unwrap()

        # Check for missing symbols
        if missing_symbols:
            self._value = Some(Failure(f"Missing symbols: {', '.join(missing_symbols)}"))
            return self._value.unwrap()

        # Evaluate the expression
        try:
            value = float(sp.N(self.actual.subs(symbol_values), 6))

            if math.isnan(value):
                self._value = Some(Failure("Validating value is NaN"))
            elif math.isinf(value):
                self._value = Some(Failure("Validating value is infinity"))
            else:
                # Apply validator if present
                if self.validator and self.validator.fn:
                    if self.validator.fn(value):
                        self._value = Some(Success(value))
                    else:
                        failure_msg = (
                            f"Assertion failed: {self.actual} = {value} does not satisfy {self.validator.name}"
                        )
                        if self.label:
                            failure_msg = f"{self.label}: {failure_msg}"
                        self._value = Some(Failure(failure_msg))
                else:
                    # No validator, just store the computed value
                    self._value = Some(Success(value))

        except Exception as e:
            self._value = Some(Failure(str(e)))

        return self._value.unwrap()

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        from dqx.display import AssertionNodeFormatter

        return AssertionNodeFormatter().format(self)

    def add_child(self, child: Any) -> None:
        """AssertionNode should not have children."""
        raise RuntimeError("AssertionNode cannot have children.")
