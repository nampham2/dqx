from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable, overload

import sympy as sp
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from dqx.common import DQXError, SeverityLevel, SymbolicValidator
from dqx.symbol_table import SymbolTable

# Type definitions
TChild = TypeVar("TChild", bound="BaseNode")
TNode = TypeVar("TNode", bound="BaseNode")


# Base Node Protocols and Classes
@runtime_checkable
class BaseNode(Protocol):
    """Base protocol for all nodes in the graph."""

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        ...

    def accept(self, visitor: NodeVisitor) -> Any:
        """Accept a visitor for traversal."""
        ...


class LeafNode(BaseNode):
    """Base class for nodes that cannot have children."""

    def accept(self, visitor: NodeVisitor) -> Any:
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
    """

    def __init__(self) -> None:
        """Initialize a CompositeNode with an empty children list.

        Creates a new CompositeNode instance with no children. Children can
        be added later using the add_child method.
        """
        self.children: list[TChild] = []

    def add_child(self, child: TChild) -> None:
        """Add a child node to this composite.

        Appends the given child node to the end of the children list.
        The child must be of the type specified by the generic parameter TChild.

        Args:
            child: The child node to add. Must be an instance of TChild,
                which is constrained to be a subtype of BaseNode.
        """
        self.children.append(child)

    def remove_child(self, child: TChild) -> None:
        """Remove a child node from this composite.

        Removes the first occurrence of the specified child from the children list.

        Args:
            child: The child node to remove. Must be an existing child
                in the children list.

        Raises:
            ValueError: If the child is not in the children list.
        """
        self.children.remove(child)

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
    """Root node of the verification graph."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.symbol_table = SymbolTable()

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        # Imported here to avoid circular import issues
        from dqx.display import RootNodeFormatter

        return RootNodeFormatter().format(self)

    def exists(self, child: "CheckNode") -> bool:
        """Check if a child node exists in the graph."""
        return child in self.children

    @overload
    def traverse(self, filter_type: None = None) -> Iterator[BaseNode]: ...

    @overload
    def traverse(self, filter_type: type[TNode]) -> Iterator[TNode]: ...

    def traverse(self, filter_type: type[BaseNode] | None = None) -> Iterator[BaseNode]:
        """Generic traversal with optional type filtering."""
        traverser = GraphTraverser(filter_type)
        self.accept(traverser)
        yield from traverser.results

    def assertions(self) -> Iterator[AssertionNode]:
        """Iterate over all assertion nodes."""
        return self.traverse(AssertionNode)

    def checks(self) -> Iterator[CheckNode]:
        """Iterate over all check nodes."""
        return self.traverse(CheckNode)

    def impute_datasets(self, datasets: list[str]) -> None:
        """
        Propagate dataset information through the graph.

        1. Impute check's datasets with the provided datasets
        2. Propagate datasets from checks to assertions
        3. Validate and propagate datasets in SymbolTable
        4. Back propagate any failures to assertions
        """
        if not datasets:
            raise DQXError("At least one dataset must be provided!")

        # First propagate through the graph structure (checks and assertions)
        for check in self.children:
            check.impute_datasets(datasets)

        # Then validate datasets in the symbol table
        errors = self.symbol_table.validate_datasets(datasets)

        # Back propagate any symbol table errors to assertions
        if errors:
            for error_msg in errors:
                # Find assertions that use the failed symbols
                for assertion in self.assertions():
                    # Check if any of the assertion's symbols had dataset errors
                    for symbol in assertion.actual.free_symbols:
                        if str(symbol) in error_msg:
                            assertion.mark_as_failure(error_msg)


class CheckNode(CompositeNode["AssertionNode"]):
    """Node representing a data quality check."""

    def __init__(
        self,
        name: str,
        tags: list[str] | None = None,
        label: str | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.tags = tags or []
        self.label = label
        self.datasets = datasets or []
        self._value: Maybe[Result[float, str]] = Nothing

        # Track which symbols belong to this check
        self.symbols: set[sp.Symbol] = set()

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
        # Validate existing datasets
        elif any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(f"The check {self.node_name()} requires datasets {self.datasets} but got {datasets}")
            )

        # Always propagate the datasets to assertions
        for child in self.children:
            child.impute_datasets(self.datasets)

    def update_status(self) -> None:
        """Update the check's status based on the status of its children."""
        # If already failed (e.g., dataset mismatch), don't update
        if isinstance(self._value, Some) and isinstance(self._value.unwrap(), Failure):
            return

        # Collect status of all children
        all_success = True
        any_failure = False
        failure_messages = []

        for child in self.children:
            if hasattr(child, "_value"):
                if isinstance(child._value, Some):
                    result = child._value.unwrap()
                    if isinstance(result, Failure):
                        any_failure = True
                        all_success = False
                        if hasattr(child, "label") and child.label:
                            failure_messages.append(f"{child.label}: {result.failure()}")
                        else:
                            failure_messages.append(result.failure())
                    # Success case - continue checking
                else:
                    # Child is pending (Nothing)
                    all_success = False

        # Update check status based on children
        if any_failure:
            # At least one child failed
            if len(failure_messages) == 1:
                self._value = Some(Failure(failure_messages[0]))
            else:
                self._value = Some(Failure(f"Multiple failures: {'; '.join(failure_messages)}"))
        elif all_success and len(self.children) > 0:
            # All children succeeded (and we have at least one child)
            self._value = Some(Success(1.0))
        # Otherwise, keep as Nothing (pending)

    def add_symbol(self, symbol: sp.Symbol) -> None:
        """Add a symbol reference to this check."""
        self.symbols.add(symbol)


class AssertionNode(LeafNode):
    """
    Node representing an assertion to be evaluated.

    AssertionNodes are leaf nodes and cannot have children.
    """

    def __init__(
        self,
        actual: sp.Expr,
        label: str | None = None,
        severity: SeverityLevel | None = None,
        validator: SymbolicValidator | None = None,
        root: RootNode | None = None,
    ) -> None:
        self.actual = actual
        self.label = label
        self.severity = severity
        self.datasets: list[str] = []
        self.validator = validator
        self._value: Maybe[Result[float, str]] = Nothing
        self._root = root

    def set_label(self, label: str) -> None:
        self.label = label

    def set_severity(self, severity: SeverityLevel) -> None:
        self.severity = severity

    def set_validator(self, validator: SymbolicValidator) -> None:
        self.validator = validator

    def set_datasource(self, datasets: list[str]) -> None:
        self.datasets = datasets

    def impute_datasets(self, datasets: list[str]) -> None:
        """Validate and set datasets for this assertion."""
        if not self.datasets:
            self.datasets = datasets
            return

        if any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(
                    f"The assertion {str(self.actual) or self.label} requires datasets {self.datasets} but got {datasets}"
                )
            )

    def mark_as_failure(self, message: str) -> None:
        """Mark this assertion as failed with a message."""
        self._value = Some(Failure(message))

    def _find_parent_check(self) -> CheckNode | None:
        """Find the parent CheckNode by traversing up from the assertion."""
        if self._root is None:
            return None

        # Traverse all check nodes and find the one containing this assertion
        for check in self._root.checks():
            if self in check.children:
                return check
        return None

    def evaluate(self) -> Result[Any, str]:
        """
        Evaluate the assertion expression using SymbolTable.

        Returns:
            Result containing the evaluated value or error message.
        """
        if self._root is None:
            raise RuntimeError("Root node not set for AssertionNode")

        # First check if parent CheckNode has failed
        parent_check = self._find_parent_check()
        if parent_check and isinstance(parent_check._value, Some):
            parent_result = parent_check._value.unwrap()
            if isinstance(parent_result, Failure):
                self._value = Some(Failure("Parent check failed!"))
                return self._value.unwrap()

        # Get symbol values from SymbolTable
        symbol_table = self._root.symbol_table
        symbol_values = {}
        failed_symbols = []
        missing_symbols = []

        # Check all free symbols in the expression
        for symbol in self.actual.free_symbols:
            entry = symbol_table.get(symbol)

            if entry is None:
                missing_symbols.append(str(symbol))
            else:
                # Check if symbol has a value
                if isinstance(entry.value, Some):
                    result = entry.value.unwrap()
                    if isinstance(result, Success):
                        symbol_values[symbol] = result.unwrap()
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
            value = sp.N(self.actual.subs(symbol_values), 6)

            if math.isnan(value):
                self._value = Some(Failure("Validating value is NaN"))
            elif math.isinf(value):
                self._value = Some(Failure("Validating value is infinity"))
            else:
                # Apply validator if present
                if self.validator and self.validator.fn:
                    if self.validator.fn(float(value)):
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

    def _find_root(self) -> RootNode:
        """Find the root node by traversing up the graph."""
        if self._root is None:
            raise RuntimeError("Root node not set for AssertionNode")
        return self._root

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        from dqx.display import AssertionNodeFormatter

        return AssertionNodeFormatter().format(self)

    def add_child(self, child: Any) -> None:
        """AssertionNode should not have children."""
        raise RuntimeError("AssertionNode cannot have children.")
