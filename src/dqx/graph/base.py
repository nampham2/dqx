from __future__ import annotations
from typing import Generic, Protocol, TypeVar

from dqx.common import DQXError


TChild = TypeVar("TChild", bound="BaseNode")
TNode = TypeVar("TNode", bound="BaseNode")


class NodeVisitor(Protocol):
    """Protocol for visitor pattern."""

    def visit(self, node: BaseNode) -> None:
        """Visit a node synchronously.

        This method is part of the visitor pattern implementation, allowing
        external operations to be performed on the node hierarchy without
        modifying the node classes themselves.

        Args:
            node: The node to visit. Must implement the BaseNode protocol.

        Returns:
            The result of the visitor's visit method. The return type
            depends on the specific visitor implementation.
        """

    async def visit_async(self, node: BaseNode) -> None:
        """Visit a node asynchronously.

        This method is part of the asynchronous visitor pattern implementation,
        allowing external operations to be performed on the node hierarchy without
        modifying the node classes themselves.

        Args:
            node: A BaseNode instance that will be processed by the visitor.

        Returns:
            The result of the visitor's visit method. The return type depends on the
            specific visitor implementation.
        """


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
        self.parent: BaseNode | None = None

    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.parent is None

    def accept(self, visitor: NodeVisitor) -> None:
        """Accept a visitor for traversal."""
        return visitor.visit(self)

    async def accept_async(self, visitor: NodeVisitor) -> None:
        """Accept an asynchronous visitor for traversal."""
        return await visitor.visit_async(self)

    def is_leaf(self) -> bool:
        """Check if this node has children.

        This method should be overridden by subclasses that can have children.
        By default, it returns False, indicating that the node does not have
        any children.
        """
        raise NotImplementedError("Subclasses must implement has_children method.")


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

    def is_leaf(self) -> bool:
        """Check if this node has children.

        This method is part of the Composite pattern and must be overridden by
        subclasses that can have children. It returns False, indicating that the
        node has children. If the node is a leaf node, it should return True.

        Returns:
            bool: True if the node is a leaf node and has no children, False otherwise.
        """
        return False

    def add_child(self, child: TChild) -> CompositeNode[TChild]:
        """Add a child node and set its parent reference.

        Args:
            child: The child node to add

        Returns:
            Self for method chaining

        Raises:
            DQXError: If the child is already in the children list
        """
        if child in self.children:
            raise DQXError("Child node is already in the children list")

        self.children.append(child)
        child.parent = self
        return self
