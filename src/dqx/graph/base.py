from __future__ import annotations

from typing import Generic, Protocol, TypeVar, Union

from dqx.common import DQXError

# Type variables
TParent = TypeVar("TParent", bound=Union["BaseNode", None])
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


class BaseNode(Generic[TParent]):
    """Base class for all nodes in the graph.

    Now generic over TParent to enable strongly typed parent relationships.
    Each node type can specify exactly what type its parent should be.
    """

    def __init__(self, parent: TParent) -> None:
        """Initialize a base node with its parent.

        Args:
            parent: The parent node. Type depends on the specific node class.
                   RootNode has None parent, all others have specific parent types.
        """
        self._parent: TParent = parent
        self._validate_parent_type(parent)

    def _validate_parent_type(self, parent: TParent) -> None:
        """Validate parent type at runtime for extra safety.

        This provides runtime enforcement of the type hierarchy even if
        type checking is bypassed.
        """
        # Import here to avoid circular imports
        from dqx.graph.nodes import CheckNode, RootNode

        class_name = self.__class__.__name__

        if class_name == "RootNode" and parent is not None:
            raise TypeError(f"RootNode must have None as parent, but got {type(parent).__name__}")
        elif class_name == "CheckNode" and not isinstance(parent, RootNode):
            parent_type = type(parent).__name__ if parent is not None else "None"
            raise TypeError(f"CheckNode requires parent of type RootNode, but got {parent_type}")
        elif class_name == "AssertionNode" and not isinstance(parent, CheckNode):
            parent_type = type(parent).__name__ if parent is not None else "None"
            raise TypeError(f"AssertionNode requires parent of type CheckNode, but got {parent_type}")

    @property
    def parent(self) -> TParent:
        """Get the parent node.

        The parent is immutable after construction to maintain hierarchy integrity.
        """
        return self._parent

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


class CompositeNode(BaseNode[TParent], Generic[TParent, TChild]):
    """Base class for nodes that can have children.

    Now generic over both TParent (parent type) and TChild (children type).
    """

    def __init__(self, parent: TParent) -> None:
        """Initialize a composite node with its parent."""
        super().__init__(parent)
        self.children: list[TChild] = []

    def is_leaf(self) -> bool:
        """Check if this node has children.

        This method is part of the Composite pattern and must be overridden by
        subclasses that can have children. It returns False, indicating that the
        node has children. If the node is a leaf node, it should return True.

        Returns:
            bool: True if the node is a leaf node and has no children, False otherwise.
        """
        return False

    def add_child(self, child: TChild) -> CompositeNode[TParent, TChild]:
        """Add a child node.

        Note: We no longer set child.parent here because the child
        already has its parent set in its constructor.

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
        # Don't set child.parent - it's already set in constructor!
        return self
