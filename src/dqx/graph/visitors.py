from typing import TypeVar, Generic
from dqx.graph.base import BaseNode


TNode = TypeVar("TNode", bound=BaseNode)


class NodeCollector(Generic[TNode]):
    """Visitor that collects nodes of a specific type during graph traversal.

    This class implements the visitor pattern to collect all nodes that match
    a specified type during graph traversal. It maintains a list of collected
    nodes that can be retrieved after traversal.

    Attributes:
        node_type: The type of BaseNode subclass to collect during traversal.
        results: List of collected nodes matching the specified type.

    Example:
        >>> from dqx.graph.nodes import SymbolNode
        >>> from dqx.graph.traversal import GraphTraversal
        >>>
        >>> # Create a collector for SymbolNode instances
        >>> collector = NodeCollector(SymbolNode)
        >>>
        >>> # Use it with graph traversal
        >>> traversal = GraphTraversal()
        >>> traversal.traverse(root_node, collector)
        >>>
        >>> # Access collected nodes
        >>> symbol_nodes = collector.results
    """

    def __init__(self, node_type: type[TNode]) -> None:
        """Initialize a NodeCollector for a specific node type.

        Args:
            node_type: The type of BaseNode subclass to collect. Only nodes
                that are instances of this type will be collected during
                traversal.
        """
        self.node_type = node_type
        self.results: list[TNode] = []

    def visit(self, node: BaseNode) -> None:
        """Visit a node and collect it if it matches the target type.

        This method is called by the graph traversal mechanism for each
        node in the graph. If the node is an instance of the specified
        node_type, it will be added to the results list.

        Args:
            node: The node to visit and potentially collect.
        """
        if isinstance(node, self.node_type):
            self.results.append(node)

    async def visit_async(self, node: BaseNode) -> None:
        """Asynchronously visit a node and collect it if it matches the target type.

        This method is called by the asynchronous graph traversal mechanism
        for each node in the graph. If the node is an instance of the specified
        node_type, it will be added to the results list.

        Args:
            node: The node to visit and potentially collect.
        """
        self.visit(node)
