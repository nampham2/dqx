"""Display module for graph visualization using Rich."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
from rich.tree import Tree
from rich.console import Console

if TYPE_CHECKING:
    from dqx.graph.base import BaseNode
    from dqx.graph.traversal import Graph


class NodeFormatter(Protocol):
    """Protocol for formatting graph nodes for display."""
    
    def format_node(self, node: BaseNode) -> str:
        """Format a node for display.
        
        Args:
            node: The node to format.
            
        Returns:
            A string representation of the node.
        """
        ...


class SimpleNodeFormatter:
    """Simple formatter that displays node label, name, or class name."""
    
    def format_node(self, node: BaseNode) -> str:
        """Format a node using priority: node_name() -> label -> name -> class name.
        
        Args:
            node: The node to format.
            
        Returns:
            A string representation of the node.
        """
        # First check for node_name() method (CheckNode has this)
        if hasattr(node, 'node_name') and callable(node.node_name):
            return node.node_name()
        
        # Then check for label attribute
        if hasattr(node, 'label') and node.label:
            return node.label
        
        # Then check for name attribute
        if hasattr(node, 'name') and node.name:
            return node.name
        
        # Finally, use class name
        return node.__class__.__name__


class TreeBuilderVisitor:
    """Visitor that builds a Rich Tree during graph traversal."""
    
    def __init__(self, formatter: NodeFormatter) -> None:
        """Initialize the visitor with a node formatter.
        
        Args:
            formatter: The formatter to use for node labels.
        """
        self._formatter = formatter
        self.tree: Tree | None = None

        # Store the corresponding tree parent of a graph node
        # so that we can add children correctly
        self.parent_map: dict[BaseNode, Tree] = {}
    
    def visit(self, node: BaseNode) -> None:
        """Visit a node and add it to the tree.
        
        Args:
            node: The node to visit.
            
        Raises:
            ValueError: If the node's parent was not visited before the child.
        """
        formatted_label = self._formatter.format_node(node)
        
        # This is the root node
        if node.is_root:
            self.tree = Tree(formatted_label)
            self.parent_map[node] = self.tree
        else:
            # Find parent's tree node
            if node.parent not in self.parent_map:
                raise ValueError(
                    f"Parent of node '{formatted_label}' was not visited before the child. "
                    f"This indicates an issue with the traversal order."
                )
            
            parent_tree = self.parent_map[node.parent]
            child_tree = parent_tree.add(formatted_label)
            self.parent_map[node] = child_tree
    
    async def visit_async(self, node: BaseNode) -> None:
        """Async visit method that delegates to sync visit.
        
        Args:
            node: The node to visit.
        """
        self.visit(node)


def print_graph(graph: Graph, formatter: NodeFormatter | None = None) -> None:
    """Print a graph structure as a tree to the console.
    
    Args:
        graph: The graph to print.
        formatter: Optional formatter for node labels. Defaults to SimpleNodeFormatter.
    """
    formatter = formatter or SimpleNodeFormatter()
    
    visitor = TreeBuilderVisitor(formatter)
    graph.dfs(visitor)
    
    if visitor.tree is not None:
        console = Console()
        console.print(visitor.tree)
