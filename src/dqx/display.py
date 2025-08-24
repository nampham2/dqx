"""Display and formatting utilities for graph visualization."""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, cast

from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success
from rich.tree import Tree

if TYPE_CHECKING:
    from dqx.graph import (
        AnalyzerNode,
        AssertionNode,
        BaseNode,
        CheckNode,
        MetricNode,
        RootNode,
        SymbolNode,
    )


# Local protocol to avoid circular import
class VisitorProtocol(Protocol):
    """Protocol for visitor pattern without importing NodeVisitor."""
    
    def visit(self, node: BaseNode) -> Any:
        """Visit a node."""
        ...


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_status(value: Maybe[Result[Any, str]], show_value: bool = False) -> str:
    """Format a Maybe[Result] value into a clean status indicator."""
    if value is Nothing:
        return "[yellow]⏳[/yellow]"
    elif isinstance(value, Some):
        result = value.unwrap()
        if isinstance(result, Success):
            val = result.unwrap()
            if show_value and val is not None:
                # Format numeric values nicely
                if isinstance(val, (int, float)):
                    if isinstance(val, float):
                        formatted_val = f"{val:.2f}" if abs(val) < 1000 else f"{val:.1f}"
                    else:
                        formatted_val = str(val)
                    return f"[green]{formatted_val}[/green] ✅"
                else:
                    return f"[green]{val}[/green] ✅"
            return "[green]✅[/green]"
        elif isinstance(result, Failure):
            return "[red]❌[/red]"
    return "[dim]❓[/dim]"


def format_error(message: str) -> str:
    """Format error messages in a clean, readable way."""
    # Truncate very long messages
    if len(message) > 100:
        message = message[:97] + "..."
    
    # Special formatting for common error patterns
    if "parent check failed:" in message:
        return "[yellow]⚠️  Skipped: parent check failed[/yellow]"
    elif "requires datasets" in message and "but got" in message:
        # Extract dataset info
        try:
            parts = message.split("requires datasets ")
            required = parts[1].split(" but got ")[0]
            return f"[red]❌ Dataset mismatch: needs {required}[/red]"
        except (IndexError, AttributeError):
            # If parsing fails, return the original message
            return f"[red]❌ {message}[/red]"
    elif "Missing symbols:" in message:
        return f"[red]❌ {message}[/red]"
    elif "Symbol dependencies failed:" in message:
        return f"[red]❌ {message}[/red]"
    elif "does not satisfy" in message:
        # Extract the key parts of validation failure
        try:
            parts = message.split(": ")
            if len(parts) > 1:
                return f"[red]❌ {parts[-1]}[/red]"
        except (IndexError, AttributeError):
            pass
    elif message in ["Validating value is NaN", "Validating value is infinity"]:
        return f"[red]❌ {message}[/red]"
    
    # Default formatting
    return f"[red]❌ {message}[/red]"


def format_datasets(datasets: list[str]) -> str:
    """Format dataset list in a compact way."""
    if not datasets:
        return ""
    if len(datasets) == 1:
        return f"[dim italic]{datasets[0]}[/dim italic]"
    return f"[dim italic]{', '.join(datasets)}[/dim italic]"


# =============================================================================
# Node Formatter Protocol and Implementations
# =============================================================================

class NodeFormatter(Protocol):
    """Protocol for node formatting."""
    
    def format(self, node: BaseNode) -> str:
        """Format a node for display."""
        ...


class RootNodeFormatter:
    """Formatter for RootNode."""
    
    def format(self, node: RootNode) -> str:
        """Format RootNode for display."""
        return f"Suite: {node.name}"


class CheckNodeFormatter:
    """Formatter for CheckNode."""
    
    def format(self, node: CheckNode) -> str:
        """Format CheckNode for display."""
        name = node.label or node.name
        status = format_status(node._value)
        datasets = format_datasets(node.datasets)
        if datasets:
            return f"📋 [bold cyan]{name}[/bold cyan] [{datasets}] {status}"
        return f"📋 [bold cyan]{name}[/bold cyan] {status}"


class AssertionNodeFormatter:
    """Formatter for AssertionNode."""
    
    def format(self, node: AssertionNode) -> str:
        """Format AssertionNode for display."""
        if node.validator:
            datasets = format_datasets(node.datasets)
            
            # Determine the prefix based on status
            prefix = "[green]●[/green]" if node._value is not Nothing and isinstance(node._value.unwrap(), Success) else "[red]●[/red]"
            
            # Format the assertion text
            assertion_text = f"{node.actual} {node.validator.name}"
            
            # Add label if present
            if node.label:
                assertion_text = f"[dim]{node.label}:[/dim] {assertion_text}"
            
            # Add value if successful
            if node._value is not Nothing and isinstance(node._value.unwrap(), Success):
                val = node._value.unwrap().unwrap()
                if isinstance(val, (int, float)) or hasattr(val, 'is_integer'):
                    # Check if it's mathematically an integer
                    if (isinstance(val, int) or 
                        (hasattr(val, 'is_integer') and float(val).is_integer())):
                        formatted_val = str(int(float(val)))
                    else:
                        formatted_val = f"{float(val):.2f}" if abs(float(val)) < 1000 else f"{float(val):.1f}"
                    assertion_text = f"{assertion_text} ({formatted_val})"
            
            # Add error message if failed
            elif node._value is not Nothing and isinstance(node._value.unwrap(), Failure):
                error_msg = node._value.unwrap().failure()
                # Clean up error messages for display
                if "Parent check failed!" in error_msg:
                    assertion_text = f"{assertion_text}: [yellow]Skipped (parent failed)[/yellow]"
                elif "does not satisfy" in error_msg:
                    # Extract just the value that failed
                    try:
                        parts = error_msg.split(" = ")
                        if len(parts) > 1:
                            value_part = parts[1].split(" does not")[0]
                            assertion_text = f"{assertion_text}: Value {value_part} exceeds limit"
                        else:
                            # Malformed "does not satisfy" message - show full error
                            assertion_text = f"{assertion_text}: {error_msg}"
                    except (IndexError, AttributeError):
                        assertion_text = f"{assertion_text}: {error_msg}"
                else:
                    assertion_text = f"{assertion_text}: {error_msg}"
            
            if datasets:
                return f"{prefix} {assertion_text} [{datasets}]"
            return f"{prefix} {assertion_text}"
        return f"{node.actual}"


class SymbolNodeFormatter:
    """Formatter for SymbolNode."""
    
    def format(self, node: SymbolNode) -> str:
        """Format SymbolNode for display."""
        symbol_text = f"{str(node.symbol)}"
        name_text = node.name
        
        # Format the value if available
        value_str = ""
        if node._value is not Nothing and isinstance(node._value.unwrap(), Success):
            val = node._value.unwrap().unwrap()
            if isinstance(val, (int, float)):
                formatted_val = f"{val:.2f}" if isinstance(val, float) and abs(val) < 1000 else str(val)
                value_str = f" = {formatted_val}"
        
        status = format_status(node._value)
        datasets = format_datasets(node.datasets)
        
        if datasets:
            return f"📊 {symbol_text}: {name_text}{value_str} {status} [{datasets}]"
        return f"📊 {symbol_text}: {name_text}{value_str} {status}"


class MetricNodeFormatter:
    """Formatter for MetricNode."""
    
    def format(self, node: MetricNode) -> str:
        """Format MetricNode for display."""
        name = node.spec.name
        key = f"[{node.eval_key()}]"
        status = format_status(node._analyzed)
        datasets = format_datasets(node.datasets)
        
        if datasets:
            return f"📈 {name} {key} [{datasets}] {status}"
        return f"📈 {name} {key} {status}"


class AnalyzerNodeFormatter:
    """Formatter for AnalyzerNode."""
    
    def format(self, node: AnalyzerNode) -> str:
        """Format AnalyzerNode for display."""
        return f"🔧 {node.analyzer.name} analyzer"


# =============================================================================
# Tree Builder
# =============================================================================

class TreeBuilder:
    """Visitor for building Rich Tree representation."""
    
    def __init__(self, tree: Tree, display: GraphDisplay, root_node: BaseNode):
        self.tree = tree
        self.current_tree = tree
        self.display = display
        self._root_node = root_node
    
    def visit(self, node: BaseNode) -> Any:
        """Visit a node and build tree representation."""
        from dqx.graph import AssertionNode, CompositeNode
        
        # Skip assertion nodes without validators
        if isinstance(node, AssertionNode) and node.validator is None:
            return None
        
        if node is not self._root_node:  # Skip root node itself
            subtree = self.current_tree.add(self.display.format_node(node))
            parent_tree = self.current_tree
            self.current_tree = subtree
            
            # Continue traversal for composite nodes
            if isinstance(node, CompositeNode):
                for child in node.get_children():
                    # Type cast to satisfy mypy
                    from dqx.graph import NodeVisitor
                    child.accept(cast(NodeVisitor, self))
            
            self.current_tree = parent_tree
        
        return None


# =============================================================================
# Display Configuration and Manager
# =============================================================================

class DisplayConfig:
    """Configuration for display formatting."""
    
    def __init__(
        self,
        show_values: bool = True,
        show_datasets: bool = True,
        compact_errors: bool = True,
    ):
        self.show_values = show_values
        self.show_datasets = show_datasets
        self.compact_errors = compact_errors


class GraphDisplay:
    """Manages graph display with configurable formatting."""
    
    def __init__(self, config: DisplayConfig | None = None):
        self.config = config or DisplayConfig()
        self.formatters = self._init_formatters()
    
    def _init_formatters(self) -> dict[type, Any]:
        """Initialize formatters for each node type."""
        from dqx.graph import (
            AnalyzerNode,
            AssertionNode,
            CheckNode,
            MetricNode,
            RootNode,
            SymbolNode,
        )
        
        return {
            RootNode: RootNodeFormatter(),
            CheckNode: CheckNodeFormatter(),
            AssertionNode: AssertionNodeFormatter(),
            SymbolNode: SymbolNodeFormatter(),
            MetricNode: MetricNodeFormatter(),
            AnalyzerNode: AnalyzerNodeFormatter(),
        }
    
    def format_node(self, node: BaseNode) -> str:
        """Format node using appropriate formatter."""
        formatter = self.formatters.get(type(node))
        if formatter:
            return formatter.format(node)
        return str(node)
    
    def inspect_tree(self, root: RootNode) -> Tree:
        """Build tree representation of the graph."""
        tree = Tree(self.format_node(root))
        builder = TreeBuilder(tree, self, root)
        # Type cast to satisfy mypy
        from dqx.graph import NodeVisitor
        for child in root.children:
            child.accept(cast(NodeVisitor, builder))
        return tree


# =============================================================================
# Legacy function exports for backward compatibility
# =============================================================================

# These are exported at module level for backward compatibility
# with existing code that imports them directly
_format_status = format_status
_format_error = format_error
_format_datasets = format_datasets
