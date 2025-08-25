from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

import sympy as sp
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success
from rich.tree import Tree

from dqx.common import ResultKey, ResultKeyProvider, RetrievalFn, SeverityLevel, SymbolicValidator
from dqx.display import GraphDisplay
from dqx.ops import Op
from dqx.specs import MetricSpec

# Type definitions
MetricState = Literal["READY", "PROVIDED", "PENDING", "ERROR"]
TChild = TypeVar("TChild", bound="BaseNode")


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
    """Base class for nodes that can have children."""
    
    def __init__(self) -> None:
        self.children: list[TChild] = []
    
    def add_child(self, child: TChild) -> None:
        """Add a child node."""
        self.children.append(child)
    
    def remove_child(self, child: TChild) -> None:
        """Remove a child node."""
        self.children.remove(child)
    
    def get_children(self) -> list[TChild]:
        """Get all children."""
        return self.children
    
    def accept(self, visitor: NodeVisitor) -> Any:
        return visitor.visit(self)


# Visitor Pattern
class NodeVisitor(Protocol):
    """Protocol for visitor pattern."""
    
    def visit(self, node: BaseNode) -> Any:
        """Visit a node."""
        ...


class GraphTraverser:
    """Concrete visitor for graph traversal."""
    
    def __init__(self, filter_type: type[BaseNode] | None = None):
        self.filter_type = filter_type
        self.results: list[BaseNode] = []
    
    def visit(self, node: BaseNode) -> None:
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

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        # Imported here to avoid circular import issues
        from dqx.display import RootNodeFormatter
        return RootNodeFormatter().format(self)

    def inspect(self) -> Tree:
        """Create a tree representation of the graph structure."""
        display = GraphDisplay()
        return display.inspect_tree(self)
    
    def exists(self, child: "CheckNode") -> bool:
        """Check if a child node exists in the graph."""
        return child in self.children
    
    def traverse(self, filter_type: type[BaseNode] | None = None) -> Iterator[BaseNode]:
        """Generic traversal with optional type filtering."""
        traverser = GraphTraverser(filter_type)
        self.accept(traverser)
        return iter(traverser.results)
    
    def assertions(self) -> Iterator[AssertionNode]:
        """Iterate over all assertion nodes."""
        return self.traverse(AssertionNode)  # type: ignore
    
    def checks(self) -> Iterator[CheckNode]:
        """Iterate over all check nodes."""
        return self.traverse(CheckNode)  # type: ignore
    
    def metrics(self) -> Iterator[MetricNode]:
        """Iterate over all metric nodes."""
        return self.traverse(MetricNode)  # type: ignore
    
    def symbols(self) -> Iterator[SymbolNode]:
        """Iterate over all symbol nodes."""
        return self.traverse(SymbolNode)  # type: ignore
    
    def ready_metrics(self) -> Iterator[MetricNode]:
        """Iterate over metrics in READY state."""
        return (metric for metric in self.metrics() if metric.state() == "READY")
    
    def provided_metrics(self) -> Iterator[MetricNode]:
        """Iterate over metrics in PROVIDED state."""
        return (metric for metric in self.metrics() if metric.state() == "PROVIDED")
    
    def pending_metrics(self, dataset: str) -> Iterator[MetricNode]:
        """Iterate over metrics in PENDING state for a specific dataset."""
        for metric in self.metrics():
            if metric.datasets == [dataset] and metric.state() == "PENDING":
                yield metric
    
    def ready_symbols(self) -> Iterator[SymbolNode]:
        """Iterate over symbols that are ready."""
        return (symbol for symbol in self.symbols() if symbol.ready())
    
    def mark_pending_metrics_success(self, dataset: str) -> None:
        """Mark all pending metrics for a dataset as successful."""
        for metric in self.pending_metrics(dataset):
            metric.mark_as_success()
    
    def mark_pending_metric_failed(self, dataset: str, message: str) -> None:
        """Mark all pending metrics for a dataset as failed."""
        for metric in self.pending_metrics(dataset):
            metric.mark_as_failure(message)
    
    def propagate(self, datasets: list[str]) -> None:
        """
        Propagate dataset information through the graph.
        
        1. Impute check's datasets with the provided datasets
        2. Propagate datasets from checks to symbols
        3. Resolve metric constraints and mark metrics accordingly
        4. Back propagate failed metrics to symbols and assertions
        """
        for check in self.children:
            check.impute_dataset(datasets)
            check.propagate()


class CheckNode(CompositeNode["AssertionNode | SymbolNode"]):
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

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        from dqx.display import CheckNodeFormatter
        return CheckNodeFormatter().format(self)

    def node_name(self) -> str:
        """Get the display name of the node."""
        return self.label or self.name

    def impute_dataset(self, datasets: list[str]) -> None:
        """Validate and set datasets for this check."""
        if any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(f"The check {self.node_name()} requires datasets {self.datasets} but got {datasets}")
            )
        elif len(self.datasets) == 0:
            self.datasets = datasets

    def propagate(self) -> None:
        """Propagate dataset information to children."""
        for child in self.children:
            child.impute_dataset(self.datasets)
            if hasattr(child, 'propagate'):
                child.propagate()
    
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
            if hasattr(child, '_value'):
                if isinstance(child._value, Some):
                    result = child._value.unwrap()
                    if isinstance(result, Failure):
                        any_failure = True
                        all_success = False
                        if hasattr(child, 'label') and child.label:
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

    def impute_dataset(self, datasets: list[str]) -> None:
        """Validate and set datasets for this assertion."""
        if any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(
                    f"The assertion {str(self.actual) or self.label} requires datasets {self.datasets} but got {datasets}"
                )
            )
        elif len(self.datasets) == 0:
            self.datasets = datasets

    def propagate(self) -> None:
        """AssertionNode has no children, so nothing to propagate."""
        pass

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
        Evaluate the assertion expression.
        
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
        
        # Get all symbols from the graph
        symbol_nodes = list(self._root.symbols())
        
        # Build symbol table from available symbols
        symbol_table = {}
        failed_symbols = []
        found_symbols = set()
        
        for symbol_node in symbol_nodes:
            if symbol_node.symbol in self.actual.free_symbols:
                found_symbols.add(symbol_node.symbol)
                if symbol_node.success():
                    symbol_table[symbol_node.symbol] = symbol_node._value.unwrap().unwrap()
                elif symbol_node.failure():
                    failed_symbols.append(str(symbol_node.symbol))
        
        # Check for failed symbols
        if failed_symbols:
            self._value = Some(Failure(f"Symbol dependencies failed: {', '.join(failed_symbols)}"))
            return self._value.unwrap()
        
        # Check for missing symbols
        missing_symbols = self.actual.free_symbols - found_symbols
        if missing_symbols:
            self._value = Some(Failure(f"Missing symbols: {missing_symbols}"))
            return self._value.unwrap()
        
        # Evaluate the expression
        try:
            value = sp.N(self.actual.subs(symbol_table), 6)
            
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
                        failure_msg = f"Assertion failed: {self.actual} = {value} does not satisfy {self.validator.name}"
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
        raise RuntimeError("AssertionNode cannot have children. Symbols should be added to CheckNode instead.")


class SymbolNode(CompositeNode["MetricNode"]):
    """Node representing a symbol that can be evaluated."""
    
    def __init__(
        self, 
        name: str, 
        symbol: sp.Symbol, 
        fn: RetrievalFn, 
        datasets: list[str]
    ) -> None:
        super().__init__()
        self.name = name
        self.symbol = symbol
        self.fn = fn
        self.datasets = datasets
        self._value: Maybe[Result[float, str]] = Nothing
        self._required_ds_count = 1

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        from dqx.display import SymbolNodeFormatter
        return SymbolNodeFormatter().format(self)

    def mark_as_failure(self, message: str) -> None:
        """Mark this symbol as failed with a message."""
        self._value = Some(Failure(message))

    def ready(self) -> bool:
        """Check if all child metrics are in PROVIDED state."""
        return all(child.state() == "PROVIDED" for child in self.children)

    def success(self) -> bool:
        """Check if this symbol has been successfully evaluated."""
        if isinstance(self._value, Some):
            result = self._value.unwrap()
            return isinstance(result, Success)
        return False

    def failure(self) -> bool:
        """Check if this symbol has failed evaluation."""
        if isinstance(self._value, Some):
            result = self._value.unwrap()
            return isinstance(result, Failure)
        return False

    def impute_dataset(self, datasets: list[str]) -> None:
        """Validate and set datasets for this symbol."""
        if len(self.datasets) == self._required_ds_count and all(ds in datasets for ds in self.datasets):
            return
            
        if any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(f"The symbol {str(self.symbol)} requires datasets {self.datasets} but got {datasets}")
            )
        elif len(self.datasets) == 0 and len(datasets) != self._required_ds_count:
            self._value = Some(
                Failure(
                    f"The symbol {str(self.symbol)} requires exactly {self._required_ds_count} datasets but got {datasets}"
                )
            )
        else:
            self.datasets = datasets

    def propagate(self) -> None:
        """Propagate dataset information to child metrics."""
        for metric in self.children:
            metric.impute_dataset(self.datasets)

    def evaluate(self, key: ResultKey) -> Result[float, str]:
        """
        Evaluate this symbol using the retrieval function.
        
        Args:
            key: The result key for evaluation.
            
        Returns:
            Result containing the evaluated value or error message.
        """
        self._value = Some(self.fn(key))
        return self._value.unwrap()


class MetricNode(CompositeNode["AnalyzerNode"]):
    """
    Node representing a metric to be computed.
    
    Metric states:
    - PENDING: The metric is pending computation
    - PROVIDED: The metric is computed and stored
    - ERROR: The metric computation failed
    """

    def __init__(
        self, 
        spec: MetricSpec, 
        key_provider: ResultKeyProvider, 
        nominal_key: ResultKey
    ) -> None:
        super().__init__()
        self.spec = spec
        self.key_provider = key_provider
        self._nominal_key = nominal_key
        self.datasets: list[str] = []
        self._analyzed: Maybe[Result[None, str]] = Nothing

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        from dqx.display import MetricNodeFormatter
        return MetricNodeFormatter().format(self)

    def eval_key(self) -> ResultKey:
        """Get the evaluation key for this metric."""
        return self.key_provider.create(self._nominal_key)

    def mark_as_provided(self) -> None:
        """Mark this metric as provided."""
        self._analyzed = Some(Success(None))

    def mark_as_success(self) -> None:
        """Mark this metric as successfully computed."""
        self._analyzed = Some(Success(None))

    def mark_as_failure(self, message: str) -> None:
        """Mark this metric as failed with a message."""
        self._analyzed = Some(Failure(message))

    def impute_dataset(self, datasets: list[str]) -> None:
        """Set datasets for this metric."""
        self.datasets = datasets

    def state(self) -> MetricState:
        """
        Get the current state of this metric.
        
        Returns:
            The metric state: ERROR, PROVIDED, or PENDING.
        """
        if isinstance(self._analyzed, Some):
            result = self._analyzed.unwrap()
            if isinstance(result, Failure):
                return "ERROR"
            elif isinstance(result, Success):
                return "PROVIDED"
        return "PENDING"


class AnalyzerNode(LeafNode):
    """Node representing an analyzer operation."""
    
    def __init__(self, analyzer: Op[Any]) -> None:
        self.analyzer = analyzer

    def format_display(self) -> str:
        """Return a string representation for display formatting."""
        from dqx.display import AnalyzerNodeFormatter
        return AnalyzerNodeFormatter().format(self)
    
    def add_child(self, child: Any) -> None:
        """AnalyzerNode cannot have children."""
        raise NotImplementedError("AnalyzerNode cannot have children")
