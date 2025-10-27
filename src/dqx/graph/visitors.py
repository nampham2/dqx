from typing import Generic, TypeVar

import sympy as sp

from dqx.common import DQXError
from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.provider import MetricProvider

TNode = TypeVar("TNode", bound=BaseNode)


class DatasetImputationVisitor:
    """Visitor that validates and imputes datasets for graph nodes.

    This visitor performs dataset validation and imputation on a graph:
    1. CheckNode: validates datasets against available datasets,
       imputes from available if not specified
    2. AssertionNode: imputes datasets for contained SymbolicMetrics

    Attributes:
        available_datasets: List of available dataset names
        provider: MetricProvider to get SymbolicMetrics
        _errors: List of collected validation errors
    """

    def __init__(self, available_datasets: list[str], provider: MetricProvider | None) -> None:
        """Initialize the DatasetImputationVisitor.

        Args:
            available_datasets: List of available dataset names
            provider: MetricProvider to get SymbolicMetrics, can be None for testing

        Raises:
            DQXError: If available_datasets is empty
        """
        if not available_datasets:
            raise DQXError("At least one dataset must be provided")

        self.available_datasets = available_datasets
        self.provider = provider
        self._errors: list[str] = []

    def visit(self, node: BaseNode) -> None:
        """Visit a node and perform dataset validation/imputation.

        Args:
            node: The node to visit
        """
        if isinstance(node, RootNode):
            self._visit_root_node(node)
        elif isinstance(node, CheckNode):
            self._visit_check_node(node)
        elif isinstance(node, AssertionNode):
            self._visit_assertion_node(node)

    def _visit_root_node(self, node: RootNode) -> None:
        """Set available datasets on the RootNode.

        This establishes the top-level datasets that will flow down
        through the hierarchy.

        Args:
            node: The RootNode to process
        """
        node.datasets = self.available_datasets.copy()

    def _visit_check_node(self, node: CheckNode) -> None:
        """Validate and impute datasets for a CheckNode.

        If the CheckNode has no datasets, impute from parent's datasets.
        If it has datasets, validate they are all in parent's datasets.

        Args:
            node: The CheckNode to process
        """
        # Get parent's datasets
        parent_datasets = node.parent.datasets

        if not node.datasets:
            # Impute from parent datasets
            node.datasets = parent_datasets.copy()
        else:
            # Validate existing datasets against parent
            for dataset in node.datasets:
                if dataset not in parent_datasets:
                    self._errors.append(
                        f"Check '{node.name}' specifies dataset '{dataset}' "
                        f"which is not in parent datasets: {parent_datasets}"
                    )

    def _visit_assertion_node(self, node: AssertionNode) -> None:
        """Process SymbolicMetrics in an AssertionNode and their dependencies.

        For each symbol in the assertion expression and all their transitive
        dependencies:
        1. Get its SymbolicMetric from the provider
        2. Validate dataset consistency
        3. Impute dataset if needed
        4. Propagate datasets to children and validate consistency

        Args:
            node: The AssertionNode to process
        """
        if not self.provider:
            return

        # Extract symbols from the assertion's actual expression
        symbols = node.actual.free_symbols

        # Process all symbols and their transitive dependencies
        processed_symbols = set()
        symbols_to_process = list(symbols)

        while symbols_to_process:
            symbol = symbols_to_process.pop()  # O(1) operation - removes from end

            # Skip if already processed
            if symbol in processed_symbols:
                continue

            processed_symbols.add(symbol)

            # Process this symbol
            metric = self.provider.get_symbol(symbol)

            # Get parent check's datasets
            parent_datasets = node.parent.datasets

            # Validate or impute dataset
            if metric.dataset:
                # Validate existing dataset
                if metric.dataset not in parent_datasets:
                    self._errors.append(
                        f"Symbol '{metric.name}' requires dataset '{metric.dataset}' "
                        f"but parent check only has datasets: {parent_datasets}"
                    )
            else:
                # Impute dataset
                if len(parent_datasets) == 1:
                    metric.dataset = parent_datasets[0]
                else:
                    self._errors.append(
                        f"Cannot impute dataset for symbol '{metric.name}': "
                        f"parent check has multiple datasets: {parent_datasets}"
                    )

            # Get children of this symbol
            children = self.provider.get_symbol(symbol).required_metrics

            # Process each child
            for child_symbol in children:
                child_metric = self.provider.get_symbol(child_symbol)

                if metric.dataset:  # Parent has a dataset
                    if child_metric.dataset and child_metric.dataset != metric.dataset:
                        self._errors.append(
                            f"Child symbol '{child_metric.name}' has dataset '{child_metric.dataset}' "
                            f"but its parent symbol '{metric.name}' has dataset '{metric.dataset}'. "
                            f"Dependent metrics must use the same dataset as their parent."
                        )
                    elif not child_metric.dataset:
                        # Propagate dataset from parent to child
                        child_metric.dataset = metric.dataset

                # Add child to processing queue if not already processed
                if child_symbol not in processed_symbols:
                    symbols_to_process.append(child_symbol)

    def get_errors(self) -> list[str]:
        """Get the list of collected errors.

        Returns:
            List of error messages
        """
        return self._errors.copy()

    def has_errors(self) -> bool:
        """Check if any errors were collected.

        Returns:
            True if there are errors, False otherwise
        """
        return len(self._errors) > 0

    def get_error_summary(self) -> str:
        """Get a formatted summary of all errors.

        Returns:
            Formatted error summary or empty string if no errors
        """
        if not self._errors:
            return ""

        return f"Dataset validation failed with {len(self._errors)} error(s):\n" + "\n".join(
            f"  - {error}" for error in self._errors
        )

    async def visit_async(self, node: BaseNode) -> None:
        """Asynchronously visit a node.

        Currently just delegates to synchronous visit.

        Args:
            node: The node to visit
        """
        self.visit(node)


class SymbolDeduplicationVisitor:
    """Visitor that replaces duplicate symbols in assertion expressions.

    This visitor traverses the graph and updates AssertionNode expressions
    by substituting duplicate symbols with their canonical representatives.

    Example:
        If x_3 is a duplicate of x_1, this visitor will replace all
        occurrences of x_3 in assertion expressions with x_1.
    """

    def __init__(self, substitutions: dict[sp.Symbol, sp.Symbol]) -> None:
        """Initialize with substitution map.

        Args:
            substitutions: Map from duplicate symbols to canonical symbols.
                          For example: {x_3: x_1, x_5: x_2}
        """
        self._substitutions = substitutions

    def visit(self, node: BaseNode) -> None:
        """Visit a node and apply symbol deduplication if it's an AssertionNode.

        Args:
            node: The node to visit
        """
        if isinstance(node, AssertionNode):
            # Apply substitutions to the actual expression
            node.actual = node.actual.subs(self._substitutions)

    async def visit_async(self, node: BaseNode) -> None:
        """Async visit method required by visitor protocol.

        Since deduplication is synchronous, this just delegates to visit.
        """
        self.visit(node)


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
