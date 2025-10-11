from typing import Generic, TypeVar

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
        """Process SymbolicMetrics in an AssertionNode.

        For each symbol in the assertion expression:
        1. Get its SymbolicMetric from the provider
        2. Validate dataset consistency
        3. Impute dataset if needed

        Args:
            node: The AssertionNode to process
        """
        if not self.provider:
            return

        # Extract symbols from the assertion's actual expression
        symbols = node.actual.free_symbols

        for symbol in symbols:
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
