from collections import deque
from typing import TYPE_CHECKING, Optional

from dqx.graph.base import BaseNode, CompositeNode, NodeVisitor
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.graph.visitors import NodeCollector

if TYPE_CHECKING:
    from dqx.display import NodeFormatter
    from dqx.provider import MetricProvider


class Graph:
    """Graph structure for data quality verification.

    The Graph class provides traversal and management functionality for a hierarchical
    structure of data quality checks and assertions. It implements various traversal
    algorithms (BFS and DFS) in both synchronous and asynchronous modes, and supports
    the visitor pattern for flexible node processing.

    The graph follows a three-level hierarchy:
        - RootNode: Top-level container for the entire verification suite
        - CheckNode: Groups related assertions under a common check
        - AssertionNode: Leaf nodes containing actual validation logic

    This class serves as the primary interface for:
        - Traversing the node hierarchy using different algorithms
        - Propagating dataset information through the graph
        - Collecting specific types of nodes for processing
        - Supporting both synchronous and asynchronous operations

    Attributes:
        root: The RootNode instance that serves as the entry point to the graph.
            All traversal operations start from this node.

    Examples:
        >>> from dqx.graph.nodes import RootNode, CheckNode, AssertionNode
        >>> from dqx.graph.visitors import NodePrinter
        >>>
        >>> # Create a graph structure
        >>> root = RootNode("quality_suite")
        >>> graph = Graph(root)
        >>>
        >>> # Add checks and assertions
        >>> check = root.add_check("completeness_check")
        >>>
        >>> # Traverse the graph
        >>> visitor = NodePrinter()
        >>> graph.dfs(visitor)
        >>>
        >>> # Collect all assertions
        >>> assertions = graph.assertions()
    """

    def __init__(self, root: RootNode) -> None:
        """Initialize a Graph with the given root node.

        Creates a new Graph instance that manages traversal and operations on
        a hierarchical structure of data quality nodes. The root node serves
        as the entry point for all graph operations.

        Args:
            root: The RootNode instance that serves as the top of the hierarchy.
                This node should already be configured with its child CheckNodes
                and their respective AssertionNodes.

        Examples:
            >>> from dqx.graph.nodes import RootNode
            >>> root = RootNode("my_quality_checks")
            >>> graph = Graph(root)
        """
        self.root = root

    def bfs(self, visitor: NodeVisitor) -> None:
        """Perform breadth-first traversal of the graph.

        Traverses the graph level by level, visiting all nodes at the current
        depth before moving to nodes at the next depth. This is useful when
        you want to process nodes in order of their distance from the root.

        The visitor pattern is used to separate the traversal logic from the
        processing logic, allowing different operations to be performed on the
        nodes without modifying the traversal code.

        Args:
            visitor: A NodeVisitor instance that implements the visit method.
                The visitor's visit method will be called once for each node
                in breadth-first order.

        Examples:
            >>> from dqx.graph.visitors import NodeCollector
            >>> visitor = NodeCollector(CheckNode)
            >>> graph.bfs(visitor)
            >>> # visitor.results now contains all CheckNodes in BFS order

        Note:
            The traversal starts from the root node and visits nodes level by level.
            For a tree with structure:
                Root
                ├── Check1
                │   ├── Assertion1
                │   └── Assertion2
                └── Check2
                    └── Assertion3

            The BFS order would be: Root, Check1, Check2, Assertion1, Assertion2, Assertion3
        """
        queue: deque[BaseNode] = deque([self.root])

        while queue:
            current = queue.popleft()
            current.accept(visitor)

            if not current.is_leaf():
                assert isinstance(current, CompositeNode)  # type hinting
                queue.extend(current.children)

    async def async_bfs(self, visitor: NodeVisitor) -> None:
        """Perform asynchronous breadth-first traversal of the graph.

        Similar to bfs() but supports asynchronous visitor operations. This is
        useful when node processing involves I/O operations or other async tasks
        that can benefit from concurrent execution.

        The traversal order is the same as synchronous BFS, but the visitor's
        visit_async method is awaited for each node, allowing for asynchronous
        processing.

        Args:
            visitor: A NodeVisitor instance that implements the visit_async method.
                The visitor's visit_async method will be awaited once for each
                node in breadth-first order.

        Examples:
            >>> import asyncio
            >>> from dqx.graph.visitors import AsyncNodeProcessor
            >>>
            >>> async def process_graph():
            ...     visitor = AsyncNodeProcessor()
            ...     await graph.async_bfs(visitor)
            ...     return visitor.results
            >>>
            >>> results = asyncio.run(process_graph())

        Note:
            While the traversal itself is sequential (maintaining BFS order),
            the visitor's async operations can perform concurrent I/O or other
            async tasks efficiently.
        """
        queue: list[BaseNode] = [self.root]

        while queue:
            current = queue.pop(0)
            await current.accept_async(visitor)

            if isinstance(current, CompositeNode):
                queue.extend(current.children)

    def dfs(self, visitor: NodeVisitor) -> None:
        """Perform depth-first traversal of the graph.

        Traverses the graph by exploring as far as possible along each branch
        before backtracking. This is useful when you want to process complete
        paths from root to leaf before moving to sibling branches.

        The visitor pattern is used to separate the traversal logic from the
        processing logic, allowing different operations to be performed on the
        nodes without modifying the traversal code.

        Args:
            visitor: A NodeVisitor instance that implements the visit method.
                The visitor's visit method will be called once for each node
                in depth-first order.

        Examples:
            >>> from dqx.graph.visitors import NodePrinter
            >>> visitor = NodePrinter()
            >>> graph.dfs(visitor)
            >>> # Nodes are printed in depth-first order

        Note:
            The traversal uses a stack-based iterative approach rather than
            recursion to avoid stack overflow with deep graphs. For a tree:
                Root
                ├── Check1
                │   ├── Assertion1
                │   └── Assertion2
                └── Check2
                    └── Assertion3

            The DFS order would be: Root, Check1, Assertion1, Assertion2, Check2, Assertion3
        """
        stack: list[BaseNode] = [self.root]

        while stack:
            current = stack.pop()
            current.accept(visitor)

            if isinstance(current, CompositeNode):
                stack.extend(reversed(current.children))

    async def async_dfs(self, visitor: NodeVisitor) -> None:
        """Perform asynchronous depth-first traversal of the graph.

        Similar to dfs() but supports asynchronous visitor operations. This is
        useful when node processing involves I/O operations or other async tasks
        that can benefit from concurrent execution.

        The traversal order is the same as synchronous DFS, but the visitor's
        visit_async method is awaited for each node, allowing for asynchronous
        processing.

        Args:
            visitor: A NodeVisitor instance that implements the visit_async method.
                The visitor's visit_async method will be awaited once for each
                node in depth-first order.

        Examples:
            >>> import asyncio
            >>> from dqx.graph.visitors import AsyncNodeValidator
            >>>
            >>> async def validate_graph():
            ...     visitor = AsyncNodeValidator()
            ...     await graph.async_dfs(visitor)
            ...     return visitor.validation_results
            >>>
            >>> results = asyncio.run(validate_graph())

        Note:
            The implementation uses an iterative approach with a stack to maintain
            the DFS order while supporting async operations.
        """
        stack: list[BaseNode] = [self.root]

        while stack:
            current = stack.pop()
            await current.accept_async(visitor)

            if isinstance(current, CompositeNode):
                stack.extend(reversed(current.children))

    def checks(self) -> list[CheckNode]:
        """Collect all CheckNode instances in the graph.

        Traverses the entire graph using depth-first search and collects all
        nodes that are instances of CheckNode. This is useful for operations
        that need to process all checks, such as generating reports or
        computing check-level metrics.

        Returns:
            A list of all CheckNode instances found in the graph, in the order
            they were encountered during depth-first traversal.

        Examples:
            >>> # Get all checks for reporting
            >>> all_checks = graph.checks()
            >>> for check in all_checks:
            ...     print(f"Check: {check.name}")
            >>>
            >>> # Count checks by dataset
            >>> from collections import Counter
            >>> dataset_counts = Counter(
            ...     dataset for check in all_checks for dataset in check.datasets
            ... )

        Note:
            The returned list contains only CheckNode instances, not the root
            node or assertion nodes. The order is consistent with DFS traversal.
        """
        visitor = NodeCollector(CheckNode)
        self.bfs(visitor)
        return visitor.results

    def assertions(self) -> list[AssertionNode]:
        """Collect all AssertionNode instances in the graph.

        Traverses the entire graph using depth-first search and collects all
        nodes that are instances of AssertionNode. This is useful for operations
        that need to process all assertions, such as evaluation or validation.

        Returns:
            A list of all AssertionNode instances found in the graph, in the order
            they were encountered during depth-first traversal.

        Examples:
            >>> # Get all assertions for evaluation
            >>> all_assertions = graph.assertions()
            >>> for assertion in all_assertions:
            ...     print(f"Assertion: {assertion.name}, Severity: {assertion.severity}")
            >>>
            >>> # Filter high-severity assertions
            >>> critical_assertions = [
            ...     a for a in all_assertions
            ...     if a.severity == SeverityLevel.CRITICAL
            ... ]

        Note:
            The returned list contains only AssertionNode instances (leaf nodes).
            The order is consistent with DFS traversal, so assertions from the
            same check will be grouped together.
        """
        visitor = NodeCollector(AssertionNode)
        self.bfs(visitor)
        return visitor.results

    def print_tree(self, formatter: Optional["NodeFormatter"] = None) -> None:
        """Print the graph structure as a tree to the console.

        This is a convenience method that uses the print_graph function from
        the display module to visualize the graph structure using Rich's tree
        rendering capabilities.

        Args:
            formatter: Optional formatter for node labels. If not provided,
                uses the default SimpleNodeFormatter which displays nodes
                with priority: label -> name -> class name.

        Examples:
            >>> # Print with default formatter
            >>> graph.print_tree()
            >>>
            >>> # Print with custom formatter
            >>> from dqx.display import NodeFormatter
            >>> class CustomFormatter:
            ...     def format_node(self, node):
            ...         return f"Custom: {node.__class__.__name__}"
            >>> graph.print_tree(formatter=CustomFormatter())

        Note:
            This method requires the Rich library to be installed for
            terminal output formatting.
        """
        from dqx.display import print_graph

        # Note: print_graph only accepts the graph argument, formatter is not supported
        if formatter is not None:
            import warnings

            warnings.warn("formatter argument is not supported by print_graph and will be ignored")

        print_graph(self)

    def impute_datasets(self, datasets: list[str], provider: "MetricProvider") -> None:
        """Propagate dataset information through the graph using visitor pattern.

        Args:
            datasets: List of available dataset names
            provider: MetricProvider to access SymbolicMetrics

        Raises:
            DQXError: If validation fails
        """
        from dqx.common import DQXError
        from dqx.graph.visitors import DatasetImputationVisitor

        visitor = DatasetImputationVisitor(datasets, provider)
        self.dfs(visitor)  # Use DFS to ensure parents are processed before children

        if visitor.has_errors():
            raise DQXError(visitor.get_error_summary())
