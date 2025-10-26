"""Symbol deduplication visitor for DQX graph traversal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode

if TYPE_CHECKING:
    pass


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

    def visit(self, node: BaseNode) -> Any:
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
