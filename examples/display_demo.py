"""Demo script for tree display functionality."""

import sympy as sp

from dqx.display import NodeFormatter
from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.graph.traversal import Graph


class ColorfulFormatter(NodeFormatter):
    """Custom formatter that adds emojis and colors."""

    def format_node(self, node: BaseNode) -> str:
        """Format node with emojis based on type."""
        if hasattr(node, "node_name") and callable(node.node_name):
            # CheckNode
            return f"✅ {node.node_name()}"
        elif hasattr(node, "name") and node.name:
            # RootNode or AssertionNode with name
            if node.__class__.__name__ == "RootNode":
                return f"🚀 {node.name}"
            else:
                return f"� {node.name}"
        else:
            # Fallback to class name
            return f"📌 {node.__class__.__name__}"


def create_sample_graph() -> Graph:
    """Create a sample data quality graph."""
    # Create root
    root = RootNode("E-commerce Data Quality Suite")

    # Create checks
    order_check = CheckNode("Order Data Validation")
    customer_check = CheckNode("Customer Data Validation")
    revenue_check = CheckNode("Revenue Monitoring")

    # Create order assertions
    order_count = sp.Symbol("order_count")
    order_assertion1 = AssertionNode(actual=order_count > 0, name="Orders exist")

    avg_order_value = sp.Symbol("avg_order_value")
    order_assertion2 = AssertionNode(actual=avg_order_value > 10, name="Average order value > $10")

    order_assertion3 = AssertionNode(actual=avg_order_value < 1000, name="Average order value < $1000")

    # Create customer assertions
    customer_count = sp.Symbol("customer_count")
    customer_assertion1 = AssertionNode(actual=customer_count > 0, name="Customers exist")

    email_nulls = sp.Symbol("email_nulls")
    customer_assertion2 = AssertionNode(actual=email_nulls == 0, name="No null emails")

    # Create revenue assertions
    daily_revenue = sp.Symbol("daily_revenue")
    revenue_assertion1 = AssertionNode(actual=daily_revenue > 10000, name="Daily revenue > $10k")

    dod_change = sp.Symbol("dod_change")
    revenue_assertion2 = AssertionNode(actual=dod_change < 0.5, name="Day-over-day change < 50%")

    # Build the graph
    root.add_child(order_check)
    root.add_child(customer_check)
    root.add_child(revenue_check)

    order_check.add_child(order_assertion1)
    order_check.add_child(order_assertion2)
    order_check.add_child(order_assertion3)

    customer_check.add_child(customer_assertion1)
    customer_check.add_child(customer_assertion2)

    revenue_check.add_child(revenue_assertion1)
    revenue_check.add_child(revenue_assertion2)

    return Graph(root)


def main() -> None:
    """Run the display demo."""
    print("DQX Tree Display Demo")
    print("=" * 50)
    print()

    # Create graph
    graph = create_sample_graph()

    # Display with default formatter
    print("1. Default formatter:")
    print("-" * 30)
    graph.print_tree()
    print()

    # Display with custom formatter
    print("2. Custom formatter with emojis:")
    print("-" * 30)
    custom_formatter = ColorfulFormatter()
    graph.print_tree(formatter=custom_formatter)
    print()

    # Show graph stats
    print("3. Graph Statistics:")
    print("-" * 30)
    checks = graph.checks()
    assertions = graph.assertions()
    print(f"Total checks: {len(checks)}")
    print(f"Total assertions: {len(assertions)}")
    print(f"Average assertions per check: {len(assertions) / len(checks):.1f}")


if __name__ == "__main__":
    main()
