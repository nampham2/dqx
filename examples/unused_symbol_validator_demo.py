"""Demonstration of UnusedSymbolValidator functionality."""

from dqx.common import SymbolicValidator
from dqx.graph.nodes import RootNode
from dqx.graph.traversal import Graph
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider
from dqx.validator import SuiteValidator


def main() -> None:
    """Demonstrate the UnusedSymbolValidator warning about unused symbols."""
    # Create a simple data quality suite
    root = RootNode("Product Quality Suite")

    # Create some checks
    revenue_check = root.add_check("Revenue Check")
    inventory_check = root.add_check("Inventory Check")

    # Create provider and define multiple metrics
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Define various metrics
    total_revenue = provider.sum("revenue")  # x_1 - Will be used
    avg_price = provider.average("price")  # x_2 - Will be used
    num_orders = provider.num_rows()  # x_3 - Will be used
    max_quantity = provider.maximum("quantity")  # x_4 - UNUSED!  # noqa: F841
    min_price = provider.minimum("price")  # x_5 - UNUSED!  # noqa: F841
    null_products = provider.null_count("product_id")  # x_6 - UNUSED!  # noqa: F841

    # Add assertions that use only some of the metrics
    revenue_check.add_assertion(
        total_revenue > 10000,
        name="Revenue above 10k",
        validator=SymbolicValidator("> 10000", lambda x: x > 10000),
    )

    revenue_check.add_assertion(
        avg_price > 50, name="Average price above $50", validator=SymbolicValidator("> 50", lambda x: x > 50)
    )

    inventory_check.add_assertion(
        num_orders < 1000000,
        name="Order count reasonable",
        validator=SymbolicValidator("< 1000000", lambda x: x < 1000000),
    )

    # Create graph and validate
    graph = Graph(root)
    validator = SuiteValidator()
    report = validator.validate(graph, provider)

    # Display validation report
    print("=== DQX Validation Report ===")
    print(report)

    # Show structured output
    print("\n=== Structured Report ===")
    structured = report.to_dict()
    print(f"Total Errors: {structured['summary']['error_count']}")
    print(f"Total Warnings: {structured['summary']['warning_count']}")

    if structured["warnings"]:
        print("\nWarnings:")
        for warning in structured["warnings"]:
            print(f"  - [{warning['rule']}] {warning['message']}")


if __name__ == "__main__":
    main()
