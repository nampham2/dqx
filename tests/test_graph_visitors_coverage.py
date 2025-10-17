"""Additional tests to improve coverage for graph/visitors.py."""

import pytest
import sympy as sp

from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.graph.visitors import DatasetImputationVisitor, NodeCollector


def test_dataset_imputation_visitor_assertion_node_no_provider() -> None:
    """Test visiting assertion node without provider."""
    from dqx.common import SymbolicValidator

    # Create visitor without provider
    visitor = DatasetImputationVisitor(["dataset1"], None)

    # Create assertion node with a proper validator
    root = RootNode("test_suite")
    check = root.add_check("test_check")
    validator = SymbolicValidator("x > 0", lambda x: x > 0)
    assertion = AssertionNode(
        parent=check, actual=sp.Symbol("x"), name="test_assertion", severity="P1", validator=validator
    )

    # This should return early without errors
    visitor.visit(assertion)

    # No errors should be collected
    assert not visitor.has_errors()
    assert visitor.get_errors() == []


@pytest.mark.asyncio
async def test_dataset_imputation_visitor_async() -> None:
    """Test DatasetImputationVisitor async visit method."""
    visitor = DatasetImputationVisitor(["dataset1", "dataset2"], None)

    root = RootNode("test_suite")
    check = root.add_check("test_check")

    # Test async visit on root node
    await visitor.visit_async(root)
    assert root.datasets == ["dataset1", "dataset2"]

    # Test async visit on check node
    await visitor.visit_async(check)
    assert check.datasets == ["dataset1", "dataset2"]


@pytest.mark.asyncio
async def test_node_collector_async() -> None:
    """Test NodeCollector async visit method."""
    collector = NodeCollector(CheckNode)

    root = RootNode("test_suite")
    check1 = root.add_check("check1")
    check2 = root.add_check("check2")

    # Use async visit method
    await collector.visit_async(root)  # Should not collect RootNode
    await collector.visit_async(check1)  # Should collect
    await collector.visit_async(check2)  # Should collect

    assert len(collector.results) == 2
    assert check1 in collector.results
    assert check2 in collector.results
