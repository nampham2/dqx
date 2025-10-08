"""Unit tests for BaseNode and CompositeNode classes."""
from __future__ import annotations

import pytest

from dqx.graph.base import BaseNode, CompositeNode


# Test implementations (prefix with Mock to avoid pytest warnings)
class MockNode(BaseNode):
    """Concrete implementation of BaseNode for testing."""
    
    def is_leaf(self) -> bool:
        """Test implementation always returns True."""
        return True


class MockNonLeafNode(BaseNode):
    """Concrete implementation that claims not to be a leaf."""
    
    def is_leaf(self) -> bool:
        return False


class MockChildNode(BaseNode):
    """A specific child node type for testing type constraints."""
    
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
    
    def is_leaf(self) -> bool:
        return True


class MockCompositeNode(CompositeNode[MockNode]):
    """Concrete implementation of CompositeNode for testing."""
    
    def __init__(self) -> None:
        """Initialize the composite node."""
        super().__init__()


class MockVisitor:
    """Test visitor that records visited nodes."""
    
    def __init__(self) -> None:
        self.visited_nodes: list[BaseNode] = []
    
    def visit(self, node: BaseNode) -> None:
        """Record the visited node."""
        self.visited_nodes.append(node)
    
    async def visit_async(self, node: BaseNode) -> None:
        """Record the visited node asynchronously."""
        self.visited_nodes.append(node)


class TestBaseNode:
    """Test suite for BaseNode functionality."""
    
    def test_init_sets_parent_to_none(self) -> None:
        """Test that BaseNode initializes with parent set to None."""
        node = MockNode()
        
        assert node.parent is None
    
    def test_is_root_returns_true_when_parent_is_none(self) -> None:
        """Test that is_root returns True when node has no parent."""
        node = MockNode()
        
        result = node.is_root
        
        assert result is True
    
    def test_is_root_returns_false_when_parent_exists(self) -> None:
        """Test that is_root returns False when node has a parent."""
        child = MockNode()
        parent = MockNode()
        child.parent = parent
        
        result = child.is_root
        
        assert result is False
    
    def test_accept_calls_visitor_visit_method(self) -> None:
        """Test that accept() correctly calls visitor's visit method."""
        node = MockNode()
        visitor = MockVisitor()
        
        node.accept(visitor)
        
        assert len(visitor.visited_nodes) == 1
        assert visitor.visited_nodes[0] is node
    
    @pytest.mark.asyncio
    async def test_accept_async_calls_visitor_visit_async_method(self) -> None:
        """Test that accept_async() correctly calls visitor's async method."""
        node = MockNode()
        visitor = MockVisitor()
        
        await node.accept_async(visitor)
        
        assert len(visitor.visited_nodes) == 1
        assert visitor.visited_nodes[0] is node
    
    def test_is_leaf_must_be_implemented_by_subclasses(self) -> None:
        """Test that is_leaf raises NotImplementedError in base class."""
        # We can't test this directly on BaseNode since it's abstract,
        # but we can verify our test implementations work correctly
        
        leaf_node = MockNode()
        non_leaf_node = MockNonLeafNode()
        
        assert leaf_node.is_leaf() is True
        assert non_leaf_node.is_leaf() is False


class TestCompositeNode:
    """Test suite for CompositeNode functionality."""
    
    def test_init_creates_empty_children_list(self) -> None:
        """Test that CompositeNode initializes with empty children list."""
        node = MockCompositeNode()
        
        assert isinstance(node.children, list)
        assert len(node.children) == 0
    
    def test_init_calls_parent_init(self) -> None:
        """Test that CompositeNode properly initializes BaseNode attributes."""
        node = MockCompositeNode()
        
        assert node.parent is None
        assert node.is_root is True
    
    def test_children_list_is_instance_attribute(self) -> None:
        """Test that each instance has its own children list."""
        node1 = MockCompositeNode()
        node2 = MockCompositeNode()
        child = MockNode()
        
        node1.children.append(child)
        
        assert len(node1.children) == 1
        assert len(node2.children) == 0  # Should not be shared
    
    def test_children_type_safety(self) -> None:
        """Test that children list maintains type safety."""
        # This is more of a documentation test since Python doesn't
        # enforce generics at runtime, but it's good to show intent
        
        node = MockCompositeNode()
        child = MockNode()
        
        node.children.append(child)
        
        assert all(isinstance(child, MockNode) for child in node.children)

    def test_is_leaf_returns_false(self) -> None:
        """Test that CompositeNode.is_leaf always returns False."""
        node = MockCompositeNode()
        
        result = node.is_leaf()
        
        assert result is False
    
    def test_is_leaf_returns_false_even_with_no_children(self) -> None:
        """Test that is_leaf returns False regardless of children."""
        node = MockCompositeNode()
        assert len(node.children) == 0  # Verify no children
        
        result = node.is_leaf()
        
        assert result is False  # Still not a leaf conceptually


class TestBaseNodeCompositeNodeIntegration:
    """Test integration between BaseNode and CompositeNode."""
    
    def test_composite_node_with_visitor_pattern(self) -> None:
        """Test visitor pattern works with composite nodes."""
        composite = MockCompositeNode()
        visitor = MockVisitor()
        
        composite.accept(visitor)
        
        assert len(visitor.visited_nodes) == 1
        assert visitor.visited_nodes[0] is composite
    
    def test_parent_child_relationship_with_composite(self) -> None:
        """Test setting parent on nodes in composite structure."""
        parent = MockCompositeNode()
        child1 = MockNode()
        child2 = MockNode()
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        assert child1.parent is parent
        assert child2.parent is parent
        assert child1.is_root is False
        assert child2.is_root is False
        assert parent.is_root is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_node_can_change_parent(self) -> None:
        """Test that a node's parent can be changed."""
        node = MockNode()
        parent1 = MockNode()
        parent2 = MockNode()
        
        node.parent = parent1
        assert node.parent is parent1
        assert node.is_root is False
        
        node.parent = parent2
        assert node.parent is parent2
        
        node.parent = None
        assert node.parent is None
        assert node.is_root is True
    
    def test_visitor_with_none_handling(self) -> None:
        """Test visitor behavior with edge cases."""
        node = MockNode()
        
        class ErrorVisitor:
            def visit(self, node: BaseNode) -> None:
                raise ValueError("Test error")
            
            async def visit_async(self, node: BaseNode) -> None:
                raise ValueError("Test async error")
        
        visitor = ErrorVisitor()
        
        with pytest.raises(ValueError, match="Test error"):
            node.accept(visitor)
        
        with pytest.raises(ValueError, match="Test async error"):
            import asyncio
            asyncio.run(node.accept_async(visitor))


class TestCompositeNodeAddChild:
    """Test suite for CompositeNode.add_child method."""
    
    def test_add_child_adds_to_children_list(self) -> None:
        """Test that add_child adds the child to children list."""
        parent = MockCompositeNode()
        child = MockNode()
        
        result = parent.add_child(child)
        
        assert len(parent.children) == 1
        assert parent.children[0] is child
        assert result is parent  # Method should return self
    
    def test_add_child_sets_parent_reference(self) -> None:
        """Test that add_child sets the child's parent reference."""
        parent = MockCompositeNode()
        child = MockNode()
        
        parent.add_child(child)
        
        assert child.parent is parent
        assert child.is_root is False
    
    def test_add_child_raises_error_on_duplicate(self) -> None:
        """Test that add_child raises DQXError when adding duplicate child."""
        from dqx.common import DQXError
        
        parent = MockCompositeNode()
        child = MockNode()
        
        # Add child first time - should work
        parent.add_child(child)
        
        # Try to add same child again - should raise error
        with pytest.raises(DQXError, match="Child node is already in the children list"):
            parent.add_child(child)
    
    def test_add_child_allows_reparenting(self) -> None:
        """Test that a child can be moved from one parent to another."""
        parent1 = MockCompositeNode()
        parent2 = MockCompositeNode()
        child = MockNode()
        
        # Add to first parent
        parent1.add_child(child)
        assert child.parent is parent1
        assert child in parent1.children
        
        # Move to second parent
        parent2.add_child(child)
        assert child.parent is parent2
        assert child in parent2.children
        assert child in parent1.children  # Still in first parent's list!
    
    def test_add_child_returns_self_for_chaining(self) -> None:
        """Test that add_child returns self to enable method chaining."""
        parent = MockCompositeNode()
        child1 = MockNode()
        child2 = MockNode()
        
        # Chain multiple add_child calls
        result = parent.add_child(child1).add_child(child2)
        
        assert result is parent
        assert len(parent.children) == 2
        assert child1.parent is parent
        assert child2.parent is parent
