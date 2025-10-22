"""Tests for strongly typed parent hierarchy."""

import pytest
import sympy as sp

from dqx.common import SymbolicValidator
from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode


def test_root_node_has_none_parent() -> None:
    """Root nodes should always have None as parent."""
    root = RootNode("test_suite")
    assert root.parent is None
    assert root.is_root is True


def test_check_node_requires_root_parent() -> None:
    """CheckNode should require a RootNode parent at construction."""
    root = RootNode("test_suite")

    # This should work
    check = CheckNode(parent=root, name="my_check")
    assert check.parent is root
    assert isinstance(check.parent, RootNode)
    assert check.is_root is False

    # This should not work (type checker should complain)
    # check = CheckNode(parent=None, name="orphan")  # Should fail type checking


def test_assertion_node_requires_check_parent() -> None:
    """AssertionNode should require a CheckNode parent at construction."""
    root = RootNode("test_suite")
    check = CheckNode(parent=root, name="my_check")

    # This should work
    test_validator = SymbolicValidator("valid", lambda x: x > 0)
    assertion = AssertionNode(parent=check, actual=sp.Symbol("x"), name="test_assertion", validator=test_validator)
    assert assertion.parent is check
    assert isinstance(assertion.parent, CheckNode)

    # This should not work (type checker should complain)
    # assertion = AssertionNode(parent=None, actual=sp.Symbol("x"))  # Should fail


def test_factory_methods_create_proper_hierarchy() -> None:
    """Test that factory methods maintain proper parent-child relationships."""
    root = RootNode("test_suite")

    # Use factory method to create check
    check = root.add_check("my_check")
    assert check.parent is root
    assert check in root.children

    # Use factory method to create assertion
    positive_validator = SymbolicValidator("> 0", lambda x: x > 0)
    assertion = check.add_assertion(actual=sp.Symbol("x"), name="x > 0", validator=positive_validator)
    assert assertion.parent is check
    assert assertion in check.children


def test_type_safety_at_runtime() -> None:
    """Verify runtime type safety for parent assignments."""
    root = RootNode("test_suite")

    # Test AssertionNode with wrong parent type
    with pytest.raises(TypeError) as exc_info:
        bad_validator = SymbolicValidator("bad", lambda x: False)
        AssertionNode(parent=root, actual=sp.Symbol("x"), name="bad_assertion", validator=bad_validator)  # type: ignore
    assert "AssertionNode requires parent of type CheckNode" in str(exc_info.value)
    assert "but got RootNode" in str(exc_info.value)

    # Test CheckNode with None parent by bypassing constructor
    with pytest.raises(TypeError) as exc_info:
        node = object.__new__(CheckNode)
        node.name = "orphan_check"
        BaseNode.__init__(node, parent=None)  # type: ignore
    assert "CheckNode requires parent of type RootNode" in str(exc_info.value)
    assert "but got None" in str(exc_info.value)


def test_parent_is_immutable() -> None:
    """Test that parent cannot be changed after construction."""
    root = RootNode("test_suite")
    check = CheckNode(parent=root, name="my_check")

    # Parent property should exist
    assert hasattr(check, "parent")

    # Get the property descriptor
    parent_prop = type(check).__dict__.get("parent") or BaseNode.__dict__.get("parent")

    # Should be a property with no setter
    assert isinstance(parent_prop, property)
    assert parent_prop.fget is not None
    assert parent_prop.fset is None

    # Trying to set parent should fail
    with pytest.raises(AttributeError):
        check.parent = RootNode("another_root")  # type: ignore


def test_improved_error_messages() -> None:
    """Test that runtime validation provides clear error messages."""
    # Test RootNode with non-None parent
    with pytest.raises(TypeError) as exc_info:
        # We need to bypass the constructor type checking
        node = object.__new__(RootNode)
        node.name = "invalid_root"
        BaseNode.__init__(node, parent=RootNode("parent"))  # type: ignore
    assert "RootNode must have None as parent" in str(exc_info.value)
    assert "but got RootNode" in str(exc_info.value)
