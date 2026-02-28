"""Tests for strongly typed parent hierarchy."""

from __future__ import annotations

import pytest
import sympy as sp

from dqx.common import DQXError, SymbolicValidator
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


class TestRootNodeNameValidation:
    """Test suite for RootNode name validation."""

    def test_root_node_with_valid_name(self) -> None:
        """Test RootNode creation with valid name."""
        root = RootNode("valid_suite_name")
        assert root.name == "valid_suite_name"

    def test_root_node_strips_whitespace(self) -> None:
        """Test RootNode strips leading/trailing whitespace from name."""
        root = RootNode("  suite_with_spaces  ")
        assert root.name == "suite_with_spaces"

    def test_root_node_with_empty_string(self) -> None:
        """Test RootNode raises DQXError for empty string name."""
        with pytest.raises(DQXError, match="Root name cannot be empty"):
            RootNode("")

    def test_root_node_with_whitespace_only(self) -> None:
        """Test RootNode raises DQXError for whitespace-only name."""
        with pytest.raises(DQXError, match="Root name cannot be empty"):
            RootNode("   ")

    def test_root_node_with_tabs_only(self) -> None:
        """Test RootNode raises DQXError for tab-only name."""
        with pytest.raises(DQXError, match="Root name cannot be empty"):
            RootNode("\t\t")

    def test_root_node_with_newlines_only(self) -> None:
        """Test RootNode raises DQXError for newline-only name."""
        with pytest.raises(DQXError, match="Root name cannot be empty"):
            RootNode("\n\n")

    def test_root_node_with_mixed_whitespace_only(self) -> None:
        """Test RootNode raises DQXError for mixed whitespace-only name."""
        with pytest.raises(DQXError, match="Root name cannot be empty"):
            RootNode(" \t\n ")

    def test_root_node_with_max_length_name(self) -> None:
        """Test RootNode accepts name at max length boundary (255 chars)."""
        max_length_name = "a" * 255
        root = RootNode(max_length_name)
        assert root.name == max_length_name
        assert len(root.name) == 255

    def test_root_node_with_name_exceeding_max_length(self) -> None:
        """Test RootNode raises DQXError for name exceeding 255 chars."""
        too_long_name = "a" * 256
        with pytest.raises(DQXError, match="Root name is too long \\(max 255 characters\\)"):
            RootNode(too_long_name)

    def test_root_node_with_name_far_exceeding_max_length(self) -> None:
        """Test RootNode raises DQXError for name far exceeding max length."""
        way_too_long_name = "a" * 1000
        with pytest.raises(DQXError, match="Root name is too long \\(max 255 characters\\)"):
            RootNode(way_too_long_name)

    def test_root_node_with_whitespace_that_exceeds_when_stripped(self) -> None:
        """Test RootNode handles whitespace correctly before length check."""
        # 253 'a's + spaces = 259 total, but 253 after strip (valid)
        name_with_spaces = "  " + ("a" * 253) + "  "
        root = RootNode(name_with_spaces)
        assert len(root.name) == 253


class TestCheckNodeNameValidation:
    """Test suite for CheckNode name validation."""

    @pytest.fixture
    def root(self) -> RootNode:
        """Fixture providing a RootNode for CheckNode tests."""
        return RootNode("test_suite")

    def test_check_node_with_valid_name(self, root: RootNode) -> None:
        """Test CheckNode creation with valid name."""
        check = CheckNode(parent=root, name="valid_check_name")
        assert check.name == "valid_check_name"

    def test_check_node_strips_whitespace(self, root: RootNode) -> None:
        """Test CheckNode strips leading/trailing whitespace from name."""
        check = CheckNode(parent=root, name="  check_with_spaces  ")
        assert check.name == "check_with_spaces"

    def test_check_node_with_empty_string(self, root: RootNode) -> None:
        """Test CheckNode raises DQXError for empty string name."""
        with pytest.raises(DQXError, match="Check name cannot be empty"):
            CheckNode(parent=root, name="")

    def test_check_node_with_whitespace_only(self, root: RootNode) -> None:
        """Test CheckNode raises DQXError for whitespace-only name."""
        with pytest.raises(DQXError, match="Check name cannot be empty"):
            CheckNode(parent=root, name="   ")

    def test_check_node_with_tabs_only(self, root: RootNode) -> None:
        """Test CheckNode raises DQXError for tab-only name."""
        with pytest.raises(DQXError, match="Check name cannot be empty"):
            CheckNode(parent=root, name="\t\t")

    def test_check_node_with_newlines_only(self, root: RootNode) -> None:
        """Test CheckNode raises DQXError for newline-only name."""
        with pytest.raises(DQXError, match="Check name cannot be empty"):
            CheckNode(parent=root, name="\n\n")

    def test_check_node_with_mixed_whitespace_only(self, root: RootNode) -> None:
        """Test CheckNode raises DQXError for mixed whitespace-only name."""
        with pytest.raises(DQXError, match="Check name cannot be empty"):
            CheckNode(parent=root, name=" \t\n ")

    def test_check_node_with_max_length_name(self, root: RootNode) -> None:
        """Test CheckNode accepts name at max length boundary (255 chars)."""
        max_length_name = "b" * 255
        check = CheckNode(parent=root, name=max_length_name)
        assert check.name == max_length_name
        assert len(check.name) == 255

    def test_check_node_with_name_exceeding_max_length(self, root: RootNode) -> None:
        """Test CheckNode raises DQXError for name exceeding 255 chars."""
        too_long_name = "b" * 256
        with pytest.raises(DQXError, match="Check name is too long \\(max 255 characters\\)"):
            CheckNode(parent=root, name=too_long_name)

    def test_check_node_with_name_far_exceeding_max_length(self, root: RootNode) -> None:
        """Test CheckNode raises DQXError for name far exceeding max length."""
        way_too_long_name = "b" * 1000
        with pytest.raises(DQXError, match="Check name is too long \\(max 255 characters\\)"):
            CheckNode(parent=root, name=way_too_long_name)

    def test_check_node_with_whitespace_that_exceeds_when_stripped(self, root: RootNode) -> None:
        """Test CheckNode handles whitespace correctly before length check."""
        # 253 'b's + spaces = 259 total, but 253 after strip (valid)
        name_with_spaces = "  " + ("b" * 253) + "  "
        check = CheckNode(parent=root, name=name_with_spaces)
        assert len(check.name) == 253

    def test_check_node_factory_method_with_invalid_name(self, root: RootNode) -> None:
        """Test that factory method add_check also validates names."""
        with pytest.raises(DQXError, match="Check name cannot be empty"):
            root.add_check("")

    def test_check_node_factory_method_with_long_name(self, root: RootNode) -> None:
        """Test that factory method add_check validates name length."""
        too_long_name = "c" * 256
        with pytest.raises(DQXError, match="Check name is too long \\(max 255 characters\\)"):
            root.add_check(too_long_name)
