"""Tests for the freeze_for_hashing utility function."""

from dqx.utils import freeze_for_hashing


class TestFreezeForHashing:
    """Test freeze_for_hashing utility function."""

    def test_freeze_simple_types(self) -> None:
        """Test freezing simple immutable types."""
        # These should be returned as-is
        assert freeze_for_hashing(42) == 42
        assert freeze_for_hashing("hello") == "hello"
        assert freeze_for_hashing(3.14) == 3.14
        assert freeze_for_hashing(True) is True
        assert freeze_for_hashing(None) is None

    def test_freeze_tuple(self) -> None:
        """Test that tuples are converted to new tuples."""
        tup = (1, 2, 3)
        result = freeze_for_hashing(tup)
        assert result == tup
        # Note: The function creates a new tuple, so it won't be the same object

    def test_freeze_list(self) -> None:
        """Test freezing lists to tuples."""
        assert freeze_for_hashing([1, 2, 3]) == (1, 2, 3)
        assert freeze_for_hashing([]) == ()
        assert freeze_for_hashing(["a", "b"]) == ("a", "b")

    def test_freeze_set(self) -> None:
        """Test freezing sets to sorted tuples."""
        assert freeze_for_hashing({1, 2, 3}) == (1, 2, 3)
        assert freeze_for_hashing(set()) == ()
        # Note: set order is not guaranteed, so we check if result is one of the possible tuples
        result = freeze_for_hashing({"a", "b"})
        assert result in [("a", "b"), ("b", "a")]

    def test_freeze_dict(self) -> None:
        """Test freezing dictionaries to tuples of sorted items."""
        # Simple dict
        assert freeze_for_hashing({"a": 1, "b": 2}) == (("a", 1), ("b", 2))

        # Empty dict
        assert freeze_for_hashing({}) == ()

        # Dict with different order should produce same result
        assert freeze_for_hashing({"b": 2, "a": 1}) == (("a", 1), ("b", 2))

    def test_freeze_nested_structures(self) -> None:
        """Test freezing nested data structures."""
        # List of lists
        assert freeze_for_hashing([[1, 2], [3, 4]]) == ((1, 2), (3, 4))

        # Dict with list values
        result = freeze_for_hashing({"a": [1, 2], "b": [3, 4]})
        assert result == (("a", (1, 2)), ("b", (3, 4)))

        # List with dict
        assert freeze_for_hashing([{"x": 1}, {"y": 2}]) == ((("x", 1),), (("y", 2),))

        # Complex nested structure
        complex_data = {"numbers": [1, 2, 3], "sets": {4, 5, 6}, "nested": {"inner": [7, 8, 9]}}
        result = freeze_for_hashing(complex_data)
        expected = (("nested", (("inner", (7, 8, 9)),)), ("numbers", (1, 2, 3)), ("sets", (4, 5, 6)))
        assert result == expected

    def test_freeze_already_frozen_types(self) -> None:
        """Test that already frozen types are converted to sorted tuples."""
        fs = frozenset({1, 2, 3})
        assert freeze_for_hashing(fs) == (1, 2, 3)

    def test_deterministic_results(self) -> None:
        """Test that freezing produces deterministic results."""
        # Same data should produce same frozen result
        data1 = {"b": [2, 3], "a": {"x": 1}}
        data2 = {"a": {"x": 1}, "b": [2, 3]}

        frozen1 = freeze_for_hashing(data1)
        frozen2 = freeze_for_hashing(data2)

        assert frozen1 == frozen2
        assert hash(frozen1) == hash(frozen2)

    def test_parameters_dict_freezing(self) -> None:
        """Test freezing parameter dictionaries as used in ops."""
        # Simple parameters
        params = {"region": "US", "category": "electronics"}
        frozen = freeze_for_hashing(params)
        assert frozen == (("category", "electronics"), ("region", "US"))

        # Parameters with list values
        params_with_list = {"regions": ["US", "EU"], "active": True}
        frozen = freeze_for_hashing(params_with_list)
        assert frozen == (("active", True), ("regions", ("US", "EU")))

        # Empty parameters
        assert freeze_for_hashing({}) == ()

    def test_hashability(self) -> None:
        """Test that frozen values are actually hashable."""
        test_cases = [
            [1, 2, 3],
            {"a": 1, "b": 2},
            {1, 2, 3},
            [[1, 2], [3, 4]],
            {"list": [1, 2], "set": {3, 4}},
        ]

        for data in test_cases:
            frozen = freeze_for_hashing(data)
            # Should not raise TypeError
            hash_value = hash(frozen)
            assert isinstance(hash_value, int)

    def test_edge_cases(self) -> None:
        """Test edge cases and corner scenarios."""
        # Dict with None values
        assert freeze_for_hashing({"a": None}) == (("a", None),)

        # List with None
        assert freeze_for_hashing([None, None]) == (None, None)

        # Mixed types in list
        assert freeze_for_hashing([1, "two", 3.0, None]) == (1, "two", 3.0, None)

        # Deeply nested empty structures
        assert freeze_for_hashing({"a": {"b": {"c": []}}}) == (("a", (("b", (("c", ()),)),)),)

    def test_mixed_type_sets(self) -> None:
        """Test sets with mixed types that can't be directly compared."""
        # Set with incomparable types (int and string)
        mixed_set = {1, "a", 2, "b"}
        frozen = freeze_for_hashing(mixed_set)
        # Should be deterministically sorted by type name then string representation
        # int comes before str alphabetically
        assert frozen == (1, 2, "a", "b")

        # Set with more complex mixed types
        complex_set = {True, 1, "test", None}
        frozen = freeze_for_hashing(complex_set)
        # bool, int, NoneType, str when sorted by type name
        # Note: True and 1 are equal in Python sets, so only one appears
        assert len(frozen) == 3  # True/1 are treated as same value in sets

        # Set with nested structures of different types
        nested_set = {("a", 1), ("b",), "c"}
        frozen = freeze_for_hashing(nested_set)
        # Should handle this deterministically
        assert isinstance(frozen, tuple)
        assert len(frozen) == 3
