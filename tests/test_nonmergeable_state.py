"""Test cases for NonMergeable state."""

import pytest

from dqx import states
from dqx.common import DQXError


def test_nonmergeable_basic_functionality() -> None:
    """Test NonMergeable state basic functionality."""
    # Test creation with different metric types
    state1 = states.NonMergeable(value=42.0, metric_type="UniqueCount")
    assert state1.value == 42.0
    assert state1.metric_type == "UniqueCount"

    state2 = states.NonMergeable(value=10.0, metric_type="DuplicateCount")
    assert state2.value == 10.0
    assert state2.metric_type == "DuplicateCount"


def test_nonmergeable_identity_raises_error() -> None:
    """Test that NonMergeable.identity() raises appropriate error."""
    with pytest.raises(DQXError, match="NonMergeable state does not support identity"):
        states.NonMergeable.identity()


def test_nonmergeable_merge_raises_error() -> None:
    """Test that NonMergeable.merge() raises appropriate error."""
    state1 = states.NonMergeable(value=10.0, metric_type="UniqueCount")
    state2 = states.NonMergeable(value=20.0, metric_type="UniqueCount")

    # Test merge with descriptive error message
    with pytest.raises(DQXError, match="Cannot merge UniqueCount: Operation not supported across partitions"):
        state1.merge(state2)

    # Test merge with different metric types
    state3 = states.NonMergeable(value=15.0, metric_type="DuplicateCount")
    with pytest.raises(DQXError, match="Cannot merge DuplicateCount: Operation not supported across partitions"):
        state3.merge(state1)


def test_nonmergeable_serialize_deserialize() -> None:
    """Test NonMergeable serialization and deserialization."""
    state = states.NonMergeable(value=42.0, metric_type="UniqueCount")

    # Test serialization
    serialized = state.serialize()
    assert isinstance(serialized, bytes)

    # Test deserialization
    deserialized = states.NonMergeable.deserialize(serialized)
    assert deserialized.value == 42.0
    assert deserialized.metric_type == "UniqueCount"
    assert isinstance(deserialized, states.NonMergeable)


def test_nonmergeable_equality() -> None:
    """Test NonMergeable equality comparison."""
    state1 = states.NonMergeable(value=42.0, metric_type="UniqueCount")
    state2 = states.NonMergeable(value=42.0, metric_type="UniqueCount")
    state3 = states.NonMergeable(value=43.0, metric_type="UniqueCount")
    state4 = states.NonMergeable(value=42.0, metric_type="DuplicateCount")

    # Same value and metric type
    assert state1 == state2

    # Different value, same metric type
    assert state1 != state3

    # Same value, different metric type
    assert state1 != state4

    # Test inequality with different types
    assert state1 != 42.0
    assert state1 != "NonMergeable"
    assert state1 is not None


def test_nonmergeable_hash() -> None:
    """Test NonMergeable hash functionality."""
    state1 = states.NonMergeable(value=42.0, metric_type="UniqueCount")
    state2 = states.NonMergeable(value=42.0, metric_type="UniqueCount")
    state3 = states.NonMergeable(value=43.0, metric_type="UniqueCount")
    state4 = states.NonMergeable(value=42.0, metric_type="DuplicateCount")

    # Same states should have same hash
    assert hash(state1) == hash(state2)

    # Different states should have different hashes (usually)
    assert hash(state1) != hash(state3)
    assert hash(state1) != hash(state4)

    # Can be used in sets/dicts
    state_set = {state1, state2, state3, state4}
    assert len(state_set) == 3  # state1 and state2 are equal


def test_nonmergeable_str_repr() -> None:
    """Test NonMergeable string representation."""
    state = states.NonMergeable(value=42.0, metric_type="UniqueCount")

    str_repr = str(state)
    assert "NonMergeable" in str_repr
    assert "42.0" in str_repr
    assert "UniqueCount" in str_repr

    repr_str = repr(state)
    assert "NonMergeable" in repr_str
    assert "value=42.0" in repr_str
    assert "metric_type='UniqueCount'" in repr_str
