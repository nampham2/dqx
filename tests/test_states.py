import math
from copy import copy

import pytest

from dqx import states


def test_average() -> None:
    avg = states.Average(avg=1.0, n=10)
    assert avg.value == pytest.approx(1.0)
    assert avg.n == 10

    binary = avg.serialize()
    assert avg.deserialize(binary) == avg

    assert isinstance(avg, states.State)
    assert copy(avg) == avg


def test_average_identity() -> None:
    identity = states.Average.identity()
    assert math.isnan(identity.value)
    assert identity.n == 0


def test_average_merge() -> None:
    a = states.Average(avg=1.0, n=10)
    b = states.Average(avg=4.0, n=20)

    merged = a.merge(b)
    assert merged.value == pytest.approx(3.0)
    assert merged.n == 30

    merged = b.merge(a)
    assert merged.value == pytest.approx(3.0)
    assert merged.n == 30


def test_average_merge_with_identity() -> None:
    assert states.Average.identity().merge(states.Average.identity()) == states.Average.identity()

    a = states.Average(avg=1.0, n=10)
    assert a.merge(states.Average.identity()) == a
    assert states.Average.identity().merge(a) == a


def test_simple_additive_state() -> None:
    s = states.SimpleAdditiveState(value=1.0)
    assert isinstance(s, states.State)
    assert s.value == 1.0

    # Test serialization/deserialization
    binary = s.serialize()
    deserialized = states.SimpleAdditiveState.deserialize(binary)
    assert s == deserialized

    # Test identity
    identity = states.SimpleAdditiveState.identity()
    assert identity.value == 0.0

    # Test merge
    s2 = states.SimpleAdditiveState(value=2.0)
    merged = s.merge(s2)
    assert merged.value == 3.0

    # Test copy
    from copy import copy

    copied = copy(s)
    assert copied == s
    assert copied is not s


def test_average_constructor_errors() -> None:
    # Test negative count error
    with pytest.raises(states.DQXError, match="Count cannot be negative"):
        states.Average(avg=1.0, n=-1)

    # Test non-zero average with zero count error
    with pytest.raises(states.DQXError, match="Cannot have non-zero average with zero count"):
        states.Average(avg=1.0, n=0)


def test_variance() -> None:
    # Test basic variance state
    v = states.Variance(var=2.5, avg=5.0, n=10)
    assert v.value == pytest.approx(2.5)
    assert v.avg == pytest.approx(5.0)
    assert v.n == 10

    # Test serialization/deserialization
    binary = v.serialize()
    deserialized = states.Variance.deserialize(binary)
    assert v == deserialized

    # Test copy
    copied = copy(v)
    assert copied == v
    assert copied is not v


def test_variance_identity() -> None:
    identity = states.Variance.identity()
    assert math.isnan(identity.value)
    assert math.isnan(identity.avg)
    assert identity.n == 0


def test_variance_constructor_error() -> None:
    # Test error when n < 2 and n != 0
    with pytest.raises(states.DQXError, match="Sample variance calculation needs 2 or more samples"):
        states.Variance(var=1.0, avg=5.0, n=1)


def test_variance_merge() -> None:
    v1 = states.Variance(var=1.0, avg=2.0, n=5)
    v2 = states.Variance(var=2.0, avg=4.0, n=5)

    merged = v1.merge(v2)
    assert merged.n == 10
    assert merged.avg == pytest.approx(3.0)  # (5*2 + 5*4)/10 = 3

    # Test merge with identity
    identity = states.Variance.identity()
    assert v1.merge(identity) == v1
    assert identity.merge(v1) == v1
    assert identity.merge(identity) == identity


def test_variance_merge_error() -> None:
    v = states.Variance(var=1.0, avg=2.0, n=5)
    with pytest.raises(states.DQXError, match="Cannot merge with non-Variance type"):
        v.merge(states.Average(avg=1.0, n=5))  # type: ignore


def test_first() -> None:
    f = states.First(value=42.0)
    assert f.value == pytest.approx(42.0)
    assert isinstance(f, states.State)

    # Test serialization/deserialization
    binary = f.serialize()
    deserialized = states.First.deserialize(binary)
    assert f == deserialized

    # Test copy
    copied = copy(f)
    assert copied == f
    assert copied is not f


def test_first_identity() -> None:
    identity = states.First.identity()
    assert identity.value == pytest.approx(0.0)


def test_first_merge() -> None:
    f1 = states.First(value=10.0)
    f2 = states.First(value=20.0)

    # First should always return the first value (non-identity)
    merged = f1.merge(f2)
    assert merged.value == pytest.approx(10.0)

    # Test merge with identity - identity should return the other value
    identity = states.First.identity()
    assert identity.merge(f1) == f1
    assert f1.merge(identity) == f1


def test_minimum() -> None:
    m = states.Minimum(value=5.5)
    assert m.value == pytest.approx(5.5)
    assert isinstance(m, states.State)

    # Test serialization/deserialization
    binary = m.serialize()
    deserialized = states.Minimum.deserialize(binary)
    assert m == deserialized

    # Test copy
    copied = copy(m)
    assert copied == m
    assert copied is not m


def test_minimum_identity() -> None:
    identity = states.Minimum.identity()
    assert identity.value == float("inf")


def test_minimum_merge() -> None:
    m1 = states.Minimum(value=10.0)
    m2 = states.Minimum(value=5.0)

    merged = m1.merge(m2)
    assert merged.value == pytest.approx(5.0)

    merged = m2.merge(m1)
    assert merged.value == pytest.approx(5.0)

    # Test merge with identity
    identity = states.Minimum.identity()
    assert m1.merge(identity) == m1
    assert identity.merge(m1) == m1


def test_maximum() -> None:
    m = states.Maximum(value=15.5)
    assert m.value == pytest.approx(15.5)
    assert isinstance(m, states.State)

    # Test serialization/deserialization
    binary = m.serialize()
    deserialized = states.Maximum.deserialize(binary)
    assert m == deserialized

    # Test copy
    copied = copy(m)
    assert copied == m
    assert copied is not m


def test_maximum_identity() -> None:
    identity = states.Maximum.identity()
    assert identity.value == float("-inf")


def test_maximum_merge() -> None:
    m1 = states.Maximum(value=10.0)
    m2 = states.Maximum(value=15.0)

    merged = m1.merge(m2)
    assert merged.value == pytest.approx(15.0)

    merged = m2.merge(m1)
    assert merged.value == pytest.approx(15.0)

    # Test merge with identity
    identity = states.Maximum.identity()
    assert m1.merge(identity) == m1
    assert identity.merge(m1) == m1


def test_equality_with_different_types() -> None:
    """Test __eq__ methods return False for different types."""

    # Test SimpleAdditiveState
    s = states.SimpleAdditiveState(value=1.0)
    assert s != "not a state"
    assert s != 42
    assert s is not None

    # Test Average
    avg = states.Average(avg=1.0, n=10)
    assert avg != "not a state"
    assert avg != 42
    assert avg != s

    # Test Variance
    var = states.Variance(var=2.5, avg=5.0, n=10)
    assert var != "not a state"
    assert var != 42
    assert var != avg

    # Test First
    first = states.First(value=42.0)
    assert first != "not a state"
    assert first != 42
    assert first != avg

    # Test Minimum
    min_state = states.Minimum(value=5.5)
    assert min_state != "not a state"
    assert min_state != 42
    assert min_state != avg

    # Test Maximum
    max_state = states.Maximum(value=15.5)
    assert max_state != "not a state"
    assert max_state != 42
    assert max_state != avg


def test_duplicate_count_state() -> None:
    # Test basic functionality
    state = states.DuplicateCount(value=42.0)
    assert state.value == 42.0

    # Test identity raises error with descriptive message
    with pytest.raises(states.DQXError, match="NonMergeable state does not support identity"):
        states.DuplicateCount.identity()

    # Test serialization
    serialized = state.serialize()
    deserialized = states.DuplicateCount.deserialize(serialized)
    assert deserialized.value == 42.0
    assert state == deserialized

    # Test copy
    copied = copy(state)
    assert copied.value == 42.0
    assert copied == state

    # Test merge raises error with descriptive message
    state1 = states.DuplicateCount(value=10.0)
    state2 = states.DuplicateCount(value=20.0)

    with pytest.raises(states.DQXError, match="Cannot merge DuplicateCount: Operation not supported across partitions"):
        state1.merge(state2)


def test_duplicate_count_state_equality() -> None:
    state1 = states.DuplicateCount(value=42.0)
    state2 = states.DuplicateCount(value=42.0)
    state3 = states.DuplicateCount(value=43.0)

    assert state1 == state2
    assert state1 != state3
    assert state1 != "not a state"
    assert state1 != 42


class TestMinLength:
    """Tests for MinLength state."""

    def test_min_length_basic(self) -> None:
        """Test basic MinLength state creation and value."""
        m = states.MinLength(value=5.0)
        assert m.value == pytest.approx(5.0)
        assert isinstance(m, states.State)

    def test_min_length_identity(self) -> None:
        """Test MinLength identity is float('inf')."""
        identity = states.MinLength.identity()
        assert identity.value == float("inf")

    def test_min_length_serialize_deserialize(self) -> None:
        """Test MinLength serialization round-trip."""
        m = states.MinLength(value=3.0)
        binary = m.serialize()
        deserialized = states.MinLength.deserialize(binary)
        assert m == deserialized

    def test_min_length_copy(self) -> None:
        """Test MinLength copy creates equal but distinct instance."""
        m = states.MinLength(value=7.0)
        copied = copy(m)
        assert copied == m
        assert copied is not m

    def test_min_length_merge_takes_minimum(self) -> None:
        """Test MinLength merge keeps the smaller value."""
        m1 = states.MinLength(value=10.0)
        m2 = states.MinLength(value=5.0)

        merged = m1.merge(m2)
        assert merged.value == pytest.approx(5.0)

        merged2 = m2.merge(m1)
        assert merged2.value == pytest.approx(5.0)

    def test_min_length_merge_with_identity(self) -> None:
        """Test MinLength merge with identity returns original value."""
        m = states.MinLength(value=3.0)
        identity = states.MinLength.identity()

        assert m.merge(identity) == m
        assert identity.merge(m) == m

    def test_min_length_equality(self) -> None:
        """Test MinLength equality comparisons."""
        m1 = states.MinLength(value=5.0)
        m2 = states.MinLength(value=5.0)
        m3 = states.MinLength(value=6.0)

        assert m1 == m2
        assert m1 != m3
        assert m1 != "not a state"
        assert m1 != 42


class TestMaxLength:
    """Tests for MaxLength state."""

    def test_max_length_basic(self) -> None:
        """Test basic MaxLength state creation and value."""
        m = states.MaxLength(value=5.0)
        assert m.value == pytest.approx(5.0)
        assert isinstance(m, states.State)

    def test_max_length_identity(self) -> None:
        """Test MaxLength identity is float('-inf')."""
        identity = states.MaxLength.identity()
        assert identity.value == float("-inf")

    def test_max_length_serialize_deserialize(self) -> None:
        """Test MaxLength serialization round-trip."""
        m = states.MaxLength(value=3.0)
        binary = m.serialize()
        deserialized = states.MaxLength.deserialize(binary)
        assert m == deserialized

    def test_max_length_copy(self) -> None:
        """Test MaxLength copy creates equal but distinct instance."""
        m = states.MaxLength(value=7.0)
        copied = copy(m)
        assert copied == m
        assert copied is not m

    def test_max_length_merge_takes_maximum(self) -> None:
        """Test MaxLength merge keeps the larger value."""
        m1 = states.MaxLength(value=10.0)
        m2 = states.MaxLength(value=5.0)

        merged = m1.merge(m2)
        assert merged.value == pytest.approx(10.0)

        merged2 = m2.merge(m1)
        assert merged2.value == pytest.approx(10.0)

    def test_max_length_merge_with_identity(self) -> None:
        """Test MaxLength merge with identity returns original value."""
        m = states.MaxLength(value=3.0)
        identity = states.MaxLength.identity()

        assert m.merge(identity) == m
        assert identity.merge(m) == m

    def test_max_length_equality(self) -> None:
        """Test MaxLength equality comparisons."""
        m1 = states.MaxLength(value=5.0)
        m2 = states.MaxLength(value=5.0)
        m3 = states.MaxLength(value=6.0)

        assert m1 == m2
        assert m1 != m3
        assert m1 != "not a state"
        assert m1 != 42
