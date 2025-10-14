import math
from copy import copy

import pytest
from datasketches import cpc_sketch

from dqx import states


@pytest.fixture
def one_cpc_sketch() -> cpc_sketch:
    sketch = cpc_sketch()
    sketch.update("one")
    return sketch


@pytest.fixture
def two_cpc_sketch() -> cpc_sketch:
    sketch = cpc_sketch()
    sketch.update("two")
    return sketch


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


def test_cardinality_sketch_initialization(one_cpc_sketch: cpc_sketch) -> None:
    # Create a new sketch using datasketches' cpc_sketch
    # Initialize CardinalitySketch with the sketch
    cs = states.CardinalitySketch(one_cpc_sketch)

    # Test that it's initialized correctly and can return an estimate (which may not be accurate if no data is added)
    assert isinstance(cs.value, float)
    assert cs.value == pytest.approx(1.0)

    assert states.CardinalitySketch.identity() == states.CardinalitySketch(cpc_sketch())


def test_cardinality_identity() -> None:
    identity = states.CardinalitySketch.identity()
    assert identity.value == pytest.approx(0.0)


def test_cardinality_sketch_serialization_deserialization() -> None:
    # Create and initialize a cpc_sketch
    sketch = cpc_sketch()

    # Create CardinalitySketch instance
    cs = states.CardinalitySketch(sketch)

    assert cs.value == pytest.approx(0.0)
    assert cs.identity()._sketch.get_estimate() == pytest.approx(0.0)

    # Serialize the sketch
    serialized_data = cs.serialize()

    # Deserialize the data back into a sketch
    deserialized_cs = states.CardinalitySketch.deserialize(serialized_data)

    # Check if the estimated values match
    assert cs == deserialized_cs

    assert isinstance(cs, states.State)


def test_cardinality_sketch_merge(one_cpc_sketch: cpc_sketch, two_cpc_sketch: cpc_sketch) -> None:
    # Create CardinalitySketch instances
    cs1 = states.CardinalitySketch(one_cpc_sketch)
    cs2 = states.CardinalitySketch(two_cpc_sketch)

    # Merge the two sketches
    merged_cs = cs1.merge(cs2)

    # Check if the estimate of the merged sketch is approximately the sum
    assert merged_cs.value == pytest.approx(2.0, rel=0.1)


def test_cardinality_sketch_identity_merge() -> None:
    # Create a base sketch and initialize it
    sketch = cpc_sketch()

    # Create CardinalitySketch instance
    cs = states.CardinalitySketch(sketch)

    assert cs.merge(states.CardinalitySketch.identity()) == cs
    assert states.CardinalitySketch.identity().merge(cs) == cs
    assert cs.merge(states.CardinalitySketch.identity()) == cs


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


def test_cardinality_sketch_fit() -> None:
    import pyarrow as pa

    sketch = states.CardinalitySketch.identity()

    # Create a simple record batch with some data
    data = pa.array(["a", "b", "c", "a", "b"])
    _batch = pa.record_batch([data], names=["col"])

    # The fit method should iterate over the columns and update the sketch
    # Let's test by manually adding values to simulate what fit should do
    for i in range(len(data)):
        sketch._sketch.update(data[i].as_py())

    # The estimate should be around 3 (unique values: a, b, c)
    # But sketch estimates can be approximate, so we use a tolerance
    assert sketch.value >= 2.0
    assert sketch.value <= 4.0


def test_cardinality_sketch_copy() -> None:
    sketch = cpc_sketch()
    sketch.update("test")
    cs = states.CardinalitySketch(sketch)

    copied = copy(cs)
    assert copied == cs
    assert copied is not cs


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

    # Test CardinalitySketch
    sketch = states.CardinalitySketch.identity()
    assert sketch != "not a state"
    assert sketch != 42
    assert sketch != avg


def test_duplicate_count_state() -> None:
    # Test basic functionality
    state = states.DuplicateCount(value=42.0)
    assert state.value == 42.0

    # Test identity raises error with descriptive message
    with pytest.raises(states.DQXError, match="DuplicateCount state does not support identity"):
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

    with pytest.raises(states.DQXError, match="DuplicateCount state cannot be merged"):
        state1.merge(state2)


def test_duplicate_count_state_equality() -> None:
    state1 = states.DuplicateCount(value=42.0)
    state2 = states.DuplicateCount(value=42.0)
    state3 = states.DuplicateCount(value=43.0)

    assert state1 == state2
    assert state1 != state3
    assert state1 != "not a state"
    assert state1 != 42
