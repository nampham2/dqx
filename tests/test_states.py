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
