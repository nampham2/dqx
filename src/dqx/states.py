from __future__ import annotations

import math
from copy import copy
from typing import Any, Protocol, Self, runtime_checkable

import msgpack
import pyarrow as pa
from datasketches import cpc_sketch, cpc_union

from dqx.common import DQXError


@runtime_checkable
class State(Protocol):
    @property
    def value(self) -> float: ...

    @classmethod
    def identity(cls) -> Self: ...

    def serialize(self) -> bytes: ...

    @classmethod
    def deserialize(cls, state: bytes) -> Self: ...

    def merge(self, other: Self) -> Self: ...

    def __copy__(self) -> Self: ...

    def __eq__(self, other: Any) -> bool: ...


@runtime_checkable
class SketchState(State, Protocol):
    def fit(self, batch: pa.RecordBatch) -> None: ...


class SimpleAdditiveState(State):
    def __init__(self, value: float) -> None:
        self._value = float(value)

    @classmethod
    def identity(cls) -> SimpleAdditiveState:
        return SimpleAdditiveState(value=0.0)

    @property
    def value(self) -> float:
        return self._value

    def serialize(self) -> bytes:
        return msgpack.packb(self._value)

    @classmethod
    def deserialize(cls, data: bytes) -> SimpleAdditiveState:
        return cls(value=msgpack.unpackb(data))

    def merge(self, other: SimpleAdditiveState) -> SimpleAdditiveState:
        total = self._value + other._value
        return SimpleAdditiveState(value=total)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SimpleAdditiveState):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> SimpleAdditiveState:
        return SimpleAdditiveState(value=self._value)


# TODO(npham): It's probably better to have a flag to indicate whether the state is an identity or not
class Average(State):
    def __init__(self, avg: float, n: float) -> None:
        if n < 0:
            raise DQXError("Count cannot be negative!")

        if n == 0 and not math.isnan(avg):
            raise DQXError("Cannot have non-zero average with zero count!")
        self._avg = float(avg)
        self._n = n

    @classmethod
    def identity(cls) -> Average:
        return Average(avg=float("nan"), n=0)

    @property
    def value(self) -> float:
        return self._avg

    @property
    def n(self) -> float:
        return self._n

    def serialize(self) -> bytes:
        return msgpack.packb((self._avg, self._n))

    @classmethod
    def deserialize(cls, data: bytes) -> Average:
        avg, n = msgpack.unpackb(data)
        return cls(avg=avg, n=n)

    def merge(self, other: Any) -> Average:
        total = self.n + other.n

        if total == 0:
            return Average.identity()

        if self.n == 0:
            return copy(other)

        if other.n == 0:
            return copy(self)

        return Average(
            self._avg * (self.n / total) + other._avg * (other.n / total),
            total,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Average):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> Average:
        return Average(self._avg, self._n)


class Variance(State):
    """Sample variance"""

    def __init__(self, var: float, avg: float, n: float) -> None:
        if n < 2 and n != 0:
            raise DQXError("Sample variance calculation needs 2 or more samples!")

        self._var = float(var)
        self._avg = float(avg)
        self._n = n

    @classmethod
    def identity(cls) -> Variance:
        return Variance(var=float("nan"), avg=float("nan"), n=0)

    @property
    def value(self) -> float:
        return self._var

    @property
    def n(self) -> float:
        return self._n

    @property
    def avg(self) -> float:
        return self._avg

    def serialize(self) -> bytes:
        return msgpack.packb((self._var, self._avg, self._n))

    @classmethod
    def deserialize(cls, data: bytes) -> Variance:
        var, avg, n = msgpack.unpackb(data)
        return cls(var=var, avg=avg, n=n)

    def merge(self, other: Any) -> Variance:
        if not isinstance(other, Variance):
            raise DQXError("Cannot merge with non-Variance type!")

        if self.n == 0 and other.n == 0:
            return Variance.identity()

        if self.n == 0:
            return copy(other)

        if other.n == 0:
            return copy(self)

        n = self.n + other.n
        delta = self.avg - other.avg
        m2 = self.value + other.value + delta**2 * self.n * other.n / n

        avg = (self.n * self.avg + other.n * other.avg) / n
        var = m2 / (n - 1)

        return Variance(var=var, avg=avg, n=n)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Variance):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> Variance:
        return Variance(self._var, self._avg, self._n)


class First(State):
    def __init__(self, value: float, identity: bool = False) -> None:
        self._value = float(value)
        self._is_identity: bool = identity

    @classmethod
    def identity(cls) -> First:
        return First(value=0.0, identity=True)

    @property
    def value(self) -> float:
        return self._value

    def serialize(self) -> bytes:
        return msgpack.packb(self._value)

    @classmethod
    def deserialize(cls, data: bytes) -> First:
        return cls(value=msgpack.unpackb(data))

    def merge(self, other: First) -> First:
        if self._is_identity:
            return copy(other)
        return copy(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, First):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> First:
        return First(value=self._value)


class Minimum(State):
    def __init__(self, value: float) -> None:
        self._value = float(value)

    @classmethod
    def identity(cls) -> Minimum:
        return Minimum(value=float("inf"))

    @property
    def value(self) -> float:
        return self._value

    def serialize(self) -> bytes:
        return msgpack.packb(self._value)

    @classmethod
    def deserialize(cls, data: bytes) -> Minimum:
        return cls(value=msgpack.unpackb(data))

    def merge(self, other: Minimum) -> Minimum:
        value = min(self._value, other._value)
        return Minimum(value=value)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Minimum):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> Minimum:
        return Minimum(value=self._value)


class Maximum(State):
    def __init__(self, value: float) -> None:
        self._value = float(value)

    @classmethod
    def identity(cls) -> Maximum:
        return Maximum(value=float("-inf"))

    @property
    def value(self) -> float:
        return self._value

    def serialize(self) -> bytes:
        return msgpack.packb(self._value)

    @classmethod
    def deserialize(cls, data: bytes) -> Maximum:
        return cls(value=msgpack.unpackb(data))

    def merge(self, other: Maximum) -> Maximum:
        value = max(self._value, other._value)
        return Maximum(value=value)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Maximum):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> Maximum:
        return Maximum(value=self._value)


class CardinalitySketch(SketchState):
    def __init__(self, sketch: cpc_sketch) -> None:
        self._sketch = sketch

    @property
    def value(self) -> float:
        return self._sketch.get_estimate()

    @classmethod
    def identity(cls) -> CardinalitySketch:
        return CardinalitySketch(cpc_sketch())

    def serialize(self) -> bytes:
        return self._sketch.serialize()

    @classmethod
    def deserialize(cls, state: bytes) -> CardinalitySketch:
        sketch = cpc_sketch.deserialize(state)
        return cls(sketch=sketch)

    def fit(self, batch: pa.RecordBatch) -> None:
        for item in batch:
            self._sketch.update(item.as_py())

    def merge(self, other: CardinalitySketch) -> CardinalitySketch:
        union = cpc_union(self._sketch.lg_k)
        union.update(self._sketch)
        union.update(other._sketch)
        return CardinalitySketch(sketch=union.get_result())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CardinalitySketch):
            return False
        return self.serialize() == other.serialize()

    def __copy__(self) -> CardinalitySketch:
        return CardinalitySketch(sketch=copy(self._sketch))
