from __future__ import annotations

import functools
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from dqx import specs
from dqx.common import DQXError, ResultKey
from dqx.states import State


@dataclass
class Metric:
    spec: specs.MetricSpec
    state: State
    key: ResultKey
    metric_id: uuid.UUID | None = None

    @property
    def value(self) -> float:
        return self.state.value

    @classmethod
    def build(
        cls,
        metric: specs.MetricSpec,
        key: ResultKey,
        state: State | None = None,
        metric_id: uuid.UUID | None = None,
    ) -> Self:
        return cls(
            metric_id=metric_id,
            spec=metric,
            state=state or metric.state(),
            key=key,
        )

    @classmethod
    def reduce(cls, metrics: Sequence[Metric]) -> Metric:
        return functools.reduce(lambda left, right: left.merge(right), metrics, metrics[0].identity())

    def merge(self, other: Metric) -> Metric:
        if self.spec != other.spec:
            raise DQXError(f"Cannot merge metrics with different spec: {self.spec.name} != {other.spec.name}")

        merged_state = self.state.merge(other.state)

        return Metric(
            spec=self.spec,
            state=merged_state,
            key=self.key,
        )

    def identity(self) -> Metric:
        return Metric(
            spec=self.spec,
            state=self.state.identity(),
            key=self.key,
        )