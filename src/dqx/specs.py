import inspect
from collections.abc import Sequence
from typing import Any, Literal, Protocol, Type, runtime_checkable

from dqx import ops, states
from dqx.common import Parameters

MetricType = Literal[
    "NumRows",
    "First",
    "Average",
    "Variance",
    "Minimum",
    "Maximum",
    "Sum",
    "NullCount",
    "NegativeCount",
    "ApproxCardinality",
]


@runtime_checkable
class MetricSpec(Protocol):
    metric_type: MetricType

    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Parameters: ...

    @property
    def analyzers(self) -> Sequence[ops.Op]: ...

    def state(self) -> states.State: ...

    @classmethod
    def deserialize(cls, state: bytes) -> states.State: ...

    def __hash__(self) -> int: ...

    def __eq__(self, other: Any) -> bool: ...


class NumRows:
    metric_type: MetricType = "NumRows"

    def __init__(self) -> None:
        self._analyzers = (ops.NumRows(),)

    @property
    def name(self) -> str:
        return "num_rows()"

    @property
    def parameters(self) -> Parameters:
        return {}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        num_rows = self._analyzers[0].value()
        return states.SimpleAdditiveState(value=num_rows)

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NumRows):
            return False
        return self.name == other.name and self.parameters == other.parameters


class First:
    metric_type: MetricType = "First"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.First(self._column),)

    @property
    def name(self) -> str:
        return f"first({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.First:
        return states.First(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.First.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, First):
            return False
        return self.name == other.name and self.parameters == other.parameters


class Average:
    metric_type: MetricType = "Average"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.NumRows(), ops.Average(self._column))

    @property
    def name(self) -> str:
        return f"average({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Average:
        num_rows, avg = self._analyzers[0].value(), self._analyzers[1].value()
        return states.Average(avg=avg, n=num_rows)

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Average.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Average):
            return False
        return self.name == other.name and self.parameters == other.parameters


class Variance:
    metric_type: MetricType = "Variance"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.NumRows(), ops.Average(self._column), ops.Variance(self._column))

    @property
    def name(self) -> str:
        return f"variance({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Variance:
        num_rows, avg, var = self._analyzers[0].value(), self._analyzers[1].value(), self._analyzers[2].value()
        return states.Variance(var=var, avg=avg, n=num_rows)

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Variance.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Variance):
            return False
        return self.name == other.name and self.parameters == other.parameters


class Minimum:
    metric_type: MetricType = "Minimum"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.Minimum(self._column),)

    @property
    def name(self) -> str:
        return f"minimum({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Minimum:
        return states.Minimum(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Minimum.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Minimum):
            return False
        return self.name == other.name and self.parameters == other.parameters


class Maximum:
    metric_type: MetricType = "Maximum"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.Maximum(self._column),)

    @property
    def name(self) -> str:
        return f"maximum({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Maximum:
        return states.Maximum(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Maximum.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Maximum):
            return False
        return self.name == other.name and self.parameters == other.parameters


class Sum:
    metric_type: MetricType = "Sum"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.Sum(self._column),)

    @property
    def name(self) -> str:
        return f"sum({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sum):
            return False
        return self.name == other.name and self.parameters == other.parameters


class NullCount:
    metric_type: MetricType = "NullCount"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.NullCount(self._column),)

    @property
    def name(self) -> str:
        return f"null_count({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NullCount):
            return False
        return self.name == other.name and self.parameters == other.parameters


class NegativeCount:
    metric_type: MetricType = "NegativeCount"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.NegativeCount(self._column),)

    @property
    def name(self) -> str:
        return f"non_negative({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NegativeCount):
            return False
        return self.name == other.name and self.parameters == other.parameters


class ApproxCardinality:
    metric_type: MetricType = "ApproxCardinality"

    def __init__(self, column: str) -> None:
        self._column = column
        self._analyzers = (ops.ApproxCardinality(self._column),)

    @property
    def name(self) -> str:
        return f"approx_cardinality({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    # TODO(npham): rename the state method
    # TODO(npham): The analyzer returns ApproxCardinality instead of a CPC sketch.
    #              This is because the analyzer does not know which sketch class to use.
    #              This is inconsistent with the SQLSketch. Let's decide the unified returning values of analyzers: spec or value.
    def state(self) -> states.CardinalitySketch:
        return self._analyzers[0].value()

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.CardinalitySketch.deserialize(state)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.parameters.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ApproxCardinality):
            return False
        return self.name == other.name and self.parameters == other.parameters


def _build_registry() -> dict[MetricType, Type[MetricSpec]]:
    """Automatically build the registry using reflection.
    
    This function discovers all MetricSpec implementations in the current module
    and creates a registry mapping from MetricType to the corresponding class.
    
    Returns:
        Dictionary mapping metric type names to their implementation classes.
    """
    registry_dict: dict[MetricType, Type[MetricSpec]] = {}
    
    # Get all classes defined in this module
    current_module = inspect.currentframe().f_globals  # type: ignore
    
    for name, obj in current_module.items():
        # Check if it's a class and has the required attributes
        if (
            inspect.isclass(obj) 
            and hasattr(obj, 'metric_type') 
            and isinstance(obj, type)
            and obj is not MetricSpec  # Exclude the protocol itself
        ):
            metric_type = getattr(obj, 'metric_type')
            if metric_type:
                registry_dict[metric_type] = obj  # type: ignore
    
    return registry_dict


# Automatically create the registry using reflection
registry: dict[MetricType, Type[MetricSpec]] = _build_registry()
