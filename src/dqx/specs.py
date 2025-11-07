import inspect
import typing
from collections.abc import Sequence
from typing import Any, Literal, Protocol, Self, Type, runtime_checkable

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
    "DuplicateCount",
    "CountValues",
    "UniqueCount",
    "CustomSQL",
    "DayOverDay",
    "WeekOverWeek",
    "Stddev",
]


# Initialize empty registry
registry: dict[MetricType, Type["MetricSpec"]] = {}


def register_spec(metric_type: MetricType, spec_class: Type["MetricSpec"]) -> None:
    """Register a spec in the global registry."""
    if metric_type in registry:
        raise ValueError(f"Spec '{metric_type}' is already registered")
    registry[metric_type] = spec_class


def auto_register(cls: Type["MetricSpec"]) -> Type["MetricSpec"]:
    """Decorator to automatically register a spec class."""
    metric_type = cls.metric_type
    register_spec(metric_type, cls)
    return cls


def _reconstruct_base_spec(base_metric_type: str, base_parameters: dict[str, Any]) -> "MetricSpec":
    """Reconstruct a MetricSpec from base_metric_type and base_parameters.

    This helper extracts the common logic for reconstructing base specs used by
    extended metrics (DayOverDay, WeekOverWeek, Stddev).

    Args:
        base_metric_type: The type of the base metric
        base_parameters: Parameters for the base metric

    Returns:
        A reconstructed MetricSpec instance

    Raises:
        KeyError: If base_metric_type is not in registry
    """
    metric_type = typing.cast(MetricType, base_metric_type)
    spec_class = registry[metric_type]

    # Extended metrics (DayOverDay, WeekOverWeek, Stddev) have specific constructor signatures
    # and don't accept a 'parameters' argument
    if metric_type in ["DayOverDay", "WeekOverWeek", "Stddev"]:
        # For extended metrics, pass all parameters as constructor args
        return typing.cast(Any, spec_class)(**base_parameters)
    else:
        # For simple metrics, split into constructor params and additional params
        sig = inspect.signature(spec_class.__init__)
        constructor_params = {}
        additional_params = {}

        for key, value in base_parameters.items():
            if key in sig.parameters and key != "parameters":
                constructor_params[key] = value
            else:
                additional_params[key] = value

        # Create spec with constructor params and additional params
        # Cast to Any to avoid mypy issues with protocol constructors
        return typing.cast(Any, spec_class)(**constructor_params, parameters=additional_params)


@runtime_checkable
class MetricSpec(Protocol):
    """Base protocol for all metrics."""

    metric_type: MetricType
    is_extended: Literal[False] | Literal[True]

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


@runtime_checkable
class SimpleMetricSpec(MetricSpec, Protocol):
    """Protocol for simple metrics that support cloning."""

    is_extended: Literal[False]

    def clone(self) -> Self: ...


@runtime_checkable
class ExtendedMetricSpec(MetricSpec, Protocol):
    """Protocol for extended metrics that don't support cloning."""

    is_extended: Literal[True]


@auto_register
class NumRows(SimpleMetricSpec):
    metric_type: MetricType = "NumRows"
    is_extended: Literal[False] = False

    def __init__(self, parameters: Parameters | None = None) -> None:
        self._parameters = parameters or {}
        self._analyzers = (ops.NumRows(parameters=self._parameters),)

    @property
    def name(self) -> str:
        return "num_rows()"

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        num_rows = self._analyzers[0].value()
        return states.SimpleAdditiveState(value=num_rows)

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NumRows):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class First(SimpleMetricSpec):
    metric_type: MetricType = "First"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (ops.First(self._column, parameters=self._parameters),)

    @property
    def name(self) -> str:
        return f"first({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.First:
        return states.First(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.First.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, First):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class Average(SimpleMetricSpec):
    metric_type: MetricType = "Average"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (
            ops.NumRows(parameters=self._parameters),
            ops.Average(self._column, parameters=self._parameters),
        )

    @property
    def name(self) -> str:
        return f"average({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Average:
        num_rows, avg = self._analyzers[0].value(), self._analyzers[1].value()
        return states.Average(avg=avg, n=num_rows)

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Average.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Average):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class Variance(SimpleMetricSpec):
    metric_type: MetricType = "Variance"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (
            ops.NumRows(parameters=self._parameters),
            ops.Average(self._column, parameters=self._parameters),
            ops.Variance(self._column, parameters=self._parameters),
        )

    @property
    def name(self) -> str:
        return f"variance({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Variance:
        num_rows, avg, var = self._analyzers[0].value(), self._analyzers[1].value(), self._analyzers[2].value()
        return states.Variance(var=var, avg=avg, n=num_rows)

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Variance.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Variance):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class Minimum(SimpleMetricSpec):
    metric_type: MetricType = "Minimum"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (ops.Minimum(self._column, parameters=self._parameters),)

    @property
    def name(self) -> str:
        return f"minimum({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Minimum:
        return states.Minimum(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Minimum.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Minimum):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class Maximum(SimpleMetricSpec):
    metric_type: MetricType = "Maximum"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (ops.Maximum(self._column, parameters=self._parameters),)

    @property
    def name(self) -> str:
        return f"maximum({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.Maximum:
        return states.Maximum(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.Maximum.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Maximum):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class Sum(SimpleMetricSpec):
    metric_type: MetricType = "Sum"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (ops.Sum(self._column, parameters=self._parameters),)

    @property
    def name(self) -> str:
        return f"sum({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sum):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class NullCount(SimpleMetricSpec):
    metric_type: MetricType = "NullCount"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (ops.NullCount(self._column, parameters=self._parameters),)

    @property
    def name(self) -> str:
        return f"null_count({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NullCount):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class NegativeCount(SimpleMetricSpec):
    metric_type: MetricType = "NegativeCount"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (ops.NegativeCount(self._column, parameters=self._parameters),)

    @property
    def name(self) -> str:
        return f"non_negative({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NegativeCount):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class DuplicateCount(SimpleMetricSpec):
    metric_type: MetricType = "DuplicateCount"
    is_extended: Literal[False] = False

    def __init__(self, columns: list[str], parameters: Parameters | None = None) -> None:
        if not columns:
            raise ValueError("At least one column must be specified")
        # Sort columns for consistent behavior
        self._columns = sorted(columns)
        self._parameters = parameters or {}
        self._analyzers = (ops.DuplicateCount(self._columns, parameters=self._parameters),)

    @property
    def name(self) -> str:
        columns_str = ",".join(self._columns)
        return f"duplicate_count({columns_str})"

    @property
    def parameters(self) -> Parameters:
        return {"columns": self._columns, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.NonMergeable:
        return states.NonMergeable(value=self._analyzers[0].value(), metric_type="DuplicateCount")

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.NonMergeable.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        # Pass a copy of the columns list to avoid shared references
        return self.__class__(self._columns.copy(), parameters=self._parameters.copy())

    def __hash__(self) -> int:
        # Convert the columns list to a tuple for hashing
        return hash((self.name, tuple(self._columns), tuple(sorted(self._parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DuplicateCount):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class CountValues(SimpleMetricSpec):
    metric_type: MetricType = "CountValues"
    is_extended: Literal[False] = False

    def __init__(
        self, column: str, values: int | str | bool | list[int] | list[str], parameters: Parameters | None = None
    ) -> None:
        self._column = column
        self._values = values
        self._parameters = parameters or {}
        self._analyzers = (ops.CountValues(self._column, self._values, parameters=self._parameters),)

    @property
    def name(self) -> str:
        # Match the op's name format
        op = self._analyzers[0]
        return op.name

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, "values": self._values, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        # If values is a list, create a copy to avoid shared references
        if isinstance(self._values, list):
            return self.__class__(self._column, self._values.copy(), parameters=self._parameters.copy())
        else:
            return self.__class__(self._column, self._values, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        # Convert lists to tuples for hashing
        hashable_values = self._values if not isinstance(self._values, list) else tuple(self._values)
        return hash((self.name, self._column, hashable_values, tuple(sorted(self._parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CountValues):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class UniqueCount(SimpleMetricSpec):
    metric_type: MetricType = "UniqueCount"
    is_extended: Literal[False] = False

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        self._column = column
        self._parameters = parameters or {}
        self._analyzers = (ops.UniqueCount(self._column, parameters=self._parameters),)

    @property
    def name(self) -> str:
        return f"unique_count({self._column})"

    @property
    def parameters(self) -> Parameters:
        return {"column": self._column, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.NonMergeable:
        return states.NonMergeable(value=self._analyzers[0].value(), metric_type="UniqueCount")

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.NonMergeable.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._column, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UniqueCount):
            return False
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class CustomSQL(SimpleMetricSpec):
    metric_type: MetricType = "CustomSQL"
    is_extended: Literal[False] = False

    def __init__(self, sql_expression: str, parameters: Parameters | None = None) -> None:
        self._sql_expression = sql_expression
        self._parameters = parameters or {}
        self._analyzers = (ops.CustomSQL(self._sql_expression, parameters=self._parameters),)

    @property
    def name(self) -> str:
        # Use hash-based naming with parentheses format
        import hashlib

        sql_hash = hashlib.md5(self._sql_expression.encode()).hexdigest()[:8]
        return f"custom_sql({sql_hash})"

    @property
    def parameters(self) -> Parameters:
        return {"sql_expression": self._sql_expression, **self._parameters}

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return self._analyzers

    def state(self) -> states.SimpleAdditiveState:
        return states.SimpleAdditiveState(value=self._analyzers[0].value())

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.SimpleAdditiveState.deserialize(state)

    def clone(self) -> Self:
        """Create a new instance with the same parameters but new analyzer prefixes."""
        return self.__class__(self._sql_expression, parameters=self._parameters.copy())

    def __hash__(self) -> int:
        return hash((self.name, self._sql_expression, tuple(sorted(self._parameters.items()))))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CustomSQL):
            return False
        return self._sql_expression == other._sql_expression and self.parameters == other.parameters

    def __str__(self) -> str:
        return self.name


@auto_register
class DayOverDay(ExtendedMetricSpec):
    metric_type: MetricType = "DayOverDay"
    is_extended: Literal[True] = True

    def __init__(self, base_metric_type: str, base_parameters: dict[str, Any]) -> None:
        self._base_metric_type = base_metric_type
        self._base_parameters = base_parameters
        self._analyzers = ()

        # Reconstruct and store the base spec for internal operations
        self._base_spec = _reconstruct_base_spec(self._base_metric_type, self._base_parameters)

    @classmethod
    def from_base_spec(cls, base_spec: MetricSpec) -> Self:
        return cls(base_metric_type=base_spec.metric_type, base_parameters=base_spec.parameters)

    @property
    def base_spec(self) -> MetricSpec:
        """Get the base metric specification."""
        return self._base_spec

    @property
    def name(self) -> str:
        return f"dod({self._base_spec.name})"

    @property
    def parameters(self) -> Parameters:
        return {
            "base_metric_type": self._base_metric_type,
            "base_parameters": self._base_parameters,
        }

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return ()

    def state(self) -> states.NonMergeable:
        return states.NonMergeable(value=0.0, metric_type="DayOverDay")

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.NonMergeable.deserialize(state)

    def __hash__(self) -> int:
        # Use base_spec which handles nested hashing properly
        return hash(("DayOverDay", self._base_spec))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DayOverDay):
            return False
        # Compare base specs directly
        return self._base_spec == other._base_spec

    def __str__(self) -> str:
        return self.name


@auto_register
class WeekOverWeek(ExtendedMetricSpec):
    metric_type: MetricType = "WeekOverWeek"
    is_extended: Literal[True] = True

    def __init__(self, base_metric_type: str, base_parameters: dict[str, Any]) -> None:
        self._base_metric_type = base_metric_type
        self._base_parameters = base_parameters
        self._analyzers = ()

        # Reconstruct and store the base spec for internal operations
        self._base_spec = _reconstruct_base_spec(self._base_metric_type, self._base_parameters)

    @classmethod
    def from_base_spec(cls, base_spec: MetricSpec) -> Self:
        return cls(base_metric_type=base_spec.metric_type, base_parameters=base_spec.parameters)

    @property
    def base_spec(self) -> MetricSpec:
        """Get the base metric specification."""
        return self._base_spec

    @property
    def name(self) -> str:
        return f"wow({self._base_spec.name})"

    @property
    def parameters(self) -> Parameters:
        return {
            "base_metric_type": self._base_metric_type,
            "base_parameters": self._base_parameters,
        }

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return ()

    def state(self) -> states.NonMergeable:
        return states.NonMergeable(value=0.0, metric_type="WeekOverWeek")

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.NonMergeable.deserialize(state)

    def __hash__(self) -> int:
        # Use base_spec which handles nested hashing properly
        return hash(("WeekOverWeek", self._base_spec))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, WeekOverWeek):
            return False
        # Compare base specs directly
        return self._base_spec == other._base_spec

    def __str__(self) -> str:
        return self.name


@auto_register
class Stddev(ExtendedMetricSpec):
    metric_type: MetricType = "Stddev"
    is_extended: Literal[True] = True

    def __init__(self, base_metric_type: str, base_parameters: dict[str, Any], offset: int, n: int) -> None:
        self._base_metric_type = base_metric_type
        self._base_parameters = base_parameters
        self._offset = offset
        self._n = n
        self._analyzers = ()

        # Reconstruct and store the base spec for internal operations
        self._base_spec = _reconstruct_base_spec(self._base_metric_type, self._base_parameters)

    @classmethod
    def from_base_spec(cls, base_spec: MetricSpec, offset: int, n: int) -> Self:
        return cls(
            base_metric_type=base_spec.metric_type,
            base_parameters=base_spec.parameters,
            offset=offset,
            n=n,
        )

    @property
    def base_spec(self) -> MetricSpec:
        """Get the base metric specification."""
        return self._base_spec

    @property
    def name(self) -> str:
        return f"stddev({self._base_spec.name}, offset={self._offset}, n={self._n})"

    @property
    def parameters(self) -> Parameters:
        return {
            "base_metric_type": self._base_metric_type,
            "base_parameters": self._base_parameters,
            "offset": self._offset,
            "n": self._n,
        }

    @property
    def analyzers(self) -> Sequence[ops.Op]:
        return ()

    def state(self) -> states.NonMergeable:
        return states.NonMergeable(value=0.0, metric_type="Stddev")

    @classmethod
    def deserialize(cls, state: bytes) -> states.State:
        return states.NonMergeable.deserialize(state)

    def __hash__(self) -> int:
        # Use base_spec which handles nested hashing properly
        return hash(("Stddev", self._base_spec, self._offset, self._n))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Stddev):
            return False
        # Compare base specs directly
        return self._base_spec == other._base_spec and self._offset == other._offset and self._n == other._n

    def __str__(self) -> str:
        return self.name
