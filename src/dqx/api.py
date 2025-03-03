from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, Self, overload, runtime_checkable

import sympy as sp
from returns.result import Result

from dqx import common, functions, graph
from dqx.analyzer import Analyzer
from dqx.common import DQXError, DuckDataSource, ResultKey, ResultKeyProvider, SeverityLevel, SymbolicValidator
from dqx.orm.repositories import MetricDB
from dqx.provider import MetricProvider
from dqx.specs import MetricSpec

CheckProducer = Callable[[MetricProvider, common.Context], None]
CheckCreator = Callable[[CheckProducer], CheckProducer]
SymbolTable = Mapping[sp.Symbol, Result[float, str]]


@runtime_checkable
class AssertListener(Protocol):
    def set_label(self, label: str) -> None: ...
    def set_severity(self, severity: SeverityLevel) -> None: ...
    def set_validator(self, validator: SymbolicValidator) -> None: ...


class SymbolicAssert:
    def __init__(self, actual: sp.Expr, listeners: list[AssertListener]) -> None:
        self._actual = actual
        self._label: str | None = None
        self._severity: SeverityLevel | None = None
        self._validator: SymbolicValidator | None = None
        self.listeners = listeners

    def label(self, label: str) -> Self:
        self._label = label

        # Update listeners
        for listener in self.listeners:
            listener.set_label(label)
        return self

    def severity(self, severity: SeverityLevel) -> Self:
        self._severity = severity

        # Update listeners
        for listener in self.listeners:
            listener.set_severity(severity)

        return self

    def _update_validator(self, validator: SymbolicValidator) -> None:
        self._validator = validator

        # Update listeners
        for listener in self.listeners:
            listener.set_validator(validator)

    def is_geq(self, other: float, tol: float = functions.EPSILON) -> None:
        self._update_validator(
            SymbolicValidator(name=f"\u2265 {other}", fn=functools.partial(functions.is_geq, b=other, tol=tol))
        )

    def is_gt(self, other: float, tol: float = functions.EPSILON) -> None:
        self._update_validator(
            SymbolicValidator(name=f"> {other}", fn=functools.partial(functions.is_gt, b=other, tol=tol))
        )

    def is_leq(self, other: float, tol: float = functions.EPSILON) -> None:
        self._update_validator(
            SymbolicValidator(name=f"\u2264 {other}", fn=functools.partial(functions.is_leq, b=other, tol=tol))
        )

    def is_lt(self, other: float, tol: float = functions.EPSILON) -> None:
        self._update_validator(
            SymbolicValidator(name=f"< {other}", fn=functools.partial(functions.is_lt, b=other, tol=tol))
        )

    def is_eq(self, other: float, tol: float = functions.EPSILON) -> None:
        self._update_validator(
            SymbolicValidator(name=f"= {other}", fn=functools.partial(functions.is_eq, b=other, tol=tol))
        )

    def is_negative(self, tol: float = functions.EPSILON) -> None:
        self._update_validator(SymbolicValidator(name="< 0", fn=functools.partial(functions.is_negative, tol=tol)))

    def is_positive(self, tol: float = functions.EPSILON) -> None:
        self._update_validator(SymbolicValidator(name="> 0", fn=functools.partial(functions.is_positive, tol=tol)))


class Context:
    def __init__(self, graph: graph.RootNode) -> None:
        self._graph = graph

    @property
    def key(self) -> ResultKeyProvider:
        return ResultKeyProvider()

    def assert_that(self, expr: sp.Expr) -> SymbolicAssert:
        node = graph.AssertionNode(actual=expr)
        sa = SymbolicAssert(actual=expr, listeners=[node])

        # Attach to the last check node, assuming that the last node is the current check
        # TODO(npham): Find a better way to attach the assertion to the correct check node
        self._graph.children[-1].add_child(node)

        return sa

    def pending_metrics(self) -> Sequence[MetricSpec]:
        return list(set(node.spec for node in self._graph.pending_metrics()))

    def eval_symbols(self, key: ResultKey) -> None:
        for symbol in self._graph.ready_symbols():
            symbol.evaluate(key)

    def validate(self, key: ResultKey) -> None:
        self.eval_symbols(key)
        for assertion in self._graph.assertions():
            assertion.evaluate()


class VerificationSuite:
    def __init__(
        self,
        checks: Sequence[CheckProducer],
        db: MetricDB,
        name: str,
    ) -> None:
        self._checks = checks
        self._provider = MetricProvider(db)
        self._name = name

    @classmethod
    def from_packages(cls, *packages: str) -> Self:
        # TODO(npham): Implement this
        raise NotImplementedError

    def collect(self, key: ResultKey) -> Context:
        # Invoke the checks to collect the assertions
        context = Context(graph=graph.RootNode(name=self._name))
        for check in self._checks:
            check(self._provider, context)

        # Update the graph with symbols from the provider
        for assertion in context._graph.assertions():
            for sym in sorted(assertion.actual.free_symbols, key=lambda x: str(x), reverse=False):
                sm = self._provider.get_symbol(sym)
                assertion.add_child(symbol_node := graph.SymbolNode(name=sm.name, symbol=sm.symbol, fn=sm.fn))
                for d_spec, d_key in sm.dependencies:
                    symbol_node.add_child(
                        metric_node := graph.MetricNode(spec=d_spec, key_provider=d_key, nominal_key=key)
                    )

                    # Mark the node as provided if key is different from the nominal key
                    if d_key.create(key) != key:
                        metric_node.mark_as_provided()

                    # Only pending nodes needs analyzers
                    if metric_node.state() == "PENDING":
                        for analyzer in d_spec.analyzers:
                            metric_node.add_child(graph.AnalyzerNode(analyzer=analyzer))

        return context

    def run(self, ds: DuckDataSource, key: ResultKey, threading: bool = False) -> Context:
        # Create a context
        ctx = self.collect(key)

        # Analyze the data source
        analyzer: Analyzer = ds.analyzer_class()

        try:
            analyzer.analyze(ds, ctx.pending_metrics(), key, threading=threading)
            analyzer.persist(self._provider._db)
            ctx._graph.mark_pending_metrics_ready()
        except Exception as e:
            ctx._graph.mark_pending_metric_failed(str(e))

        # Run the checks
        ctx.validate(key)

        return ctx


def _create_check(
    provider: MetricProvider,
    context: Context,
    _check: CheckProducer,
    tags: list[str] = [],
    description: str | None = None,
) -> None:
    node = graph.CheckNode(name=_check.__name__, tags=tags, description=description)

    if context._graph.exists(node):
        raise DQXError(f"Check {node.name} already exists in the graph!")

    context._graph.add_child(node)  # This node should be the last node in the graph

    # Call the symbolic check to collect assertions for this check node
    _check(provider, context)


@overload
def check(_check: CheckProducer) -> CheckProducer: ...


@overload
def check(*, tags: list[str] = [], description: str | None = None) -> CheckCreator: ...


def check(
    _check: CheckProducer | None = None, *, tags: list[str] = [], description: str | None = None
) -> CheckProducer | CheckCreator:
    if _check is not None:
        return functools.wraps(_check)(
            functools.partial(
                _create_check,
                _check=_check,
                tags=tags,
                description=description,
            )
        )

    def decorator(fn: CheckProducer) -> CheckProducer:
        return functools.wraps(fn)(
            functools.partial(
                _create_check,
                _check=fn,
                tags=tags,
                description=description,
            )
        )

    return decorator
