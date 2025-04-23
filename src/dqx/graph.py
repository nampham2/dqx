from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

import sympy as sp
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success
from rich.tree import Tree

from dqx.common import ResultKey, ResultKeyProvider, RetrievalFn, SeverityLevel, SymbolicValidator
from dqx.ops import Op
from dqx.specs import MetricSpec

NodeType = Literal["Root", "Check", "Assert", "Symbol", "Metric", "Analyzer"]
MetricState = Literal["READY", "PROVIDED", "PENDING", "ERROR"]

T = TypeVar("T", bound="Node")


@runtime_checkable
class Node(Protocol, Generic[T]):
    children: list[T]

    def add_child(self, child: T) -> None: ...

    def inspect_str(self) -> str: ...


class NodeMixin(Node[T]):
    def add_child(self, child: T) -> None:
        self.children.append(child)


class RootNode(NodeMixin["CheckNode"], Node["CheckNode"]):
    def __init__(self, name: str) -> None:
        self.name = name
        self.children: list[CheckNode] = []

    def inspect(self) -> Tree:
        """
        Inspects the current node and its children, constructing a tree representation.
        This method performs a breadth-first traversal of the node and its children,
        creating a `Tree` object that represents the structure of the nodes.

        Returns:
            Tree: A `Tree` object representing the structure of the node and its children.
        """
        root = Tree(self.inspect_str())

        queue: deque = deque(zip(self.children[::-1], [root] * len(self.children)))
        while queue:
            node, tree = queue.pop()
            # if hasattr(node, "_value"):
            #     match node._value:
            #         case Some(Failure(_)):
            #             subtree = tree.add(node.inspect_str() + "\n")
            #         case _:
            #             subtree = tree.add(node.inspect_str())
            #             queue.extend(zip(reversed(node.children), [subtree] * len(node.children)))
            # else:
            subtree = tree.add(node.inspect_str())
            queue.extend(zip(reversed(node.children), [subtree] * len(node.children)))

        return fix_tree(root)

    def inspect_str(self) -> str:
        return f"Suite: {self.name}"

    def exists(self, child: "CheckNode") -> bool:
        return child in self.children

    def assertions(self) -> Iterator[AssertionNode]:
        queue: deque = deque([self])

        while queue:
            node = queue.pop()
            if isinstance(node, AssertionNode):
                yield node
            else:
                queue.extend(node.children)

    def checks(self) -> Iterator[CheckNode]:
        queue: deque = deque([self])

        while queue:
            node = queue.pop()
            if isinstance(node, CheckNode):
                yield node
            else:
                queue.extend(node.children)

    def metrics(self) -> Iterator[MetricNode]:
        queue: deque = deque([self])

        while queue:
            node = queue.pop()
            if isinstance(node, MetricNode):
                yield node
            else:
                queue.extend(node.children)

    def symbols(self) -> Iterator[SymbolNode]:
        queue: deque = deque([self])

        while queue:
            node = queue.pop()
            if isinstance(node, SymbolNode):
                yield node
            else:
                queue.extend(node.children)

    def ready_metrics(self) -> Iterator[MetricNode]:
        return filter(lambda node: node.state() == "READY", self.metrics())

    def provided_metrics(self) -> Iterator[MetricNode]:
        return filter(lambda node: node.state() == "PROVIDED", self.metrics())

    def pending_metrics(self, dataset: str) -> Iterator[MetricNode]:
        for node in self.checks():
            for metric in node.children:
                for symbol in metric.children:
                    if symbol.datasets == [dataset]:
                        for analyzer in symbol.children:
                            if analyzer.state() == "PENDING":
                                yield analyzer

    def ready_symbols(self) -> Iterator[SymbolNode]:
        return filter(lambda node: node.ready(), self.symbols())

    def mark_pending_metrics_success(self, dataset: str) -> None:
        for node in self.pending_metrics(dataset):
            node.mark_as_success()

    def mark_pending_metric_failed(self, dataset: str, message: str) -> None:
        for node in self.pending_metrics(dataset):
            node.mark_as_failure(message)

    def propagate(self, ds: list[str]) -> None:
        """Dataset propagation to the immediate children.
        1. Impute check's datasets with the provided datasets
        2. Propagate datasets from checks to symbols
        2.1. Resolve the individual metric constrains and mark the metrics as failed or put the datasets in the metric
        3. Back propagate the failed metrics to symbols and assertions
        """
        for node in self.children:
            node.impute_dataset(ds)
            node.propagate()


class CheckNode(NodeMixin["AssertionNode"], Node["AssertionNode"]):
    def __init__(
        self,
        name: str,
        tags: list[str] | None = None,
        label: str | None = None,
        datasets: list[str] | None = None,
    ) -> None:
        self.name = name
        self.tags = tags or []
        self.label = label
        self.children = []
        self.datasets: list[str] = datasets or []
        self._value: Maybe[Result[float, str]] = Nothing

    def inspect_str(self) -> str:
        return f"{self.label or self.name} {self._value} {self.datasets}"

    def node_name(self) -> str:
        return self.label or self.name

    def impute_dataset(self, datasets: list[str]) -> None:
        if any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(f"The check {self.node_name()} requires datasets {self.datasets} but got {datasets}")
            )
        elif len(self.datasets) == 0:
            self.datasets = datasets

    def propagate(self) -> None:
        for node in self.children:
            node.impute_dataset(self.datasets)
            node.propagate()


class AssertionNode(NodeMixin["SymbolNode"], Node["SymbolNode"]):
    def __init__(
        self,
        actual: sp.Expr,
        label: str | None = None,
        severity: SeverityLevel | None = None,
        validator: SymbolicValidator | None = None,
    ) -> None:
        self.actual = actual
        self.label: str | None = label
        self.severity: SeverityLevel | None = severity
        self.datasets: list[str] = []
        self.validator: SymbolicValidator | None = validator
        self._value: Maybe[Result[float, str]] = Nothing

        self.children = []

    def set_label(self, label: str) -> None:
        self.label = label

    def set_severity(self, severity: SeverityLevel) -> None:
        self.severity = severity

    def set_validator(self, validator: SymbolicValidator) -> None:
        self.validator = validator

    def set_datasource(self, datasets: list[str]) -> None:
        self.datasets = datasets

    def impute_dataset(self, datasets: list[str]) -> None:
        if any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(f"The assertion {str(self.actual) or self.label} requires datasets {self.datasets} but got {datasets}")
            )
        elif len(self.datasets) == 0:
            self.datasets = datasets

    def propagate(self) -> None:
        for node in self.children:
            node.impute_dataset(self.datasets)
            node.propagate()

    def mark_as_failure(self, message: str) -> None:
        self._value = Some(Failure(message))

    def evaluate(self) -> Result[Any, str]:
        if all(child.success() for child in self.children):
            symbol_table = {child.symbol: child._value.unwrap().unwrap() for child in self.children}
            try:
                value = sp.N(self.actual.subs(symbol_table), 6)
                if math.isnan(value):
                    self._value = Some(Failure("Validating value is NaN"))
                else:
                    self._value = Some(Success(value))
            except Exception as e:
                self._value = Some(Failure(str(e)))
        else:
            # TODO(npham): Better failure message
            self._value = Some(Failure("Dependencies failure"))

        return self._value.unwrap()

    def inspect_str(self) -> str:
        if self.validator:
            return f"Assert that [green]{self.actual} {self.validator.name}[/green] :right_arrow: {self._value} {self.datasets}"
        return f"{self.actual}"


class SymbolNode(NodeMixin["MetricNode"], Node["MetricNode"]):
    def __init__(self, name: str, symbol: sp.Symbol, fn: RetrievalFn, datasets: list[str]) -> None:
        self.name = name
        self.symbol = symbol
        self.fn = fn
        self.datasets: list[str] = datasets
        self.children = []
        self._value: Maybe[Result[float, str]] = Nothing
        self._required_ds_count: int = 1

    def inspect_str(self) -> str:
        return f"[yellow2]{str(self.symbol)}[/yellow2]: [chartreuse3]{self.name}[/chartreuse3] :right_arrow: {self._value} {self.datasets}"

    def mark_as_failure(self, message: str) -> None:
        self._value = Some(Failure(message))

    def ready(self) -> bool:
        return all(child.state() == "PROVIDED" for child in self.children)

    def success(self) -> bool:
        match self._value:
            case Some(Success(_)):
                return True
            case _:
                return False

    def failure(self) -> bool:
        match self._value:
            case Some(Failure(_)):
                return True
            case _:
                return False

    def impute_dataset(self, datasets: list[str]) -> None:
        if len(self.datasets) == self._required_ds_count and all(ds in datasets for ds in self.datasets):
            return
            
        if any(ds not in datasets for ds in self.datasets):
            self._value = Some(
                Failure(f"The symbol {str(self.symbol)} requires datasets {self.datasets} but got {datasets}")
            )
        elif len(self.datasets) == 0 and len(datasets) != self._required_ds_count:
            self._value = Some(
                Failure(
                    f"The symbol {str(self.symbol)} requires exactly {self._required_ds_count} datasets but got {datasets}"
                )
            )
        else:
            self.datasets = datasets

    def propagate(self) -> None:
        for node in self.children:
            node.impute_dataset(self.datasets)

    def evaluate(self, key: ResultKey) -> Result[float, str]:
        self._value = Some(self.fn(key))
        return self._value.unwrap()


class MetricNode(NodeMixin["AnalyzerNode"], Node["AnalyzerNode"]):
    """Metric states:
    - PENDING: The metric is pending computed
    - PROVIDED: The metric is computed and stored in DB
    - ERROR: The metric computation failed. The error message is stored in _value
    """

    def __init__(self, spec: MetricSpec, key_provider: ResultKeyProvider, nominal_key: ResultKey) -> None:
        self.spec = spec
        self.key_provider = key_provider
        self._nominal_key = nominal_key
        self.datasets: list[str] = []
        self.children = []

        # Nothing -> Not analyzed, PENDING
        # Some(Success) -> Analyzed successfully PROVIDED
        # Some(Failure) -> Analyzed with failure ERROR
        self._analyzed: Maybe[Result[None, str]] = Nothing

    def inspect_str(self) -> str:
        return f"{self.spec.name} with {self.eval_key()} :right_arrow: {self._analyzed} {self.datasets}"

    def eval_key(self) -> ResultKey:
        return self.key_provider.create(self._nominal_key)

    def mark_as_provided(self) -> None:
        self._analyzed = Some(Success(None))

    def mark_as_success(self) -> None:
        self._analyzed = Some(Success(None))

    def mark_as_failure(self, message: str) -> None:
        self._analyzed = Some(Failure(message))

    def impute_dataset(self, datasets: list[str]) -> None:
        self.datasets = datasets

    def state(self) -> MetricState:
        match self._analyzed:
            case Some(Failure(_)):
                return "ERROR"
            case Some(Success(_)):
                return "PROVIDED"
            case _:
                return "PENDING"


class AnalyzerNode(NodeMixin, Node):
    def __init__(self, analyzer: Op) -> None:
        self.analyzer = analyzer
        self.children = []

    def add_child(self, child: Any) -> None:
        raise NotImplementedError("AnalyzerNode cannot have children")

    def inspect_str(self) -> str:
        return f"Analyze {self.analyzer.name}"


def fix_tree(tree: Tree, depth: int = 0) -> Tree:
    """
    Fix the tree display, adding newlines to the last label of metric nodes.

    Args:
        tree (Tree): The root of the tree to be fixed.
        depth (int, optional): The current depth of the traversal. Defaults to 0.

    Returns:
        The original tree
    """
    if depth == 4 and len(tree.children) == 0:
        tree.label = str(tree.label) + "\n"
        return tree
    elif depth == 4:
        tree.children[-1].label = str(tree.children[-1].label) + "\n"
        return tree

    for child in tree.children:
        fix_tree(child, depth + 1)

    return tree
