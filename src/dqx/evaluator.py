import math
from typing import Tuple

import sympy as sp
from returns.result import Failure, Result, Success

from dqx.common import DQXError, EvaluationFailure, ResultKey, SymbolInfo
from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode
from dqx.provider import MetricProvider, SymbolicMetric


class Evaluator:
    """Evaluates symbolic expressions using collected metrics.

    The Evaluator is responsible for evaluating symbolic expressions by collecting
    metric values from a MetricProvider and substituting them into expressions. It
    implements the visitor pattern to traverse assertion nodes in the DQX graph.

    The evaluation process involves:
    1. Collecting metrics for a given ResultKey
    2. Gathering symbol values from the collected metrics
    3. Evaluating expressions with symbol substitution
    4. Handling special cases like NaN and infinity values

    Attributes:
        provider: The MetricProvider instance for accessing metric definitions
        key: The ResultKey for contextual metric evaluation
        _metrics: Dictionary mapping symbols to their computed Result values
    """

    def __init__(self, provider: MetricProvider, key: ResultKey):
        """Initialize the Evaluator with a metric provider and result key.

        Args:
            provider: MetricProvider instance containing symbolic metric definitions
            key: ResultKey specifying the context for metric evaluation (e.g., date, tags)
        """
        self.provider = provider
        self.key = key
        self._metrics: dict[sp.Basic, Result[float, str]] | None = None

    @property
    def metrics(self) -> dict[sp.Basic, Result[float, str]]:
        """Lazily collect and cache metrics for the current ResultKey.

        On first access, collects all metric values from the provider for the
        specified ResultKey and caches them. Subsequent accesses return the
        cached metrics without re-collecting.

        Returns:
            Dictionary mapping symbolic expressions to their Result values.
            Each Result is either Success[float] or Failure[str].
        """
        if self._metrics is None:
            self._metrics = self.collect_metrics(self.key)
        return self._metrics

    def collect_metrics(self, key: ResultKey) -> dict[sp.Basic, Result[float, str]]:
        """Collect all metric values from the provider for the given key.

        Iterates through all symbolic metrics in the provider and evaluates their
        functions with the provided ResultKey. Each metric evaluation returns a
        Result that either contains a successful float value or an error message.

        Args:
            key: ResultKey containing the evaluation context (date, tags, etc.)

        Returns:
            Dictionary mapping symbolic expressions to their Result values.
            Each Result is either Success[float] or Failure[str].
        """
        return {metric.symbol: metric.fn(key) for metric in self.provider.symbolic_metrics}

    def metric_for_symbol(self, symbol: sp.Symbol) -> SymbolicMetric:
        """Retrieve the SymbolicMetric associated with a given symbol.

        Args:
            symbol: The sympy Symbol to look up in the provider

        Returns:
            The SymbolicMetric containing metadata for the symbol

        Raises:
            DQXError: If the symbol is not found in the provider
        """
        return self.provider.get_symbol(symbol)

    def _gather(self, expr: sp.Expr) -> Tuple[dict[sp.Symbol, float], list[SymbolInfo]]:
        """Gather metric values and symbol information for all symbols in an expression.

        Extracts all free symbols from the expression and retrieves their
        corresponding values from the collected metrics. Always collects both
        successful values and symbol information for all symbols, regardless
        of their evaluation status.

        Args:
            expr: Symbolic expression containing symbols to gather values for

        Returns:
            Tuple containing:
            - Dictionary mapping symbols to their float values (empty if failures)
            - List of SymbolInfo objects for all symbols in the expression

        Raises:
            DQXError: If a symbol in the expression is not found in collected metrics
        """
        symbol_values: dict[sp.Symbol, float] = {}
        symbol_infos: list[SymbolInfo] = []

        # Convert to sympy expression if needed (handles boolean values)
        if not isinstance(expr, sp.Basic):
            expr = sp.sympify(expr)

        for sym in expr.free_symbols:
            if sym not in self.metrics:
                sm = self.metric_for_symbol(sym)
                raise DQXError(f"Symbol {sm.name} not found in collected metrics.")

            # Get the symbolic metric for this symbol
            sm = self.metric_for_symbol(sym)
            metric_result = self.metrics[sym]

            # Create SymbolInfo for this symbol
            symbol_info = SymbolInfo(
                name=str(sym), metric=str(sm.metric_spec), dataset=sm.dataset or "", value=metric_result
            )
            symbol_infos.append(symbol_info)

            # Collect successful values
            if isinstance(metric_result, Success):
                symbol_values[sym] = metric_result.unwrap()

        return symbol_values, symbol_infos

    def evaluate(self, expr: sp.Expr) -> Result[float, list[EvaluationFailure]]:
        """Evaluate a symbolic expression by substituting collected metric values.

        Gathers all symbol values from the expression, then substitutes them
        and evaluates the result. Handles both metric failures and special
        numeric cases (NaN/infinity) by returning EvaluationFailure objects.

        The evaluation process:
        1. Gather all symbol values and information using _gather()
        2. Check for metric failures and return early if found
        3. Substitute values into the expression
        4. Evaluate to a float with 6 decimal precision
        5. Check for NaN or infinity results

        Args:
            expr: Symbolic expression to evaluate

        Returns:
            Success containing the evaluated float value if evaluation succeeds.
            Failure containing a list of EvaluationFailure objects if any
            symbols fail to evaluate or if the result is NaN/infinity.
        """
        # Convert to sympy expression if needed (handles boolean values)
        if not isinstance(expr, sp.Basic):
            expr = sp.sympify(expr)

        # Gather symbol values and information
        symbol_values, symbol_infos = self._gather(expr)

        # Check if any symbols failed to evaluate
        failed_symbols = [si for si in symbol_infos if isinstance(si.value, Failure)]
        if failed_symbols:
            return Failure(
                [
                    EvaluationFailure(
                        error_message="One or more metrics failed to evaluate",
                        expression=str(expr),
                        symbols=symbol_infos,
                    )
                ]
            )

        # All symbols evaluated successfully, compute the expression
        try:
            # Substitute values and evaluate
            substituted = expr.subs(symbol_values)

            # Check for complex infinity (zoo) before converting to float
            if substituted == sp.zoo:
                return Failure(
                    [
                        EvaluationFailure(
                            error_message="Validating value is infinity", expression=str(expr), symbols=symbol_infos
                        )
                    ]
                )

            # Check if the result is complex (has imaginary part)
            if substituted.is_complex and not substituted.is_real:
                real_part, imag_part = substituted.as_real_imag()
                return Failure(
                    [
                        EvaluationFailure(
                            error_message=f"Validating value is complex: {float(real_part)} + {float(imag_part)}i",
                            expression=str(expr),
                            symbols=symbol_infos,
                        )
                    ]
                )

            expr_val = float(sp.N(substituted, 6))

            # Handle NaN and regular infinity
            if math.isnan(expr_val):
                return Failure(
                    [
                        EvaluationFailure(
                            error_message="Validating value is NaN", expression=str(expr), symbols=symbol_infos
                        )
                    ]
                )
            elif math.isinf(expr_val):
                return Failure(
                    [
                        EvaluationFailure(
                            error_message="Validating value is infinity", expression=str(expr), symbols=symbol_infos
                        )
                    ]
                )

            return Success(expr_val)

        except Exception as e:
            # Handle any unexpected errors during evaluation
            return Failure(
                [
                    EvaluationFailure(
                        error_message=f"Error evaluating expression: {str(e)}",
                        expression=str(expr),
                        symbols=symbol_infos,
                    )
                ]
            )

    def visit(self, node: BaseNode) -> None:
        """Visit a node in the DQX graph and evaluate assertions.

        Implements the visitor pattern for traversing the DQX computation graph.
        When visiting an AssertionNode, evaluates its actual expression and stores
        the result in the node's _value attribute. Other node types are passed
        through without modification.

        This method is synchronous and is called during graph traversal to compute
        assertion values that will later be compared against expected values.

        Args:
            node: The graph node to visit. If it's an AssertionNode, its actual
                  expression will be evaluated.
        """
        if isinstance(node, AssertionNode):
            node._value = self.evaluate(node.actual)

    async def visit_async(self, node: BaseNode) -> None:
        """Asynchronously visit a node in the DQX graph.

        A wrapper around the synchronous visit method to support async graph
        traversal. Currently delegates directly to the synchronous visit() method
        as metric evaluation is synchronous. This allows the Evaluator to be used
        in both synchronous and asynchronous graph traversal contexts.

        Args:
            node: The graph node to visit asynchronously
        """
        self.visit(node)
