import logging
import math
from typing import Tuple

import sympy as sp
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success

from dqx.common import DQXError, EvaluationFailure, ResultKey
from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode
from dqx.provider import MetricProvider, SymbolicMetric, SymbolInfo

logger = logging.getLogger(__name__)


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

    def __init__(self, provider: MetricProvider, key: ResultKey, suite_name: str, data_av_threshold: float):
        """Initialize the Evaluator with a metric provider and result key.

        Args:
            provider: MetricProvider instance containing symbolic metric definitions
            key: ResultKey specifying the context for metric evaluation (e.g., date, tags)
            suite_name: Name of the verification suite for context tracking
            data_av_threshold: Minimum data availability ratio to evaluate assertions
        """
        self.provider = provider
        self._key = key
        self._suite_name = suite_name
        self._data_av_threshold = data_av_threshold
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
            self._metrics = self.collect_metrics(self._key)
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
        result: dict[sp.Basic, Result[float, str]] = {}
        for sm in self.provider.metrics:
            result[sm.symbol] = sm.fn(key)

        return result

        # return {metric.symbol: metric.fn(key) for metric in self.provider.metrics}

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
            # Use the name from SymbolicMetric which includes the proper function name
            # For extended metrics: "day_over_day(maximum(tax))"
            # For regular metrics: "maximum(tax)"
            # For lag metrics: "lag(1)(x_1)"
            symbol_info = SymbolInfo(
                name=str(sym),
                metric=sm.name,
                dataset=sm.dataset,
                value=metric_result,
                yyyy_mm_dd=self._key.yyyy_mm_dd,
                tags=self._key.tags,
            )
            symbol_infos.append(symbol_info)

            # Collect successful values
            match metric_result:
                case Success(value):
                    symbol_values[sym] = value
                case _:
                    pass

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
        # Gather symbol values and information
        symbol_values, symbol_infos = self._gather(expr)

        # Check if any symbols failed to evaluate
        failed_symbols = [si for si in symbol_infos if not is_successful(si.value)]
        if failed_symbols:
            # Generate specific error message based on failure types
            return Failure(
                [
                    EvaluationFailure(
                        error_message=f"Failed to evaluate symbol(s): {', '.join(si.name for si in failed_symbols)} ",
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

    def _check_data_availability(self, expr: sp.Expr) -> bool:
        """Check if all metrics in the expression meet the data availability threshold.

        Args:
            expr: Symbolic expression containing metrics to check

        Returns:
            True if all metrics meet the threshold, False otherwise
        """
        # Convert to sympy expression if needed
        if not isinstance(expr, sp.Basic):
            expr = sp.sympify(expr)

        # Check each symbol in the expression
        for sym in expr.free_symbols:
            if sym in self.metrics:
                # Get the symbolic metric
                sm = self.metric_for_symbol(sym)

                # If ratio is below threshold, return False
                if sm.data_av_ratio < self._data_av_threshold:
                    return False

        return True

    def visit(self, node: BaseNode) -> None:
        """Visit a node in the DQX graph and evaluate assertions.

        For AssertionNodes:
        1. Checks data availability for all metrics in the expression
        2. Skips evaluation if any metric is below the threshold
        3. Evaluates the metric expression if data is available
        4. Applies the validator function if metric succeeds
        5. Stores both metric result and validation status
        """
        if isinstance(node, AssertionNode):
            # Check data availability first
            if not self._check_data_availability(node.actual):
                # Skip this assertion due to insufficient data
                node._result = "SKIPPED"
                node._metric = Failure(
                    [
                        EvaluationFailure(
                            error_message="Insufficient data availability",
                            expression=str(node.actual),
                            symbols=[],
                        )
                    ]
                )
                return

            # Evaluate the metric
            node._metric = self.evaluate(node.actual)

            # Apply validator to determine pass/fail
            match node._metric:
                case Success(value):
                    try:
                        # validator.fn returns True if assertion passes
                        passed = node.validator.fn(value)
                        node._result = "PASSED" if passed else "FAILED"
                    except Exception as e:
                        raise DQXError(f"Validator execution failed: {str(e)}") from e
                case Failure(_):
                    # If metric computation failed, assertion fails
                    node._result = "FAILED"

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

    def collect_symbols(self, expr: sp.Expr) -> list[SymbolInfo]:
        """Collect symbol information for all symbols in an expression.

        This method gathers SymbolInfo objects for all symbols referenced
        in the given expression without evaluating the expression itself.
        It's useful for introspection and debugging purposes.

        Args:
            expr: Symbolic expression containing symbols to collect

        Returns:
            List of SymbolInfo objects with complete context information
            including suite name, date, and tags from the evaluator's context

        Example:
            evaluator = Evaluator(provider, key, "My Suite")
            expr = x_1 + x_2 * 2
            symbols = evaluator.collect_symbols(expr)
            # Returns [SymbolInfo(...), SymbolInfo(...)]
        """
        _, symbol_infos = self._gather(expr)
        return symbol_infos
