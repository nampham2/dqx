import math
import sympy as sp
from dqx.common import DQXError, ResultKey
from dqx.graph.nodes import AssertionNode
from dqx.graph.base import BaseNode
from dqx.provider import MetricProvider, SymbolicMetric
from returns.result import Result, Success, Failure


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

    def _gather(self, expr: sp.Expr) -> Result[dict[sp.Symbol, float], dict[SymbolicMetric, str]]:
        """Gather metric values for all symbols in an expression.

        Extracts all free symbols from the expression and retrieves their
        corresponding values from the collected metrics. If any symbol fails
        to evaluate, returns a Failure with error messages for all failed symbols.

        Args:
            expr: Symbolic expression containing symbols to gather values for

        Returns:
            Success containing a dictionary mapping symbols to their float values if
            all symbols evaluated successfully. Failure containing a dictionary
            mapping SymbolicMetrics to error messages if any symbols failed.

        Raises:
            DQXError: If a symbol in the expression is not found in collected metrics
        """
        successes: dict[sp.Symbol, float] = {}
        failures: dict[SymbolicMetric, str] = {}

        for sym in expr.free_symbols:
            if sym not in self.metrics:
                sm = self.metric_for_symbol(sym)
                raise DQXError(f"Symbol {sm.name} not found in collected metrics.")

            match self.metrics[sym]:
                case Failure(err):
                    failures[sm] = err
                case Success(v):
                    successes[sym] = v

        if failures:
            return Failure(failures)
        return Success(successes)

    def evaluate(self, expr: sp.Expr) -> Result[float, dict[SymbolicMetric | sp.Expr, str]]:
        """Evaluate a symbolic expression by substituting collected metric values.

        First gathers all symbol values from the expression, then substitutes them
        and evaluates the result. Handles special numeric cases like NaN and infinity
        by returning appropriate failure messages.

        The evaluation process:
        1. Gather all symbol values using _gather()
        2. Substitute values into the expression
        3. Evaluate to a float with 6 decimal precision
        4. Check for NaN or infinity results

        Args:
            expr: Symbolic expression to evaluate

        Returns:
            Success containing the evaluated float value if evaluation succeeds.
            Failure containing error messages if any symbols fail to evaluate
            or if the result is NaN/infinity.
        """
        sv = self._gather(expr)
        match sv:
            case Success(symbol_values):
                expr_val = float(sp.N(expr.subs(symbol_values), 6))

                # Handling nan and inf values
                if math.isnan(expr_val):
                    return Failure({sp.Expr: "Validating value is NaN"})
                elif math.isinf(expr_val):
                    return Failure({sp.Expr: "Validating value is infinity"})

                return Success(expr_val)

            case Failure(errors):
                return Failure(errors)

        # Unreachable state
        raise RuntimeError("Unreachable state in evaluation.")

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
