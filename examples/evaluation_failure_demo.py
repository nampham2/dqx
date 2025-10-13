"""Demo of the improved evaluation failure handling in DQX.

This script demonstrates how the new EvaluationFailure and SymbolInfo
dataclasses provide structured and detailed error reporting when metrics
fail to evaluate or produce invalid results.
"""

from datetime import date
from unittest.mock import Mock

import sympy as sp
from returns.result import Failure, Success

from dqx.common import EvaluationFailure, ResultKey
from dqx.evaluator import Evaluator
from dqx.provider import MetricProvider


def print_failure_details(failures: list[EvaluationFailure]) -> None:
    """Pretty print evaluation failure details."""
    for i, failure in enumerate(failures, 1):
        print(f"\n🚨 Failure {i}:")
        print(f"   Error: {failure.error_message}")
        print(f"   Expression: {failure.expression}")

        if failure.symbols:
            print(f"   Symbols involved ({len(failure.symbols)}):")
            for symbol in failure.symbols:
                status = "✅ Success" if isinstance(symbol.value, Success) else "❌ Failed"
                print(f"      - {symbol.name}: {symbol.metric} from '{symbol.dataset}' [{status}]")
                if isinstance(symbol.value, Failure):
                    print(f"        Error: {symbol.value.failure()}")
                else:
                    print(f"        Value: {symbol.value.unwrap()}")


def demo_metric_failure() -> None:
    """Demo 1: Metric failure propagation."""
    print("\n" + "=" * 60)
    print("DEMO 1: Metric Failure Propagation")
    print("=" * 60)

    # Setup provider with mock database
    provider = MetricProvider(db=Mock())
    key = ResultKey(yyyy_mm_dd=date.today(), tags={"env": "prod"})

    # Create successful and failing metrics
    revenue = provider.sum("revenue", dataset="sales")
    provider._symbol_index[revenue].fn = lambda k: Success(10000.0)

    costs = provider.sum("costs", dataset="expenses")
    provider._symbol_index[costs].fn = lambda k: Failure("Database connection timeout after 30s")

    users = provider.average("active_users", dataset="users")
    provider._symbol_index[users].fn = lambda k: Success(100.0)

    # Create evaluator and evaluate expression
    evaluator = Evaluator(provider, key, "Test Suite")

    # Try to calculate profit per user: (revenue - costs) / users
    expr = (revenue - costs) / users
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print("\nExpression: (revenue - costs) / users")
        print("Result: FAILED")
        print_failure_details(result.failure())
    else:
        print(f"Result: {result.unwrap()}")


def demo_nan_handling() -> None:
    """Demo 2: NaN handling."""
    print("\n" + "=" * 60)
    print("DEMO 2: NaN/Infinity Handling")
    print("=" * 60)

    provider = MetricProvider(db=Mock())
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})

    # Create metrics that will produce infinity
    total_sales = provider.sum("amount", dataset="sales")
    provider._symbol_index[total_sales].fn = lambda k: Success(1000.0)

    zero_divisor = provider.sum("cancelled_orders", dataset="sales")
    provider._symbol_index[zero_divisor].fn = lambda k: Success(0.0)

    evaluator = Evaluator(provider, key, "Test Suite")

    # Calculate metric that produces infinity: total_sales / zero_divisor = 1000/0 = infinity
    expr = total_sales / zero_divisor
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print("\nExpression: total_sales / zero_divisor")
        print("Result: FAILED")
        print_failure_details(result.failure())


def demo_complex_expression() -> None:
    """Demo 3: Complex expression with multiple failures."""
    print("\n" + "=" * 60)
    print("DEMO 3: Complex Expression with Multiple Data Sources")
    print("=" * 60)

    provider = MetricProvider(db=Mock())
    key = ResultKey(yyyy_mm_dd=date.today(), tags={"region": "US"})

    # Metrics from different datasets
    orders_total = provider.sum("total", dataset="orders")
    provider._symbol_index[orders_total].fn = lambda k: Success(50000.0)

    shipping_cost = provider.average("cost", dataset="shipping")
    provider._symbol_index[shipping_cost].fn = lambda k: Failure("Table 'shipping' not found")

    returns_count = provider.num_rows(dataset="returns")
    provider._symbol_index[returns_count].fn = lambda k: Success(100.0)

    customer_count = provider.approx_cardinality("customer_id", dataset="customers")
    provider._symbol_index[customer_count].fn = lambda k: Failure("Permission denied on table 'customers'")

    evaluator = Evaluator(provider, key, "Test Suite")

    # Complex business metric: (orders - shipping*returns) / customers
    expr = (orders_total - shipping_cost * returns_count) / customer_count
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print("\nExpression: (orders_total - shipping_cost * returns_count) / customer_count")
        print("Result: FAILED")
        print_failure_details(result.failure())


def demo_boolean_expression_failure() -> None:
    """Demo 4: Boolean expressions should fail."""
    print("\n" + "=" * 60)
    print("DEMO 4: Boolean Expression Failure")
    print("=" * 60)

    provider = MetricProvider(db=Mock())
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})

    # Create metrics for comparison
    current_revenue = provider.sum("revenue", dataset="sales_current")
    provider._symbol_index[current_revenue].fn = lambda k: Success(15000.0)

    target_revenue = provider.sum("target", dataset="sales_targets")
    provider._symbol_index[target_revenue].fn = lambda k: Success(12000.0)

    evaluator = Evaluator(provider, key, "Test Suite")

    # Boolean comparison: current_revenue > target_revenue
    # This should fail because metrics must evaluate to numeric values
    expr = current_revenue > target_revenue
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print("\nExpression: current_revenue > target_revenue")
        print("Result: FAILED (as expected)")
        print_failure_details(result.failure())
        print("\n💡 Note: Metrics must evaluate to numeric values.")
        print("   Use validators to perform comparisons on metric values.")


def demo_complex_number_handling() -> None:
    """Demo 5: Complex number handling."""
    print("\n" + "=" * 60)
    print("DEMO 5: Complex Number Handling")
    print("=" * 60)

    provider = MetricProvider(db=Mock())
    key = ResultKey(yyyy_mm_dd=date.today(), tags={})

    # Create metrics that will produce complex numbers
    negative_returns = provider.sum("return_amount", dataset="returns")
    provider._symbol_index[negative_returns].fn = lambda k: Success(-100.0)

    negative_balance = provider.sum("balance", dataset="accounts")
    provider._symbol_index[negative_balance].fn = lambda k: Success(-50.0)

    positive_multiplier = provider.sum("multiplier", dataset="config")
    provider._symbol_index[positive_multiplier].fn = lambda k: Success(2.0)

    evaluator = Evaluator(provider, key, "Test Suite")

    # Example 1: Square root of negative returns
    print("\n📊 Example 1: Square root of negative value")
    expr = sp.sqrt(negative_returns)
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print(f"\nExpression: sqrt({-100.0})")
        print("Result: FAILED")
        print_failure_details(result.failure())

    # Example 2: Logarithm of negative balance
    print("\n\n📊 Example 2: Logarithm of negative value")
    expr = sp.log(negative_balance)
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print(f"\nExpression: log({-50.0})")
        print("Result: FAILED")
        print_failure_details(result.failure())

    # Example 3: Complex arithmetic
    print("\n\n📊 Example 3: Complex arithmetic expression")
    expr = sp.sqrt(negative_returns) * positive_multiplier
    result = evaluator.evaluate(expr)

    if isinstance(result, Failure):
        print(f"\nExpression: sqrt({-100.0}) * {2.0}")
        print("Result: FAILED")
        print_failure_details(result.failure())
        print("\n💡 Note: Complex numbers cannot be used as metric values.")
        print("   Ensure all metric calculations produce real numbers.")


if __name__ == "__main__":
    print("🔍 DQX Evaluation Failure Handling Demo")
    print("This demo showcases the improved error reporting with EvaluationFailure\n")

    demo_metric_failure()
    demo_nan_handling()
    demo_complex_expression()
    demo_boolean_expression_failure()
    demo_complex_number_handling()

    print("\n" + "=" * 60)
    print("✨ Key Features Demonstrated:")
    print("- Structured error reporting with EvaluationFailure dataclass")
    print("- Detailed symbol information including dataset and metric specs")
    print("- Graceful handling of NaN and infinity values")
    print("- Boolean expressions properly fail (metrics must be numeric)")
    print("- Complex number detection and proper error reporting")
    print("- Clear indication of which specific metrics failed")
    print("=" * 60)
