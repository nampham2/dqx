"""Demonstrate the performance improvement in extended metrics.

This example shows how the refactored extended metrics now:
1. Use get_symbol() only once instead of twice per method
2. Properly set parent_symbol on base metrics
"""

from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def main() -> None:
    """Run the extended metric performance demo."""
    # Create metric provider directly
    db = InMemoryMetricDB()
    provider = MetricProvider(db)

    # Create a base metric
    revenue = provider.sum("revenue")
    print(f"Created base metric: {revenue}")

    # Check initial parent (should be None)
    revenue_info = provider.get_symbol(revenue)
    print(f"Initial parent_symbol: {revenue_info.parent_symbol}")

    # Create day_over_day extended metric
    # Previously: would call get_symbol() twice internally
    # Now: calls get_symbol() only once
    dod = provider.ext.day_over_day(revenue)
    print(f"\nCreated day_over_day metric: {dod}")

    # Check that base metric now has parent set
    revenue_info_after = provider.get_symbol(revenue)
    print(f"Base metric parent_symbol after extended metric: {revenue_info_after.parent_symbol}")
    print(f"Parent matches day_over_day: {revenue_info_after.parent_symbol == dod}")

    # Check the extended metric has no parent
    dod_info = provider.get_symbol(dod)
    print(f"\nday_over_day parent_symbol: {dod_info.parent_symbol}")

    # Show children tracking
    children = provider.get_children(dod)
    print(f"\nChildren of day_over_day: {children}")
    print(f"Base metric is a child: {revenue in children}")

    # Create week_over_week on a different metric
    print("\n" + "=" * 50 + "\n")

    price = provider.average("price")
    wow = provider.ext.week_over_week(price)

    price_info = provider.get_symbol(price)
    print(f"Price metric parent_symbol: {price_info.parent_symbol}")
    print(f"Parent is week_over_week: {price_info.parent_symbol == wow}")

    # Create stddev with multiple lag dependencies
    print("\n" + "=" * 50 + "\n")

    score = provider.variance("score")
    stddev = provider.ext.stddev(score, lag=1, n=7)

    score_info = provider.get_symbol(score)
    print(f"Score metric parent_symbol: {score_info.parent_symbol}")
    print(f"Parent is stddev: {score_info.parent_symbol == stddev}")

    # Show all children of stddev (base + lag dependencies)
    stddev_children = provider.get_children(stddev)
    print(f"\nStddev has {len(stddev_children)} children (base + 7 lag dependencies)")

    # Performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE IMPROVEMENT SUMMARY:")
    print("=" * 50)
    print("Before: Each extended metric method called get_symbol() twice")
    print("After: Each extended metric method calls get_symbol() only once")
    print("\nAdditional benefits:")
    print("- Base metrics now track their parent extended metrics")
    print("- Cleaner code with better variable naming (symbolic_metric)")
    print("- Removed redundant _resolve_metric_spec helper method")


if __name__ == "__main__":
    main()
