"""Test integration of clone() method with provider."""

import datetime

import pyarrow as pa

from dqx import specs
from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


def test_provider_with_cloned_specs() -> None:
    """Test that cloned specs work correctly with provider in a VerificationSuite."""
    # Create test data using PyArrow
    data = pa.table(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["A", "B", "A", "B", "C"],
            "quantity": [1, 2, 3, 4, 5],
            "is_valid": [True, True, False, True, True],
        }
    )
    datasource = DuckRelationDataSource.from_arrow(data, "test_data")

    # Create original specs
    original_specs: list[specs.SimpleMetricSpec] = [
        specs.NumRows(),
        specs.Average("price"),
        specs.Sum("quantity"),
        specs.CountValues("category", ["A", "B"]),
    ]

    # Clone all specs - explicitly typed to preserve MetricSpec type
    cloned_specs: list[specs.SimpleMetricSpec] = [spec.clone() for spec in original_specs]

    # Verify clones are different instances
    for orig, clone in zip(original_specs, cloned_specs):
        assert orig is not clone
        assert orig == clone  # But logically equal

    # Create a check that uses cloned specs
    @check(name="Test with cloned specs")
    def test_cloned_specs(mp: MetricProvider, ctx: Context) -> None:
        # Use cloned specs via the provider and make assertions
        num_rows = mp.metric(cloned_specs[0])  # NumRows
        avg_price = mp.metric(cloned_specs[1])  # Average("price")
        sum_qty = mp.metric(cloned_specs[2])  # Sum("quantity")
        count_vals = mp.metric(cloned_specs[3])  # CountValues("category", ["A", "B"])

        # Make assertions to ensure metrics are computed
        ctx.assert_that(num_rows).where(name="Row count", severity="P1").is_eq(5)
        ctx.assert_that(avg_price).where(name="Average price", severity="P1").is_eq(30.0)
        ctx.assert_that(sum_qty).where(name="Sum quantity", severity="P1").is_eq(15)
        ctx.assert_that(count_vals).where(name="Count A+B", severity="P1").is_eq(4)

    # Create another check with original specs
    @check(name="Test with original specs")
    def test_original_specs(mp: MetricProvider, ctx: Context) -> None:
        # Use original specs via the provider and make assertions
        num_rows = mp.metric(original_specs[0])  # NumRows
        avg_price = mp.metric(original_specs[1])  # Average("price")
        sum_qty = mp.metric(original_specs[2])  # Sum("quantity")
        count_vals = mp.metric(original_specs[3])  # CountValues("category", ["A", "B"])

        # Make same assertions
        ctx.assert_that(num_rows).where(name="Row count orig", severity="P1").is_eq(5)
        ctx.assert_that(avg_price).where(name="Average price orig", severity="P1").is_eq(30.0)
        ctx.assert_that(sum_qty).where(name="Sum quantity orig", severity="P1").is_eq(15)
        ctx.assert_that(count_vals).where(name="Count A+B orig", severity="P1").is_eq(4)

    # Run both suites
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    # Suite with cloned specs
    suite1 = VerificationSuite([test_cloned_specs], db, "Cloned Suite")
    suite1.run([datasource], key)

    # Suite with original specs
    suite2 = VerificationSuite([test_original_specs], db, "Original Suite")
    suite2.run([datasource], key)

    # Verify both suites computed metrics correctly by checking the metric trace
    trace1 = suite1.metric_trace(db)
    trace2 = suite2.metric_trace(db)

    # Both should have computed 4 metrics
    assert trace1.num_rows == 4
    assert trace2.num_rows == 4


def test_analyzer_prefixes_are_unique() -> None:
    """Test that cloned specs have unique analyzer prefixes."""
    # Create a spec with multiple analyzers
    original = specs.Variance("values")

    # Clone it multiple times
    clone1 = original.clone()
    clone2 = original.clone()
    clone3 = clone1.clone()  # Clone of a clone

    # Collect all prefixes
    all_specs = [original, clone1, clone2, clone3]
    all_prefixes = set()

    for spec in all_specs:
        for analyzer in spec.analyzers:
            # Each prefix should be unique
            assert analyzer.prefix not in all_prefixes
            all_prefixes.add(analyzer.prefix)

    # Should have 3 analyzers per spec * 4 specs = 12 unique prefixes
    assert len(all_prefixes) == 12


def test_cloned_specs_independent_execution() -> None:
    """Test that cloned specs can be executed independently without interference."""
    # Create two different datasets
    data1 = pa.table({"value": [1, 2, 3, 4, 5]})
    data2 = pa.table({"value": [10, 20, 30, 40, 50]})

    datasource1 = DuckRelationDataSource.from_arrow(data1, "data1")
    datasource2 = DuckRelationDataSource.from_arrow(data2, "data2")

    # Create and clone a spec
    original = specs.Average("value")
    cloned = original.clone()

    # Create checks that use the specs
    @check(name="Check with original spec", datasets=["data1"])
    def check_original(mp: MetricProvider, ctx: Context) -> None:
        avg = mp.average("value")
        # Make assertion so metric gets computed
        ctx.assert_that(avg).where(name="Average value", severity="P1").is_eq(3.0)

    @check(name="Check with cloned spec", datasets=["data2"])
    def check_cloned(mp: MetricProvider, ctx: Context) -> None:
        # Use the cloned spec via provider
        avg = mp.metric(cloned)
        # Make assertion so metric gets computed
        ctx.assert_that(avg).where(name="Average value cloned", severity="P1").is_eq(30.0)

    # Run both suites
    db = InMemoryMetricDB()
    key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})

    suite1 = VerificationSuite([check_original], db, "Original Suite")
    suite1.run([datasource1], key)

    suite2 = VerificationSuite([check_cloned], db, "Cloned Suite")
    suite2.run([datasource2], key)

    # The specs should compute different values due to different data
    # We can verify this through the metric trace
    trace1 = suite1.metric_trace(db)
    trace2 = suite2.metric_trace(db)

    # Both should have computed average metrics
    assert trace1.num_rows == 1
    assert trace2.num_rows == 1

    # Extract the average values from the traces
    avg_values1 = [row["value_final"] for row in trace1.to_pylist() if "average" in row["metric"]]
    avg_values2 = [row["value_final"] for row in trace2.to_pylist() if "average" in row["metric"]]

    # Should have computed different averages
    assert len(avg_values1) == 1
    assert len(avg_values2) == 1
    assert avg_values1[0] == 3.0  # Average of [1,2,3,4,5]
    assert avg_values2[0] == 30.0  # Average of [10,20,30,40,50]
