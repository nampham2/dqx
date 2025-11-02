"""Tests for nested extended metric specifications."""

from dqx import specs


class TestNestedExtendedMetrics:
    """Test extended metrics with nested base metrics."""

    def test_nested_dayoverday_hash(self) -> None:
        """Test that DayOverDay of DayOverDay can be hashed."""
        # Create base metric
        avg = specs.Average("price")

        # Create first level DoD
        dod1 = specs.DayOverDay.from_base_spec(avg)

        # Create nested DoD - this should not raise TypeError
        dod2 = specs.DayOverDay.from_base_spec(dod1)

        # Should be able to hash without error
        hash_value = hash(dod2)
        assert isinstance(hash_value, int)

    def test_stddev_of_dayoverday_hash(self) -> None:
        """Test that Stddev of DayOverDay can be hashed."""
        avg = specs.Average("tax")
        dod = specs.DayOverDay.from_base_spec(avg)
        stddev = specs.Stddev.from_base_spec(dod, offset=1, n=7)

        # This is the exact case from the failing test
        hash_value = hash(stddev)
        assert isinstance(hash_value, int)

    def test_deeply_nested_metrics(self) -> None:
        """Test 3+ levels of nesting."""
        # stddev(dod(wow(sum(revenue))))
        sum_spec = specs.Sum("revenue")
        wow = specs.WeekOverWeek.from_base_spec(sum_spec)
        dod = specs.DayOverDay.from_base_spec(wow)
        stddev = specs.Stddev.from_base_spec(dod, offset=1, n=14)

        # All operations should work
        assert hash(stddev) is not None
        assert stddev.name == "stddev(dod(wow(sum(revenue))), offset=1, n=14)"

    def test_nested_weekoverweek_hash(self) -> None:
        """Test that WeekOverWeek of extended metrics can be hashed."""
        min_spec = specs.Minimum("cost")
        dod = specs.DayOverDay.from_base_spec(min_spec)
        wow = specs.WeekOverWeek.from_base_spec(dod)

        hash_value = hash(wow)
        assert isinstance(hash_value, int)
        assert wow.name == "wow(dod(minimum(cost)))"

    def test_nested_metric_database_roundtrip(self) -> None:
        """Test that nested metrics can be stored and retrieved."""
        import datetime as dt

        from dqx import models
        from dqx.common import ResultKey
        from dqx.orm.repositories import InMemoryMetricDB

        # Create nested metric
        avg = specs.Average("tax")
        dod = specs.DayOverDay.from_base_spec(avg)
        stddev = specs.Stddev.from_base_spec(dod, offset=1, n=7)

        # Store in database
        db = InMemoryMetricDB()
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 10, 24), tags={})
        metric = models.Metric.build(stddev, key, dataset="test_dataset", state=stddev.state())

        persisted = list(db.persist([metric]))[0]

        # Retrieve and verify
        assert persisted.metric_id is not None

        # Since the get() method is no longer available, verify the persisted metric directly
        # The persist() method returns the model with all fields populated
        assert persisted.spec == stddev
        assert persisted.spec.name == "stddev(dod(average(tax)), offset=1, n=7)"
        assert persisted.key == key
        assert persisted.dataset == "test_dataset"

        # Verify it exists in the database
        assert db.exists(persisted.metric_id)
