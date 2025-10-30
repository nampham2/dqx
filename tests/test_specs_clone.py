"""Test clone() method for metric specs."""

import pytest

from dqx import specs


class TestSpecsClone:
    """Test clone() method for all basic metric specs."""

    def test_num_rows_clone(self) -> None:
        """Test NumRows.clone() creates independent instance with new analyzer prefixes."""
        # Create original
        original = specs.NumRows()

        # Clone it
        cloned = original.clone()

        # Verify they are different instances
        assert original is not cloned
        assert isinstance(cloned, specs.NumRows)

        # Verify same parameters
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        # Verify analyzers are different instances with different prefixes
        assert len(original.analyzers) == len(cloned.analyzers)
        for orig_analyzer, clone_analyzer in zip(original.analyzers, cloned.analyzers):
            assert orig_analyzer is not clone_analyzer
            assert orig_analyzer.prefix != clone_analyzer.prefix
            assert orig_analyzer.name == clone_analyzer.name

    def test_first_clone(self) -> None:
        """Test First.clone() creates independent instance with new analyzer prefixes."""
        original = specs.First("test_column")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.First)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        # Verify analyzer independence
        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

    def test_average_clone(self) -> None:
        """Test Average.clone() creates independent instance with new analyzer prefixes."""
        original = specs.Average("price")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.Average)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        # Average has multiple analyzers (NumRows, Average)
        assert len(original.analyzers) == 2
        assert len(cloned.analyzers) == 2

        # Check all analyzers have different prefixes
        for orig_analyzer, clone_analyzer in zip(original.analyzers, cloned.analyzers):
            assert orig_analyzer is not clone_analyzer
            assert orig_analyzer.prefix != clone_analyzer.prefix

    def test_variance_clone(self) -> None:
        """Test Variance.clone() creates independent instance with new analyzer prefixes."""
        original = specs.Variance("values")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.Variance)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        # Variance has 3 analyzers (NumRows, Average, Variance)
        assert len(original.analyzers) == 3
        assert len(cloned.analyzers) == 3

        # Check all analyzers have different prefixes
        for orig_analyzer, clone_analyzer in zip(original.analyzers, cloned.analyzers):
            assert orig_analyzer is not clone_analyzer
            assert orig_analyzer.prefix != clone_analyzer.prefix

    def test_minimum_clone(self) -> None:
        """Test Minimum.clone() creates independent instance with new analyzer prefixes."""
        original = specs.Minimum("score")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.Minimum)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

    def test_maximum_clone(self) -> None:
        """Test Maximum.clone() creates independent instance with new analyzer prefixes."""
        original = specs.Maximum("score")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.Maximum)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

    def test_sum_clone(self) -> None:
        """Test Sum.clone() creates independent instance with new analyzer prefixes."""
        original = specs.Sum("amount")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.Sum)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

    def test_null_count_clone(self) -> None:
        """Test NullCount.clone() creates independent instance with new analyzer prefixes."""
        original = specs.NullCount("email")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.NullCount)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

    def test_negative_count_clone(self) -> None:
        """Test NegativeCount.clone() creates independent instance with new analyzer prefixes."""
        original = specs.NegativeCount("balance")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.NegativeCount)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

    def test_duplicate_count_clone(self) -> None:
        """Test DuplicateCount.clone() creates independent instance with new analyzer prefixes."""
        # Single column
        original = specs.DuplicateCount(["email"])
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.DuplicateCount)
        assert original._columns == cloned._columns
        assert original._columns is not cloned._columns  # List should be copied
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

        # Multiple columns
        original2 = specs.DuplicateCount(["user_id", "email"])
        cloned2 = original2.clone()

        assert original2 is not cloned2
        assert original2._columns == cloned2._columns
        assert original2._columns is not cloned2._columns  # List should be copied

    def test_count_values_clone_single_value(self) -> None:
        """Test CountValues.clone() with single values."""
        # Integer value
        original = specs.CountValues("status", 1)
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.CountValues)
        assert original._column == cloned._column
        assert original._values == cloned._values
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

        # String value
        original2 = specs.CountValues("status", "active")
        cloned2 = original2.clone()
        assert original2 is not cloned2
        assert original2._values == cloned2._values

        # Boolean value
        original3 = specs.CountValues("is_valid", True)
        cloned3 = original3.clone()
        assert original3 is not cloned3
        assert original3._values is cloned3._values  # True is singleton

    def test_count_values_clone_list_values(self) -> None:
        """Test CountValues.clone() with list values."""
        # Integer list
        original = specs.CountValues("type_id", [1, 2, 3])
        cloned = original.clone()

        assert original is not cloned
        assert original._values == cloned._values
        assert original._values is not cloned._values  # List should be copied

        # String list
        original2 = specs.CountValues("status", ["pending", "active"])
        cloned2 = original2.clone()
        assert original2 is not cloned2
        assert original2._values == cloned2._values
        assert original2._values is not cloned2._values  # List should be copied

    def test_unique_count_clone(self) -> None:
        """Test UniqueCount.clone() creates independent instance with new analyzer prefixes."""
        original = specs.UniqueCount("product_id")
        cloned = original.clone()

        assert original is not cloned
        assert isinstance(cloned, specs.UniqueCount)
        assert original._column == cloned._column
        assert original.parameters == cloned.parameters
        assert original.name == cloned.name

        assert original.analyzers[0] is not cloned.analyzers[0]
        assert original.analyzers[0].prefix != cloned.analyzers[0].prefix

    def test_clone_equality_and_hash(self) -> None:
        """Test that cloned specs are equal and have same hash."""
        specs_to_test = [
            specs.NumRows(),
            specs.First("col"),
            specs.Average("col"),
            specs.Variance("col"),
            specs.Minimum("col"),
            specs.Maximum("col"),
            specs.Sum("col"),
            specs.NullCount("col"),
            specs.NegativeCount("col"),
            specs.DuplicateCount(["col1", "col2"]),
            specs.CountValues("col", [1, 2, 3]),
            specs.UniqueCount("col"),
        ]

        for original in specs_to_test:
            cloned = original.clone()  # type: ignore[attr-defined]

            # Despite being different instances, they should be equal
            assert original == cloned
            assert hash(original) == hash(cloned)

            # But they are different objects
            assert original is not cloned

    def test_clone_multiple_times(self) -> None:
        """Test cloning multiple times creates unique instances."""
        original = specs.Average("price")

        clones = [original.clone() for _ in range(5)]

        # All clones should be different instances
        for i, clone1 in enumerate(clones):
            for j, clone2 in enumerate(clones):
                if i != j:
                    assert clone1 is not clone2
                    # But they should still be equal
                    assert clone1 == clone2

        # All should have different analyzer prefixes
        prefixes_seen = set()

        # Add original's prefixes
        for analyzer in original.analyzers:
            prefixes_seen.add(analyzer.prefix)

        # Check all clones have unique prefixes
        for clone in clones:
            for analyzer in clone.analyzers:
                assert analyzer.prefix not in prefixes_seen
                prefixes_seen.add(analyzer.prefix)

    def test_extended_metrics_no_clone(self) -> None:
        """Verify extended metrics don't have clone method."""
        # Extended metrics should not have clone method
        dod = specs.DayOverDay(base_metric_type="Average", base_parameters={"column": "price"})
        assert not hasattr(dod, "clone")

        wow = specs.WeekOverWeek(base_metric_type="Sum", base_parameters={"column": "revenue"})
        assert not hasattr(wow, "clone")

        stddev = specs.Stddev(base_metric_type="Average", base_parameters={"column": "price"}, offset=1, n=7)
        assert not hasattr(stddev, "clone")

    @pytest.mark.parametrize(
        "spec_class,args",
        [
            (specs.NumRows, ()),
            (specs.First, ("column_name",)),
            (specs.Average, ("price",)),
            (specs.Variance, ("values",)),
            (specs.Minimum, ("score",)),
            (specs.Maximum, ("score",)),
            (specs.Sum, ("amount",)),
            (specs.NullCount, ("email",)),
            (specs.NegativeCount, ("balance",)),
            (specs.DuplicateCount, (["col1", "col2"],)),
            (specs.CountValues, ("status", [1, 2, 3])),
            (specs.UniqueCount, ("product_id",)),
        ],
    )
    def test_all_basic_specs_have_clone(self, spec_class: type[specs.MetricSpec], args: tuple) -> None:
        """Verify all basic metric specs have clone method."""
        spec = spec_class(*args)
        assert hasattr(spec, "clone")
        assert callable(getattr(spec, "clone"))

        # Verify it returns the correct type
        cloned = spec.clone()
        assert type(cloned) is type(spec)

    def test_clone_preserves_spec_behavior(self) -> None:
        """Test that cloned specs behave identically to originals."""
        # Create a spec with specific parameters
        original = specs.CountValues("category", ["A", "B", "C"])
        cloned = original.clone()

        # Both should produce the same name
        assert original.name == cloned.name

        # Both should produce the same parameters
        assert original.parameters == cloned.parameters

        # Both should have the same metric_type
        assert original.metric_type == cloned.metric_type

        # Both should have the same is_extended flag
        assert original.is_extended == cloned.is_extended

        # Both should produce the same string representation
        assert str(original) == str(cloned)
