import inspect
from unittest.mock import Mock, patch

from dqx import ops, specs, states


class TestMetricSpec:
    """Test the MetricSpec Protocol"""

    def test_num_rows_is_metric_spec(self) -> None:
        nr = specs.NumRows()
        assert isinstance(nr, specs.MetricSpec)

    def test_all_specs_are_metric_specs(self) -> None:
        spec_classes = [
            specs.NumRows,
            specs.First,
            specs.Average,
            specs.Variance,
            specs.Minimum,
            specs.Maximum,
            specs.Sum,
            specs.NullCount,
            specs.NegativeCount,
            specs.ApproxCardinality,
        ]

        for spec_class in spec_classes:
            if spec_class == specs.NumRows:
                instance = spec_class()
            else:
                instance = spec_class("test_column")
            assert isinstance(instance, specs.MetricSpec)


class TestNumRows:
    """Test NumRows metric spec"""

    def test_metric_type(self) -> None:
        nr = specs.NumRows()
        assert nr.metric_type == "NumRows"

    def test_name(self) -> None:
        nr = specs.NumRows()
        assert nr.name == "num_rows()"

    def test_parameters(self) -> None:
        nr = specs.NumRows()
        assert nr.parameters == {}

    def test_analyzers(self) -> None:
        nr = specs.NumRows()
        assert len(nr.analyzers) == 1
        assert isinstance(nr.analyzers[0], ops.NumRows)

    @patch("dqx.ops.NumRows")
    def test_state(self, mock_ops_numrows: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 100.0
        mock_ops_numrows.return_value = mock_analyzer

        nr = specs.NumRows()
        state = nr.state()

        assert isinstance(state, states.SimpleAdditiveState)
        assert state.value == 100.0

    def test_deserialize(self) -> None:
        with patch.object(states.SimpleAdditiveState, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.NumRows.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        nr1 = specs.NumRows()
        nr2 = specs.NumRows()
        assert hash(nr1) == hash(nr2)

    def test_equality(self) -> None:
        nr1 = specs.NumRows()
        nr2 = specs.NumRows()
        assert nr1 == nr2

    def test_inequality_different_type(self) -> None:
        nr = specs.NumRows()
        assert nr != "not_a_numrows"
        assert nr != 42
        assert nr is not None


class TestFirst:
    """Test First metric spec"""

    def test_metric_type(self) -> None:
        first = specs.First("test_col")
        assert first.metric_type == "First"

    def test_name(self) -> None:
        first = specs.First("test_column")
        assert first.name == "first(test_column)"

    def test_parameters(self) -> None:
        first = specs.First("test_column")
        assert first.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        first = specs.First("test_column")
        assert len(first.analyzers) == 1
        assert isinstance(first.analyzers[0], ops.First)

    @patch("dqx.ops.First")
    def test_state(self, mock_ops_first: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 10.0
        mock_ops_first.return_value = mock_analyzer

        first = specs.First("test_column")
        state = first.state()

        assert isinstance(state, states.First)
        assert state.value == 10.0

    def test_deserialize(self) -> None:
        with patch.object(states.First, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.First.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        first1 = specs.First("test_col")
        first2 = specs.First("test_col")
        first3 = specs.First("other_col")
        assert hash(first1) == hash(first2)
        assert hash(first1) != hash(first3)

    def test_equality(self) -> None:
        first1 = specs.First("test_col")
        first2 = specs.First("test_col")
        first3 = specs.First("other_col")
        assert first1 == first2
        assert first1 != first3

    def test_inequality_different_type(self) -> None:
        first = specs.First("test_col")
        assert first != specs.NumRows()
        assert first != "not_a_first"


class TestAverage:
    """Test Average metric spec"""

    def test_metric_type(self) -> None:
        avg = specs.Average("test_col")
        assert avg.metric_type == "Average"

    def test_name(self) -> None:
        avg = specs.Average("impressions")
        assert avg.name == "average(impressions)"

    def test_parameters(self) -> None:
        avg = specs.Average("impressions")
        assert avg.parameters == {"column": "impressions"}

    def test_analyzers(self) -> None:
        avg = specs.Average("impressions")
        assert len(avg.analyzers) == 2
        assert isinstance(avg.analyzers[0], ops.NumRows)
        assert isinstance(avg.analyzers[1], ops.Average)

    @patch("dqx.ops.NumRows")
    @patch("dqx.ops.Average")
    def test_state(self, mock_ops_average: Mock, mock_ops_numrows: Mock) -> None:
        mock_numrows_analyzer = Mock()
        mock_numrows_analyzer.value.return_value = 100.0
        mock_avg_analyzer = Mock()
        mock_avg_analyzer.value.return_value = 50.0

        mock_ops_numrows.return_value = mock_numrows_analyzer
        mock_ops_average.return_value = mock_avg_analyzer

        avg = specs.Average("test_column")
        state = avg.state()

        assert isinstance(state, states.Average)
        assert state.value == 50.0
        assert state.n == 100.0

    def test_deserialize(self) -> None:
        with patch.object(states.Average, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.Average.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        avg1 = specs.Average("impressions")
        avg2 = specs.Average("impressions")
        avg3 = specs.Average("clicks")
        assert hash(avg1) == hash(avg2)
        assert hash(avg1) != hash(avg3)

    def test_equality(self) -> None:
        avg1 = specs.Average("impressions")
        avg2 = specs.Average("impressions")
        avg3 = specs.Average("clicks")
        assert avg1 == avg2
        assert avg1 != avg3

    def test_inequality_different_type(self) -> None:
        avg = specs.Average("impressions")
        assert avg != specs.NumRows()
        assert avg != "not_an_average"


class TestVariance:
    """Test Variance metric spec"""

    def test_metric_type(self) -> None:
        var = specs.Variance("test_col")
        assert var.metric_type == "Variance"

    def test_name(self) -> None:
        var = specs.Variance("test_column")
        assert var.name == "variance(test_column)"

    def test_parameters(self) -> None:
        var = specs.Variance("test_column")
        assert var.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        var = specs.Variance("test_column")
        assert len(var.analyzers) == 3
        assert isinstance(var.analyzers[0], ops.NumRows)
        assert isinstance(var.analyzers[1], ops.Average)
        assert isinstance(var.analyzers[2], ops.Variance)

    @patch("dqx.ops.NumRows")
    @patch("dqx.ops.Average")
    @patch("dqx.ops.Variance")
    def test_state(self, mock_ops_variance: Mock, mock_ops_average: Mock, mock_ops_numrows: Mock) -> None:
        mock_numrows_analyzer = Mock()
        mock_numrows_analyzer.value.return_value = 100.0
        mock_avg_analyzer = Mock()
        mock_avg_analyzer.value.return_value = 50.0
        mock_var_analyzer = Mock()
        mock_var_analyzer.value.return_value = 25.0

        mock_ops_numrows.return_value = mock_numrows_analyzer
        mock_ops_average.return_value = mock_avg_analyzer
        mock_ops_variance.return_value = mock_var_analyzer

        var = specs.Variance("test_column")
        state = var.state()

        assert isinstance(state, states.Variance)
        assert state.value == 25.0
        assert state.avg == 50.0
        assert state.n == 100.0

    def test_deserialize(self) -> None:
        with patch.object(states.Variance, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.Variance.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        var1 = specs.Variance("test_col")
        var2 = specs.Variance("test_col")
        var3 = specs.Variance("other_col")
        assert hash(var1) == hash(var2)
        assert hash(var1) != hash(var3)

    def test_equality(self) -> None:
        var1 = specs.Variance("test_col")
        var2 = specs.Variance("test_col")
        var3 = specs.Variance("other_col")
        assert var1 == var2
        assert var1 != var3

    def test_inequality_different_type(self) -> None:
        var = specs.Variance("test_col")
        assert var != specs.NumRows()
        assert var != "not_a_variance"


class TestMinimum:
    """Test Minimum metric spec"""

    def test_metric_type(self) -> None:
        min_spec = specs.Minimum("test_col")
        assert min_spec.metric_type == "Minimum"

    def test_name(self) -> None:
        min_spec = specs.Minimum("test_column")
        assert min_spec.name == "minimum(test_column)"

    def test_parameters(self) -> None:
        min_spec = specs.Minimum("test_column")
        assert min_spec.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        min_spec = specs.Minimum("test_column")
        assert len(min_spec.analyzers) == 1
        assert isinstance(min_spec.analyzers[0], ops.Minimum)

    @patch("dqx.ops.Minimum")
    def test_state(self, mock_ops_minimum: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 10.0
        mock_ops_minimum.return_value = mock_analyzer

        min_spec = specs.Minimum("test_column")
        state = min_spec.state()

        assert isinstance(state, states.Minimum)
        assert state.value == 10.0

    def test_deserialize(self) -> None:
        with patch.object(states.Minimum, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.Minimum.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        min1 = specs.Minimum("test_col")
        min2 = specs.Minimum("test_col")
        min3 = specs.Minimum("other_col")
        assert hash(min1) == hash(min2)
        assert hash(min1) != hash(min3)

    def test_equality(self) -> None:
        min1 = specs.Minimum("test_col")
        min2 = specs.Minimum("test_col")
        min3 = specs.Minimum("other_col")
        assert min1 == min2
        assert min1 != min3

    def test_inequality_different_type(self) -> None:
        min_spec = specs.Minimum("test_col")
        assert min_spec != specs.NumRows()
        assert min_spec != "not_a_minimum"


class TestMaximum:
    """Test Maximum metric spec"""

    def test_metric_type(self) -> None:
        max_spec = specs.Maximum("test_col")
        assert max_spec.metric_type == "Maximum"

    def test_name(self) -> None:
        max_spec = specs.Maximum("test_column")
        assert max_spec.name == "maximum(test_column)"

    def test_parameters(self) -> None:
        max_spec = specs.Maximum("test_column")
        assert max_spec.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        max_spec = specs.Maximum("test_column")
        assert len(max_spec.analyzers) == 1
        assert isinstance(max_spec.analyzers[0], ops.Maximum)

    @patch("dqx.ops.Maximum")
    def test_state(self, mock_ops_maximum: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 100.0
        mock_ops_maximum.return_value = mock_analyzer

        max_spec = specs.Maximum("test_column")
        state = max_spec.state()

        assert isinstance(state, states.Maximum)
        assert state.value == 100.0

    def test_deserialize(self) -> None:
        with patch.object(states.Maximum, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.Maximum.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        max1 = specs.Maximum("test_col")
        max2 = specs.Maximum("test_col")
        max3 = specs.Maximum("other_col")
        assert hash(max1) == hash(max2)
        assert hash(max1) != hash(max3)

    def test_equality(self) -> None:
        max1 = specs.Maximum("test_col")
        max2 = specs.Maximum("test_col")
        max3 = specs.Maximum("other_col")
        assert max1 == max2
        assert max1 != max3

    def test_inequality_different_type(self) -> None:
        max_spec = specs.Maximum("test_col")
        assert max_spec != specs.NumRows()
        assert max_spec != "not_a_maximum"


class TestSum:
    """Test Sum metric spec"""

    def test_metric_type(self) -> None:
        sum_spec = specs.Sum("test_col")
        assert sum_spec.metric_type == "Sum"

    def test_name(self) -> None:
        sum_spec = specs.Sum("test_column")
        assert sum_spec.name == "sum(test_column)"

    def test_parameters(self) -> None:
        sum_spec = specs.Sum("test_column")
        assert sum_spec.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        sum_spec = specs.Sum("test_column")
        assert len(sum_spec.analyzers) == 1
        assert isinstance(sum_spec.analyzers[0], ops.Sum)

    @patch("dqx.ops.Sum")
    def test_state(self, mock_ops_sum: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 500.0
        mock_ops_sum.return_value = mock_analyzer

        sum_spec = specs.Sum("test_column")
        state = sum_spec.state()

        assert isinstance(state, states.SimpleAdditiveState)
        assert state.value == 500.0

    def test_deserialize(self) -> None:
        with patch.object(states.SimpleAdditiveState, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.Sum.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        sum1 = specs.Sum("test_col")
        sum2 = specs.Sum("test_col")
        sum3 = specs.Sum("other_col")
        assert hash(sum1) == hash(sum2)
        assert hash(sum1) != hash(sum3)

    def test_equality(self) -> None:
        sum1 = specs.Sum("test_col")
        sum2 = specs.Sum("test_col")
        sum3 = specs.Sum("other_col")
        assert sum1 == sum2
        assert sum1 != sum3

    def test_inequality_different_type(self) -> None:
        sum_spec = specs.Sum("test_col")
        assert sum_spec != specs.NumRows()
        assert sum_spec != "not_a_sum"


class TestNullCount:
    """Test NullCount metric spec"""

    def test_metric_type(self) -> None:
        null_count = specs.NullCount("test_col")
        assert null_count.metric_type == "NullCount"

    def test_name(self) -> None:
        null_count = specs.NullCount("test_column")
        assert null_count.name == "null_count(test_column)"

    def test_parameters(self) -> None:
        null_count = specs.NullCount("test_column")
        assert null_count.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        null_count = specs.NullCount("test_column")
        assert len(null_count.analyzers) == 1
        assert isinstance(null_count.analyzers[0], ops.NullCount)

    @patch("dqx.ops.NullCount")
    def test_state(self, mock_ops_nullcount: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 5.0
        mock_ops_nullcount.return_value = mock_analyzer

        null_count = specs.NullCount("test_column")
        state = null_count.state()

        assert isinstance(state, states.SimpleAdditiveState)
        assert state.value == 5.0

    def test_deserialize(self) -> None:
        with patch.object(states.SimpleAdditiveState, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.NullCount.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        null1 = specs.NullCount("test_col")
        null2 = specs.NullCount("test_col")
        null3 = specs.NullCount("other_col")
        assert hash(null1) == hash(null2)
        assert hash(null1) != hash(null3)

    def test_equality(self) -> None:
        null1 = specs.NullCount("test_col")
        null2 = specs.NullCount("test_col")
        null3 = specs.NullCount("other_col")
        assert null1 == null2
        assert null1 != null3

    def test_inequality_different_type(self) -> None:
        null_count = specs.NullCount("test_col")
        assert null_count != specs.NumRows()
        assert null_count != "not_a_nullcount"


class TestNegativeCount:
    """Test NegativeCount metric spec"""

    def test_metric_type(self) -> None:
        neg_count = specs.NegativeCount("test_col")
        assert neg_count.metric_type == "NegativeCount"

    def test_name(self) -> None:
        neg_count = specs.NegativeCount("test_column")
        assert neg_count.name == "non_negative(test_column)"

    def test_parameters(self) -> None:
        neg_count = specs.NegativeCount("test_column")
        assert neg_count.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        neg_count = specs.NegativeCount("test_column")
        assert len(neg_count.analyzers) == 1
        assert isinstance(neg_count.analyzers[0], ops.NegativeCount)

    @patch("dqx.ops.NegativeCount")
    def test_state(self, mock_ops_negcount: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 3.0
        mock_ops_negcount.return_value = mock_analyzer

        neg_count = specs.NegativeCount("test_column")
        state = neg_count.state()

        assert isinstance(state, states.SimpleAdditiveState)
        assert state.value == 3.0

    def test_deserialize(self) -> None:
        with patch.object(states.SimpleAdditiveState, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.NegativeCount.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        neg1 = specs.NegativeCount("test_col")
        neg2 = specs.NegativeCount("test_col")
        neg3 = specs.NegativeCount("other_col")
        assert hash(neg1) == hash(neg2)
        assert hash(neg1) != hash(neg3)

    def test_equality(self) -> None:
        neg1 = specs.NegativeCount("test_col")
        neg2 = specs.NegativeCount("test_col")
        neg3 = specs.NegativeCount("other_col")
        assert neg1 == neg2
        assert neg1 != neg3

    def test_inequality_different_type(self) -> None:
        neg_count = specs.NegativeCount("test_col")
        assert neg_count != specs.NumRows()
        assert neg_count != "not_a_negativecount"


class TestApproxCardinality:
    """Test ApproxCardinality metric spec"""

    def test_metric_type(self) -> None:
        approx_card = specs.ApproxCardinality("test_col")
        assert approx_card.metric_type == "ApproxCardinality"

    def test_name(self) -> None:
        approx_card = specs.ApproxCardinality("test_column")
        assert approx_card.name == "approx_cardinality(test_column)"

    def test_parameters(self) -> None:
        approx_card = specs.ApproxCardinality("test_column")
        assert approx_card.parameters == {"column": "test_column"}

    def test_analyzers(self) -> None:
        approx_card = specs.ApproxCardinality("test_column")
        assert len(approx_card.analyzers) == 1
        assert isinstance(approx_card.analyzers[0], ops.ApproxCardinality)

    @patch("dqx.ops.ApproxCardinality")
    def test_state(self, mock_ops_approxcard: Mock) -> None:
        mock_sketch = Mock(spec=states.CardinalitySketch)
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = mock_sketch
        mock_ops_approxcard.return_value = mock_analyzer

        approx_card = specs.ApproxCardinality("test_column")
        state = approx_card.state()

        assert state == mock_sketch

    def test_deserialize(self) -> None:
        with patch.object(states.CardinalitySketch, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.ApproxCardinality.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        approx1 = specs.ApproxCardinality("test_col")
        approx2 = specs.ApproxCardinality("test_col")
        approx3 = specs.ApproxCardinality("other_col")
        assert hash(approx1) == hash(approx2)
        assert hash(approx1) != hash(approx3)

    def test_equality(self) -> None:
        approx1 = specs.ApproxCardinality("test_col")
        approx2 = specs.ApproxCardinality("test_col")
        approx3 = specs.ApproxCardinality("other_col")
        assert approx1 == approx2
        assert approx1 != approx3

    def test_inequality_different_type(self) -> None:
        approx_card = specs.ApproxCardinality("test_col")
        assert approx_card != specs.NumRows()
        assert approx_card != "not_an_approxcardinality"


class TestBuildRegistry:
    """Test the _build_registry function"""

    def test_build_registry_returns_dict(self) -> None:
        """Test that _build_registry returns a dictionary."""
        result = specs._build_registry()
        assert isinstance(result, dict)

    def test_build_registry_finds_all_metric_classes(self) -> None:
        """Test that _build_registry finds all expected metric spec classes."""
        result = specs._build_registry()
        expected_types = {
            "NumRows",
            "First",
            "Average",
            "Minimum",
            "Maximum",
            "Sum",
            "NegativeCount",
            "NullCount",
            "ApproxCardinality",
            "Variance",
            "DuplicateCount",
        }
        assert set(result.keys()) == expected_types

    def test_build_registry_maps_correct_classes(self) -> None:
        """Test that _build_registry maps metric types to correct classes."""
        result = specs._build_registry()
        assert result["NumRows"] == specs.NumRows
        assert result["First"] == specs.First
        assert result["Average"] == specs.Average
        assert result["Minimum"] == specs.Minimum
        assert result["Maximum"] == specs.Maximum
        assert result["Sum"] == specs.Sum
        assert result["NegativeCount"] == specs.NegativeCount
        assert result["NullCount"] == specs.NullCount
        assert result["ApproxCardinality"] == specs.ApproxCardinality
        assert result["Variance"] == specs.Variance

    def test_build_registry_excludes_protocol(self) -> None:
        """Test that _build_registry excludes the MetricSpec protocol."""
        result = specs._build_registry()
        # Verify the protocol class is not in the registry values
        assert specs.MetricSpec not in result.values()

    def test_build_registry_excludes_non_classes(self) -> None:
        """Test that _build_registry excludes non-class objects."""
        result = specs._build_registry()
        # Should only contain actual classes, not functions, modules, etc.
        for name, obj in result.items():
            assert inspect.isclass(obj)
            assert hasattr(obj, "metric_type")

    def test_build_registry_classes_have_metric_type(self) -> None:
        """Test that all classes in registry have metric_type attribute."""
        result = specs._build_registry()
        for metric_type, spec_class in result.items():
            assert hasattr(spec_class, "metric_type")
            assert spec_class.metric_type == metric_type

    def test_build_registry_classes_are_instantiable(self) -> None:
        """Test that all classes in registry can be instantiable."""
        result = specs._build_registry()
        for metric_type, spec_class in result.items():
            if metric_type == "NumRows":
                instance = spec_class()
            else:
                # Other classes require a column parameter
                instance = spec_class("test_column")  # type: ignore[call-arg]
            assert isinstance(instance, specs.MetricSpec)

    @patch("inspect.currentframe")
    def test_build_registry_with_mock_frame(self, mock_currentframe: Mock) -> None:
        """Test _build_registry with mocked currentframe to ensure it handles globals correctly."""
        # Create a mock frame with f_globals containing our test classes
        mock_frame = Mock()
        mock_frame.f_globals = {
            "NumRows": specs.NumRows,
            "First": specs.First,
            "Average": specs.Average,
            "MetricSpec": specs.MetricSpec,  # Should be excluded
            "some_function": lambda: None,  # Should be excluded
            "some_string": "test",  # Should be excluded
            "inspect": inspect,  # Should be excluded
        }
        mock_currentframe.return_value = mock_frame

        result = specs._build_registry()

        # Should only include actual metric spec classes
        expected_classes = {"NumRows", "First", "Average"}
        assert set(result.keys()) == expected_classes
        assert specs.MetricSpec not in result.values()

    def test_build_registry_consistent_with_actual_registry(self) -> None:
        """Test that _build_registry produces the same result as the actual registry."""
        built_registry = specs._build_registry()
        actual_registry = specs.registry

        assert set(built_registry.keys()) == set(actual_registry.keys())
        for metric_type in built_registry:
            assert built_registry[metric_type] == actual_registry[metric_type]


class TestRegistry:
    """Test the registry dictionary"""

    def test_registry_contains_all_metric_types(self) -> None:
        expected_types = {
            "NumRows",
            "First",
            "Average",
            "Minimum",
            "Maximum",
            "Sum",
            "NegativeCount",
            "NullCount",
            "ApproxCardinality",
            "Variance",
            "DuplicateCount",
        }
        assert set(specs.registry.keys()) == expected_types

    def test_registry_maps_to_correct_classes(self) -> None:
        assert specs.registry["NumRows"] == specs.NumRows
        assert specs.registry["First"] == specs.First
        assert specs.registry["Average"] == specs.Average
        assert specs.registry["Minimum"] == specs.Minimum
        assert specs.registry["Maximum"] == specs.Maximum
        assert specs.registry["Sum"] == specs.Sum
        assert specs.registry["NegativeCount"] == specs.NegativeCount
        assert specs.registry["NullCount"] == specs.NullCount
        assert specs.registry["ApproxCardinality"] == specs.ApproxCardinality
        assert specs.registry["Variance"] == specs.Variance

    def test_registry_classes_are_metric_specs(self) -> None:
        for metric_type, spec_class in specs.registry.items():
            if metric_type == "NumRows":
                instance = spec_class()
            else:
                instance = spec_class("test_column")  # type: ignore[call-arg]
            assert isinstance(instance, specs.MetricSpec)

    def test_registry_is_built_automatically(self) -> None:
        """Test that the registry is built automatically and matches _build_registry output."""
        manually_built = specs._build_registry()
        assert specs.registry == manually_built


class TestMetricTypes:
    """Test the MetricType literal"""

    def test_all_classes_have_correct_metric_type(self) -> None:
        spec_instances: list[tuple[specs.MetricSpec, str]] = [
            (specs.NumRows(), "NumRows"),
            (specs.First("col"), "First"),
            (specs.Average("col"), "Average"),
            (specs.Variance("col"), "Variance"),
            (specs.Minimum("col"), "Minimum"),
            (specs.Maximum("col"), "Maximum"),
            (specs.Sum("col"), "Sum"),
            (specs.NullCount("col"), "NullCount"),
            (specs.NegativeCount("col"), "NegativeCount"),
            (specs.ApproxCardinality("col"), "ApproxCardinality"),
        ]

        for instance, expected_type in spec_instances:
            assert instance.metric_type == expected_type


# Legacy test functions for backward compatibility
def test_num_rows() -> None:
    nr = specs.NumRows()
    assert isinstance(nr, specs.MetricSpec)
    nr_1 = specs.NumRows()
    assert nr == nr_1
    assert nr.name == "num_rows()"


def test_average() -> None:
    nr = specs.Average("impressions")
    assert isinstance(nr, specs.MetricSpec)
    nr_1 = specs.Average("impressions")
    assert nr == nr_1
    assert nr.name == "average(impressions)"
