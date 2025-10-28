"""Test cases for UniqueCount spec."""

from unittest.mock import Mock, patch

from dqx import ops, specs, states


class TestUniqueCount:
    """Test UniqueCount metric spec"""

    def test_metric_type(self) -> None:
        unique = specs.UniqueCount("product_id")
        assert unique.metric_type == "UniqueCount"

    def test_name(self) -> None:
        unique = specs.UniqueCount("product_id")
        assert unique.name == "unique_count(product_id)"

    def test_parameters(self) -> None:
        unique = specs.UniqueCount("product_id")
        assert unique.parameters == {"column": "product_id"}

    def test_analyzers(self) -> None:
        unique = specs.UniqueCount("product_id")
        assert len(unique.analyzers) == 1
        assert isinstance(unique.analyzers[0], ops.UniqueCount)
        assert unique.analyzers[0].column == "product_id"

    @patch("dqx.ops.UniqueCount")
    def test_state(self, mock_ops_unique: Mock) -> None:
        mock_analyzer = Mock()
        mock_analyzer.value.return_value = 42.0
        mock_ops_unique.return_value = mock_analyzer

        unique = specs.UniqueCount("product_id")
        state = unique.state()

        assert isinstance(state, states.NonMergeable)
        assert state.value == 42.0
        assert state.metric_type == "UniqueCount"

    def test_deserialize(self) -> None:
        with patch.object(states.NonMergeable, "deserialize") as mock_deserialize:
            mock_state = Mock()
            mock_deserialize.return_value = mock_state

            result = specs.UniqueCount.deserialize(b"test_bytes")

            mock_deserialize.assert_called_once_with(b"test_bytes")
            assert result == mock_state

    def test_hash(self) -> None:
        unique1 = specs.UniqueCount("product_id")
        unique2 = specs.UniqueCount("product_id")
        unique3 = specs.UniqueCount("user_id")

        assert hash(unique1) == hash(unique2)
        assert hash(unique1) != hash(unique3)

    def test_equality(self) -> None:
        unique1 = specs.UniqueCount("product_id")
        unique2 = specs.UniqueCount("product_id")
        unique3 = specs.UniqueCount("user_id")

        assert unique1 == unique2
        assert unique1 != unique3

    def test_inequality_different_type(self) -> None:
        unique = specs.UniqueCount("product_id")
        assert unique != specs.NumRows()
        assert unique != specs.DuplicateCount(["product_id"])
        assert unique != "not_a_unique_count"
        assert unique != 42
        assert unique is not None

    def test_str(self) -> None:
        unique = specs.UniqueCount("product_id")
        assert str(unique) == "unique_count(product_id)"

    def test_is_metric_spec(self) -> None:
        unique = specs.UniqueCount("product_id")
        assert isinstance(unique, specs.MetricSpec)
