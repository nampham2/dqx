"""End-to-end tests for date exclusion skip validation."""

from datetime import date
from unittest.mock import MagicMock

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx.api import VerificationSuite
from dqx.common import ResultKey, SymbolicValidator
from dqx.evaluator import Evaluator
from dqx.graph.nodes import AssertionNode, CheckNode, RootNode
from dqx.specs import NumRows, Sum


class TestDateExclusionSkipValidation:
    """Tests for end-to-end skip validation with date exclusion."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create a mock database."""
        db = MagicMock()
        db.get_metrics_stats.return_value = MagicMock(total_metrics=0, expired_metrics=0)
        db.delete_expired_metrics = MagicMock()
        return db

    def test_custom_threshold_configuration(self, mock_db: MagicMock) -> None:
        """Test that custom data availability threshold is properly configured."""
        # Test default threshold
        suite1 = VerificationSuite(checks=[lambda mp, ctx: None], db=mock_db, name="Default Suite")
        assert suite1.data_av_threshold == 0.9  # Default 90%

        # Test custom threshold
        suite2 = VerificationSuite(
            checks=[lambda mp, ctx: None], db=mock_db, name="Custom Suite", data_av_threshold=0.95
        )
        assert suite2.data_av_threshold == 0.95  # Custom 95%

        # Test zero threshold (never skip)
        suite3 = VerificationSuite(
            checks=[lambda mp, ctx: None], db=mock_db, name="Zero Threshold Suite", data_av_threshold=0.0
        )
        assert suite3.data_av_threshold == 0.0

    def test_data_av_threshold_property(self, mock_db: MagicMock) -> None:
        """Test that data_av_threshold property is properly stored."""
        # Test with custom threshold
        suite = VerificationSuite(checks=[lambda mp, ctx: None], db=mock_db, name="Test Suite", data_av_threshold=0.75)
        assert suite.data_av_threshold == 0.75

        # Test default threshold
        suite2 = VerificationSuite(checks=[lambda mp, ctx: None], db=mock_db, name="Test Suite 2")
        assert suite2.data_av_threshold == 0.9  # Default value

    def test_evaluator_skip_logic_below_threshold(self) -> None:
        """Test that Evaluator skips assertions when data availability is below threshold."""
        # Create mock provider
        provider = MagicMock()
        registry = MagicMock()
        provider.registry = registry

        # Create specs
        sum_spec = Sum("value")

        # Mock provider to return low data availability
        provider.get_data_av_ratio.return_value = 0.3  # 30% - below default threshold

        # Create evaluator with threshold
        key = ResultKey(date(2024, 1, 5), {})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.5)

        # Mock provider.get_symbol
        mock_metric = MagicMock()
        mock_metric.metric_spec = sum_spec
        mock_metric.dataset = "test_data"
        mock_metric.name = "sum(value)"
        mock_metric.data_av_ratio = 0.3  # Below threshold
        provider.get_symbol.return_value = mock_metric

        # Create node hierarchy
        root = RootNode("test_root")
        check = CheckNode(parent=root, name="Test Check")
        validator = SymbolicValidator("> 0", lambda x: x > 0)
        node = AssertionNode(
            parent=check,
            name="Sum is positive",
            actual=sp.Symbol("x_1"),
            validator=validator,
        )

        # Mock metrics
        evaluator._metrics = {sp.Symbol("x_1"): Success(100.0)}

        # Visit the node
        evaluator.visit(node)

        # Verify it was skipped
        assert node._result == "SKIPPED"
        assert isinstance(node._metric, Failure)
        assert node._metric.failure()[0].error_message == "Insufficient data availability"

    def test_evaluator_skip_logic_above_threshold(self) -> None:
        """Test that Evaluator evaluates assertions when data availability is above threshold."""
        # Create mock provider
        provider = MagicMock()
        registry = MagicMock()
        provider.registry = registry

        # Create specs
        sum_spec = Sum("value")

        # Mock provider to return high data availability
        provider.get_data_av_ratio.return_value = 0.9  # 90% - above threshold

        # Create evaluator
        key = ResultKey(date(2024, 1, 5), {})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.5)

        # Mock provider.get_symbol
        mock_metric = MagicMock()
        mock_metric.metric_spec = sum_spec
        mock_metric.dataset = "test_data"
        mock_metric.name = "sum(value)"
        mock_metric.data_av_ratio = 0.9  # Above threshold
        provider.get_symbol.return_value = mock_metric

        # Create node hierarchy
        root = RootNode("test_root")
        check = CheckNode(parent=root, name="Test Check")
        validator = SymbolicValidator("> 0", lambda x: x > 0)
        node = AssertionNode(
            parent=check,
            name="Sum is positive",
            actual=sp.Symbol("x_1"),
            validator=validator,
        )

        # Mock metrics
        evaluator._metrics = {sp.Symbol("x_1"): Success(100.0)}

        # Visit the node
        evaluator.visit(node)

        # Verify it was evaluated
        assert node._result == "PASSED"  # 100 > 0, so passes
        assert isinstance(node._metric, Success)
        assert node._metric.unwrap() == 100.0

    def test_evaluator_skip_multiple_metrics_any_below(self) -> None:
        """Test that assertions with multiple metrics are skipped if ANY metric is below threshold."""
        # Create mock provider
        provider = MagicMock()
        registry = MagicMock()
        provider.registry = registry

        # Create specs
        sum_spec = Sum("value")
        num_rows_spec = NumRows()

        # Mock registry to return different availability for each metric
        def get_ratio_side_effect(spec: object, dataset: str | None) -> float | None:
            if isinstance(spec, Sum):
                return 0.3  # Below threshold
            elif isinstance(spec, NumRows):
                return 0.9  # Above threshold
            return None

        provider.get_data_av_ratio.side_effect = get_ratio_side_effect

        # Create evaluator
        key = ResultKey(date(2024, 1, 5), {})
        evaluator = Evaluator(provider, key, "Test Suite", data_av_threshold=0.5)

        # Mock provider.get_symbol for both symbols
        def get_symbol_side_effect(symbol: sp.Symbol) -> MagicMock:
            mock_metric = MagicMock()
            if str(symbol) == "x_1":
                mock_metric.metric_spec = sum_spec
                mock_metric.name = "sum(value)"
                mock_metric.data_av_ratio = 0.3  # Below threshold
            else:
                mock_metric.metric_spec = num_rows_spec
                mock_metric.name = "count()"
                mock_metric.data_av_ratio = 0.9  # Above threshold
            mock_metric.dataset = "test_data"
            return mock_metric

        provider.get_symbol.side_effect = get_symbol_side_effect

        # Create node hierarchy
        root = RootNode("test_root")
        check = CheckNode(parent=root, name="Test Check")
        validator = SymbolicValidator("> 0", lambda x: x > 0)
        node = AssertionNode(
            parent=check,
            name="Average is positive",
            actual=sp.Symbol("x_1") / sp.Symbol("x_2"),  # Average
            validator=validator,
        )

        # Mock metrics
        evaluator._metrics = {sp.Symbol("x_1"): Success(100.0), sp.Symbol("x_2"): Success(10.0)}

        # Visit the node
        evaluator.visit(node)

        # Verify it was skipped because one metric is below threshold
        assert node._result == "SKIPPED"
        assert isinstance(node._metric, Failure)
        assert node._metric.failure()[0].error_message == "Insufficient data availability"
