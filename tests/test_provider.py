from unittest.mock import ANY, Mock, patch

import pytest
import sympy as sp

from dqx import specs
from dqx.common import DQXError
from dqx.orm.repositories import MetricDB
from dqx.provider import (
    ExtendedMetricProvider,
    MetricProvider,
    SymbolicMetric,
    SymbolicMetricBase,
    SymbolIndex,
)


class TestSymbolicMetric:
    """Test the SymbolicMetric dataclass."""

    def test_symbolic_metric_creation(self) -> None:
        """Test creating a SymbolicMetric instance."""
        symbol = sp.Symbol("x_1")
        fn = Mock()
        metric_spec = Mock(spec=specs.MetricSpec)

        metric = SymbolicMetric(
            name="test_metric",
            symbol=symbol,
            fn=fn,
            metric_spec=metric_spec,
            lag=7,
            dataset="dataset1",
            required_metrics=[sp.Symbol("x_0")],
        )

        assert metric.name == "test_metric"
        assert metric.symbol == symbol
        assert metric.fn == fn
        assert metric.metric_spec == metric_spec
        assert metric.lag == 7
        assert metric.dataset == "dataset1"
        assert metric.required_metrics == [sp.Symbol("x_0")]

    def test_symbolic_metric_default_values(self) -> None:
        """Test that optional fields have correct defaults."""
        symbol = sp.Symbol("x_1")
        fn = Mock()
        metric_spec = Mock(spec=specs.MetricSpec)

        metric = SymbolicMetric(name="test_metric", symbol=symbol, fn=fn, metric_spec=metric_spec)

        assert metric.lag == 0
        assert metric.dataset is None
        assert metric.required_metrics == []


class TestSymbolicMetricBase:
    """Test the SymbolicMetricBase class."""

    @pytest.fixture
    def base(self) -> SymbolicMetricBase:
        """Create a SymbolicMetricBase instance for testing."""
        return SymbolicMetricBase()

    def test_init(self, base: SymbolicMetricBase) -> None:
        """Test initialization of SymbolicMetricBase."""
        assert base._registry._metrics == []
        assert base._registry._symbol_index == {}
        assert base._registry._curr_index == 0
        assert base._registry._mutex is not None

    def test_symbols_empty(self, base: SymbolicMetricBase) -> None:
        """Test symbols() method when no symbols are registered."""
        symbols = list(base.symbols())
        assert symbols == []

    def test_get_symbol_success(self, base: SymbolicMetricBase) -> None:
        """Test get_symbol() method with existing symbol."""
        symbol = sp.Symbol("x_1")
        fn = Mock()
        metric_spec = Mock(spec=specs.MetricSpec)

        # Manually create and add the symbolic metric for this test
        base._registry._metrics.append(
            SymbolicMetric(name="test_metric", symbol=symbol, fn=fn, metric_spec=metric_spec, lag=3, dataset="dataset1")
        )
        base._registry._symbol_index[symbol] = base._registry._metrics[-1]

        result = base.get_symbol(symbol)
        assert result.name == "test_metric"
        assert result.symbol == symbol
        assert result.fn == fn
        assert result.lag == 3
        assert result.dataset == "dataset1"

    def test_get_symbol_not_found(self, base: SymbolicMetricBase) -> None:
        """Test get_symbol() method with non-existent symbol."""
        symbol = sp.Symbol("x_nonexistent")

        with pytest.raises(DQXError, match="Symbol x_nonexistent not found"):
            base.get_symbol(symbol)

    def test_next_symbol_default_prefix(self, base: SymbolicMetricBase) -> None:
        """Test _next_symbol() method with default prefix."""
        symbol1 = base._registry._next_symbol()
        symbol2 = base._registry._next_symbol()

        assert symbol1.name == "x_1"
        assert symbol2.name == "x_2"
        assert base._registry._curr_index == 2

    def test_next_symbol_custom_prefix(self, base: SymbolicMetricBase) -> None:
        """Test _next_symbol() method with custom prefix."""
        symbol1 = base._registry._next_symbol("metric")
        symbol2 = base._registry._next_symbol("test")

        assert symbol1.name == "metric_1"
        assert symbol2.name == "test_2"
        assert base._registry._curr_index == 2

    def test_next_symbol_thread_safety(self, base: SymbolicMetricBase) -> None:
        """Test that _next_symbol() is thread-safe."""
        import threading

        symbols = []

        def generate_symbol() -> None:
            symbols.append(base._registry._next_symbol())

        threads = [threading.Thread(target=generate_symbol) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All symbols should be unique
        symbol_names = [s.name for s in symbols]
        assert len(set(symbol_names)) == 10


class TestMetricProvider:
    """Test the MetricProvider class."""

    @pytest.fixture
    def mock_db(self) -> Mock:
        """Create a mock MetricDB."""
        return Mock(spec=MetricDB)

    @pytest.fixture
    def provider(self, mock_db: Mock) -> MetricProvider:
        """Create a MetricProvider instance."""
        return MetricProvider(mock_db, execution_id="test-exec-123")

    def test_init(self, mock_db: Mock) -> None:
        """Test MetricProvider initialization."""
        execution_id = "test-exec-123"
        provider = MetricProvider(mock_db, execution_id)
        assert provider._db == mock_db
        assert provider._execution_id == execution_id
        assert provider._registry._metrics == []
        assert provider._registry._symbol_index == {}
        assert provider._registry._curr_index == 0

    def test_execution_id_property(self, provider: MetricProvider) -> None:
        """Test execution_id property."""
        assert provider.execution_id == "test-exec-123"

    def test_ext_property(self, provider: MetricProvider) -> None:
        """Test the ext property."""
        ext = provider.ext
        assert isinstance(ext, ExtendedMetricProvider)
        assert ext._provider == provider

    @patch("dqx.provider.compute.simple_metric")
    def test_metric(self, mock_simple_metric: Mock, provider: MetricProvider) -> None:
        """Test metric() method."""
        mock_metric_spec = Mock(spec=specs.MetricSpec)
        mock_metric_spec.name = "test_metric"
        lag = 7
        dataset = "dataset1"

        symbol = provider.metric(mock_metric_spec, lag, dataset)

        assert isinstance(symbol, sp.Symbol)
        assert symbol.name == "x_1"

        # Check that the symbol was registered
        registered_metric = provider.get_symbol(symbol)
        assert registered_metric.name == "test_metric"  # No lag prefix with new naming convention
        assert registered_metric.lag == 7
        assert registered_metric.dataset == dataset
        assert registered_metric.metric_spec == mock_metric_spec

    def test_metric_with_defaults(self, provider: MetricProvider) -> None:
        """Test metric() method with default parameters."""
        mock_metric_spec = Mock(spec=specs.MetricSpec)
        mock_metric_spec.name = "test_metric"

        symbol = provider.metric(mock_metric_spec)

        registered_metric = provider.get_symbol(symbol)
        assert registered_metric.name == "test_metric"  # No lag prefix when lag=0
        assert registered_metric.lag == 0
        assert registered_metric.dataset is None

    @patch("dqx.provider.specs.NumRows")
    def test_num_rows(self, mock_num_rows: Mock, provider: MetricProvider) -> None:
        """Test num_rows() method."""
        mock_spec = Mock()
        mock_num_rows.return_value = mock_spec
        lag = 3
        dataset = "dataset1"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.num_rows(lag, dataset)

            mock_num_rows.assert_called_once()
            mock_metric.assert_called_once_with(mock_spec, lag, dataset)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.First")
    def test_first(self, mock_first: Mock, provider: MetricProvider) -> None:
        """Test first() method."""
        mock_spec = Mock()
        mock_first.return_value = mock_spec
        column = "test_column"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.first(column)

            mock_first.assert_called_once_with(column)
            mock_metric.assert_called_once_with(mock_spec, 0, None)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.Average")
    def test_average(self, mock_average: Mock, provider: MetricProvider) -> None:
        """Test average() method."""
        mock_spec = Mock()
        mock_average.return_value = mock_spec
        column = "test_column"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.average(column)

            mock_average.assert_called_once_with(column)
            mock_metric.assert_called_once_with(mock_spec, ANY, None)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.Minimum")
    def test_minimum(self, mock_minimum: Mock, provider: MetricProvider) -> None:
        """Test minimum() method."""
        mock_spec = Mock()
        mock_minimum.return_value = mock_spec
        column = "test_column"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.minimum(column)

            mock_minimum.assert_called_once_with(column)
            mock_metric.assert_called_once_with(mock_spec, ANY, None)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.Maximum")
    def test_maximum(self, mock_maximum: Mock, provider: MetricProvider) -> None:
        """Test maximum() method."""
        mock_spec = Mock()
        mock_maximum.return_value = mock_spec
        column = "test_column"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.maximum(column)

            mock_maximum.assert_called_once_with(column)
            mock_metric.assert_called_once_with(mock_spec, ANY, None)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.Sum")
    def test_sum(self, mock_sum: Mock, provider: MetricProvider) -> None:
        """Test sum() method."""
        mock_spec = Mock()
        mock_sum.return_value = mock_spec
        column = "test_column"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.sum(column)

            mock_sum.assert_called_once_with(column)
            mock_metric.assert_called_once_with(mock_spec, ANY, None)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.NullCount")
    def test_null_count(self, mock_null_count: Mock, provider: MetricProvider) -> None:
        """Test null_count() method."""
        mock_spec = Mock()
        mock_null_count.return_value = mock_spec
        column = "test_column"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.null_count(column)

            mock_null_count.assert_called_once_with(column)
            mock_metric.assert_called_once_with(mock_spec, ANY, None)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.Variance")
    def test_variance(self, mock_variance: Mock, provider: MetricProvider) -> None:
        """Test variance() method."""
        mock_spec = Mock()
        mock_variance.return_value = mock_spec
        column = "test_column"

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.variance(column)

            mock_variance.assert_called_once_with(column)
            mock_metric.assert_called_once_with(mock_spec, ANY, None)
            assert result == sp.Symbol("x_1")

    @patch("dqx.provider.specs.DuplicateCount")
    def test_duplicate_count(self, mock_duplicate_count: Mock, provider: MetricProvider) -> None:
        """Test duplicate_count() method."""
        mock_spec = Mock()
        mock_duplicate_count.return_value = mock_spec
        columns = ["col1", "col2"]

        with patch.object(provider, "metric") as mock_metric:
            mock_metric.return_value = sp.Symbol("x_1")

            result = provider.duplicate_count(columns)

            mock_duplicate_count.assert_called_once_with(columns)
            mock_metric.assert_called_once_with(mock_spec, ANY, None)
            assert result == sp.Symbol("x_1")

    def test_duplicate_count_integration(self, provider: MetricProvider) -> None:
        """Test duplicate_count() method integration."""
        columns = ["user_id", "session_id"]

        symbol = provider.duplicate_count(columns)

        assert isinstance(symbol, sp.Symbol)

        # Check that the symbol was registered
        registered_metric = provider.get_symbol(symbol)
        assert registered_metric.name == "duplicate_count(session_id,user_id)"  # Should be sorted
        assert registered_metric.lag == 0
        assert registered_metric.dataset is None
        assert isinstance(registered_metric.metric_spec, specs.DuplicateCount)
        assert registered_metric.metric_spec.parameters == {"columns": ["session_id", "user_id"]}


class TestExtendedMetricProvider:
    """Test the ExtendedMetricProvider class."""

    @pytest.fixture
    def mock_db(self) -> Mock:
        """Create a mock MetricDB."""
        return Mock(spec=MetricDB)

    @pytest.fixture
    def provider(self, mock_db: Mock) -> MetricProvider:
        """Create a MetricProvider instance."""
        return MetricProvider(mock_db, execution_id="test-exec-123")

    @pytest.fixture
    def ext_provider(self, provider: MetricProvider) -> ExtendedMetricProvider:
        """Create an ExtendedMetricProvider instance."""
        return provider.ext

    def test_init(self, provider: MetricProvider, ext_provider: ExtendedMetricProvider) -> None:
        """Test ExtendedMetricProvider initialization."""
        assert ext_provider._provider == provider
        assert ext_provider.db == provider._db
        assert ext_provider.execution_id == provider.execution_id
        # ExtendedMetricProvider.registry returns provider._registry
        assert ext_provider.registry == provider._registry
        assert ext_provider.registry._next_symbol == provider._registry._next_symbol
        assert ext_provider.registry.register == provider._registry.register


class TestTypeAliases:
    """Test type aliases and imports."""

    def test_symbol_index_type(self) -> None:
        """Test SymbolIndex type alias."""
        symbol = sp.Symbol("x_1")
        metric = Mock(spec=SymbolicMetric)

        index: SymbolIndex = {symbol: metric}

        assert index[symbol] == metric
