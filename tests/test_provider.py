import threading

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx import specs
from dqx.common import DQXError, ResultKey
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import (
    ExtendedMetricProvider,
    MetricProvider,
    SymbolicMetric,
    SymbolicMetricBase,
)


class TestSymbolicMetric:
    """Test the SymbolicMetric dataclass."""

    def test_symbolic_metric_creation(self) -> None:
        """Test creating a SymbolicMetric instance."""
        symbol = sp.Symbol("x_1")

        def fn(key: ResultKey) -> Success:
            return Success(100.0)

        metric_spec = specs.NumRows()  # Real spec instead of mock

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

        def fn(key: ResultKey) -> Success:
            return Success(50.0)

        metric_spec = specs.Average("price")  # Real spec

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

        def fn(key: ResultKey) -> Success:
            return Success(42.0)

        metric_spec = specs.Average("value")

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

    def test_get_symbol_string_input(self, base: SymbolicMetricBase) -> None:
        """Test get_symbol() method with string input."""
        symbol = sp.Symbol("x_1")

        def fn(key: ResultKey) -> Success:
            return Success(42.0)

        metric_spec = specs.Average("value")

        # Add the symbolic metric
        base._registry._metrics.append(
            SymbolicMetric(name="test_metric", symbol=symbol, fn=fn, metric_spec=metric_spec)
        )
        base._registry._symbol_index[symbol] = base._registry._metrics[-1]

        # Get symbol using string
        result = base.get_symbol("x_1")
        assert result.name == "test_metric"
        assert result.symbol == symbol

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

    def test_remove_symbol(self, base: SymbolicMetricBase) -> None:
        """Test remove_symbol functionality."""
        # Add a symbol
        symbol = sp.Symbol("x_1")

        def fn(key: ResultKey) -> Success:
            return Success(100.0)

        metric_spec = specs.NumRows()

        base._registry._metrics.append(
            SymbolicMetric(name="test_metric", symbol=symbol, fn=fn, metric_spec=metric_spec)
        )
        base._registry._symbol_index[symbol] = base._registry._metrics[-1]

        # Verify it exists
        assert symbol in base._registry._symbol_index
        assert len(base._registry._metrics) == 1

        # Remove it
        base.remove_symbol(symbol)

        # Verify it's gone
        assert symbol not in base._registry._symbol_index
        assert len(base._registry._metrics) == 0


class TestMetricProvider:
    """Test the MetricProvider class."""

    @pytest.fixture
    def db(self) -> InMemoryMetricDB:
        """Create an in-memory database."""
        return InMemoryMetricDB()

    @pytest.fixture
    def provider(self, db: InMemoryMetricDB) -> MetricProvider:
        """Create a MetricProvider instance."""
        return MetricProvider(db, execution_id="test-exec-123")

    def test_init(self, db: InMemoryMetricDB) -> None:
        """Test MetricProvider initialization."""
        execution_id = "test-exec-123"
        provider = MetricProvider(db, execution_id)
        assert provider._db == db
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

    def test_metric_registration(self, provider: MetricProvider) -> None:
        """Test metric() method with real specs."""
        metric_spec = specs.Average("price")
        lag = 7
        dataset = "dataset1"

        symbol = provider.metric(metric_spec, lag, dataset)

        assert isinstance(symbol, sp.Symbol)
        assert symbol.name == "x_1"

        # Check that the symbol was registered
        registered_metric = provider.get_symbol(symbol)
        assert registered_metric.name == "average(price)"
        assert registered_metric.lag == 7
        assert registered_metric.dataset == dataset
        assert registered_metric.metric_spec == metric_spec

    def test_metric_with_defaults(self, provider: MetricProvider) -> None:
        """Test metric() method with default parameters."""
        metric_spec = specs.NumRows()

        symbol = provider.metric(metric_spec)

        registered_metric = provider.get_symbol(symbol)
        assert registered_metric.name == "num_rows()"
        assert registered_metric.lag == 0
        assert registered_metric.dataset is None

    def test_num_rows(self, provider: MetricProvider) -> None:
        """Test num_rows() method."""
        lag = 3
        dataset = "dataset1"

        result = provider.num_rows(lag, dataset)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "num_rows()"
        assert registered.lag == 3
        assert registered.dataset == "dataset1"
        assert isinstance(registered.metric_spec, specs.NumRows)

    def test_first(self, provider: MetricProvider) -> None:
        """Test first() method."""
        column = "test_column"

        result = provider.first(column)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "first(test_column)"
        assert registered.lag == 0
        assert registered.dataset is None
        assert isinstance(registered.metric_spec, specs.First)
        assert registered.metric_spec.parameters["column"] == column

    def test_average(self, provider: MetricProvider) -> None:
        """Test average() method."""
        column = "test_column"
        lag = 2
        dataset = "sales"

        result = provider.average(column, lag, dataset)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "average(test_column)"
        assert registered.lag == 2
        assert registered.dataset == "sales"
        assert isinstance(registered.metric_spec, specs.Average)

    def test_minimum(self, provider: MetricProvider) -> None:
        """Test minimum() method."""
        column = "test_column"

        result = provider.minimum(column)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "minimum(test_column)"
        assert isinstance(registered.metric_spec, specs.Minimum)

    def test_maximum(self, provider: MetricProvider) -> None:
        """Test maximum() method."""
        column = "test_column"

        result = provider.maximum(column)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "maximum(test_column)"
        assert isinstance(registered.metric_spec, specs.Maximum)

    def test_sum(self, provider: MetricProvider) -> None:
        """Test sum() method."""
        column = "test_column"

        result = provider.sum(column)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "sum(test_column)"
        assert isinstance(registered.metric_spec, specs.Sum)

    def test_null_count(self, provider: MetricProvider) -> None:
        """Test null_count() method."""
        column = "test_column"

        result = provider.null_count(column)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "null_count(test_column)"
        assert isinstance(registered.metric_spec, specs.NullCount)

    def test_variance(self, provider: MetricProvider) -> None:
        """Test variance() method."""
        column = "test_column"

        result = provider.variance(column)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "variance(test_column)"
        assert isinstance(registered.metric_spec, specs.Variance)

    def test_duplicate_count(self, provider: MetricProvider) -> None:
        """Test duplicate_count() method."""
        columns = ["col1", "col2"]

        result = provider.duplicate_count(columns)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "duplicate_count(col1,col2)"
        assert isinstance(registered.metric_spec, specs.DuplicateCount)
        assert registered.metric_spec.parameters["columns"] == ["col1", "col2"]

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

    def test_count_values(self, provider: MetricProvider) -> None:
        """Test count_values() method with different value types."""
        # Test with single string value
        result = provider.count_values("status", "active")
        registered = provider.get_symbol(result)
        assert isinstance(registered.metric_spec, specs.CountValues)
        assert "count_values" in registered.name

        # Test with list of strings
        result2 = provider.count_values("status", ["active", "pending"])
        registered2 = provider.get_symbol(result2)
        assert isinstance(registered2.metric_spec, specs.CountValues)

        # Test with integer
        result3 = provider.count_values("type_id", 1)
        registered3 = provider.get_symbol(result3)
        assert isinstance(registered3.metric_spec, specs.CountValues)

        # Test with boolean
        result4 = provider.count_values("is_active", True)
        registered4 = provider.get_symbol(result4)
        assert isinstance(registered4.metric_spec, specs.CountValues)

    def test_unique_count(self, provider: MetricProvider) -> None:
        """Test unique_count() method."""
        column = "product_id"

        result = provider.unique_count(column)

        assert isinstance(result, sp.Symbol)
        registered = provider.get_symbol(result)
        assert registered.name == "unique_count(product_id)"
        assert isinstance(registered.metric_spec, specs.UniqueCount)

    def test_metric_deduplication(self, provider: MetricProvider) -> None:
        """Test that identical metrics can be deduplicated."""
        # Create the same metric twice
        symbol1 = provider.average("price", lag=0, dataset="sales")
        symbol2 = provider.average("price", lag=0, dataset="sales")

        # They will have different symbols initially
        assert symbol1 != symbol2  # Different symbols are created

        # But they represent the same metric
        metric1 = provider.get_symbol(symbol1)
        metric2 = provider.get_symbol(symbol2)
        assert metric1.name == metric2.name
        assert metric1.lag == metric2.lag
        assert metric1.dataset == metric2.dataset

        # Different parameters should create different metrics
        symbol3 = provider.average("price", lag=1, dataset="sales")
        metric3 = provider.get_symbol(symbol3)
        assert metric3.lag != metric1.lag

    def test_create_metric_with_simple_specs(self, provider: MetricProvider) -> None:
        """Test create_metric with simple metric specs."""
        # Test with NumRows
        spec1 = specs.NumRows()
        symbol1 = provider.create_metric(spec1, lag=1, dataset="test")
        registered1 = provider.get_symbol(symbol1)
        assert registered1.name == "num_rows()"
        assert registered1.lag == 1
        assert registered1.dataset == "test"

        # Test with Average
        spec2 = specs.Average("price")
        symbol2 = provider.create_metric(spec2)
        registered2 = provider.get_symbol(symbol2)
        assert registered2.name == "average(price)"
        assert registered2.lag == 0
        assert registered2.dataset is None


class TestExtendedMetricProvider:
    """Test the ExtendedMetricProvider class."""

    @pytest.fixture
    def db(self) -> InMemoryMetricDB:
        """Create an in-memory database."""
        return InMemoryMetricDB()

    @pytest.fixture
    def provider(self, db: InMemoryMetricDB) -> MetricProvider:
        """Create a MetricProvider instance."""
        return MetricProvider(db, execution_id="test-exec-123")

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

    def test_day_over_day(self, provider: MetricProvider) -> None:
        """Test day_over_day method."""
        # Create a base metric
        base_metric = provider.average("price", dataset="sales")

        # Create day over day metric
        dod_symbol = provider.ext.day_over_day(base_metric, lag=2, dataset="sales")

        # Verify the DayOverDay metric was created correctly
        registered = provider.get_symbol(dod_symbol)
        assert registered.name == "dod(average(price))"
        assert registered.lag == 2
        assert registered.dataset == "sales"
        assert isinstance(registered.metric_spec, specs.DayOverDay)

        # Check that it has two required metrics (lag+0 and lag+1)
        assert len(registered.required_metrics) == 2

    def test_week_over_week(self, provider: MetricProvider) -> None:
        """Test week_over_week method."""
        # Create a base metric
        base_metric = provider.sum("revenue", dataset="sales")

        # Create week over week metric
        wow_symbol = provider.ext.week_over_week(base_metric, lag=1, dataset="sales")

        # Verify the WeekOverWeek metric was created correctly
        registered = provider.get_symbol(wow_symbol)
        assert registered.name == "wow(sum(revenue))"
        assert registered.lag == 1
        assert registered.dataset == "sales"
        assert isinstance(registered.metric_spec, specs.WeekOverWeek)

        # Check that it has two required metrics (lag+0 and lag+7)
        assert len(registered.required_metrics) == 2

    def test_stddev(self, provider: MetricProvider) -> None:
        """Test stddev method."""
        # Create a base metric
        base_metric = provider.maximum("temperature", dataset="weather")

        # Create stddev metric for 7 days starting from 2 days ago
        stddev_symbol = provider.ext.stddev(base_metric, offset=2, n=7, dataset="weather")

        # Verify the Stddev metric was created correctly
        registered = provider.get_symbol(stddev_symbol)
        assert registered.name == "stddev(maximum(temperature), offset=2, n=7)"
        assert registered.lag == 2  # lag should equal offset
        assert registered.dataset == "weather"
        assert isinstance(registered.metric_spec, specs.Stddev)

        # Check that it has 7 required metrics (one for each day in the window)
        assert len(registered.required_metrics) == 7

    def test_create_metric_with_extended_specs(self, provider: MetricProvider) -> None:
        """Test create_metric with extended metric specs."""
        # Test with DayOverDay spec
        base_spec = specs.Average("price")
        dod_spec = specs.DayOverDay.from_base_spec(base_spec)

        symbol = provider.create_metric(dod_spec, lag=3, dataset="sales")
        registered = provider.get_symbol(symbol)

        assert registered.name == "dod(average(price))"
        assert registered.lag == 3
        assert registered.dataset == "sales"
        assert isinstance(registered.metric_spec, specs.DayOverDay)

        # Test with WeekOverWeek spec
        wow_spec = specs.WeekOverWeek.from_base_spec(base_spec)
        symbol2 = provider.create_metric(wow_spec, lag=0, dataset="sales")
        registered2 = provider.get_symbol(symbol2)

        assert registered2.name == "wow(average(price))"
        assert isinstance(registered2.metric_spec, specs.WeekOverWeek)

        # Test with Stddev spec
        stddev_spec = specs.Stddev.from_base_spec(base_spec, offset=1, n=5)
        symbol3 = provider.create_metric(stddev_spec, lag=2, dataset="sales")
        registered3 = provider.get_symbol(symbol3)

        assert registered3.name == "stddev(average(price), offset=3, n=5)"  # offset=1 + lag=2 = 3
        assert registered3.lag == 3  # offset=1 + lag=2
        assert isinstance(registered3.metric_spec, specs.Stddev)

    def test_unsupported_extended_metric(self, provider: MetricProvider) -> None:
        """Test create_metric with unsupported extended metric type."""

        # Create a mock extended metric spec that's not supported
        class UnsupportedExtendedSpec:
            metric_type = "UnsupportedType"
            is_extended = True
            name = "unsupported()"

        with pytest.raises(ValueError, match="Unsupported extended metric type"):
            provider.create_metric(UnsupportedExtendedSpec())  # type: ignore


class TestLazyEvaluation:
    """Test lazy evaluation functionality."""

    @pytest.fixture
    def provider(self) -> MetricProvider:
        """Create a provider with in-memory database."""
        db = InMemoryMetricDB()
        return MetricProvider(db, "test-exec-123")

    def test_lazy_retrieval_function(self, provider: MetricProvider) -> None:
        """Test that metrics use lazy retrieval functions."""
        # Create a metric without dataset
        symbol = provider.average("price", lag=0, dataset=None)
        registered = provider.get_symbol(symbol)

        # The retrieval function should be set
        assert registered.fn is not None
        assert callable(registered.fn)

        # When called without dataset imputed, it should fail
        import datetime

        from dqx.common import ResultKey

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        result = registered.fn(key)

        # Result is a Failure, so we can get the error message
        assert isinstance(result, Failure)
        error_msg = result.failure()
        assert "Dataset not imputed" in error_msg

    def test_lazy_retrieval_with_imputed_dataset(self, provider: MetricProvider) -> None:
        """Test that lazy retrieval works after dataset imputation."""
        # Create a metric without dataset
        symbol = provider.average("price", lag=0, dataset=None)
        registered = provider.get_symbol(symbol)

        # Simulate dataset imputation by updating the symbolic metric
        registered.dataset = "sales"

        # Now the retrieval function should try to fetch from compute
        # (It will still fail because we don't have actual data, but differently)
        import datetime

        from dqx.common import ResultKey

        key = ResultKey(yyyy_mm_dd=datetime.date.today(), tags={})
        result = registered.fn(key)

        # The error will be different - no data rather than no dataset
        # Result is a Failure, but for a different reason now
        assert isinstance(result, Failure)
        error_msg = result.failure()
        assert "Dataset not imputed" not in error_msg


class TestSymbolDeduplication:
    """Test symbol deduplication functionality."""

    @pytest.fixture
    def provider(self) -> MetricProvider:
        """Create a provider with in-memory database."""
        db = InMemoryMetricDB()
        return MetricProvider(db, "test-exec-123")

    def test_build_deduplication_map(self, provider: MetricProvider) -> None:
        """Test building deduplication map for duplicate symbols."""
        import datetime

        from dqx.common import ResultKey

        # Create some duplicate symbols
        # Same metric, same effective date
        symbol1 = provider.average("price", lag=0, dataset="sales")
        symbol2 = provider.average("price", lag=1, dataset="sales")  # Different lag
        symbol3 = provider.average("price", lag=0, dataset="sales")  # Duplicate of symbol1

        # Build deduplication map for analysis date
        context_key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 16), tags={})
        dedup_map = provider.build_deduplication_map(context_key)

        # symbol3 should map to symbol1 (same metric, same effective date)
        assert symbol3 in dedup_map
        assert dedup_map[symbol3] == symbol1

        # symbol2 has different effective date due to lag, so no duplicate
        assert symbol2 not in dedup_map

    def test_prune_duplicate_symbols(self, provider: MetricProvider) -> None:
        """Test pruning duplicate symbols from provider."""
        # Create duplicate symbols
        symbol1 = provider.average("price", lag=0, dataset="sales")
        symbol2 = provider.average("price", lag=0, dataset="sales")  # Duplicate
        symbol3 = provider.sum("quantity", lag=0, dataset="sales")  # Different metric

        # Create substitution map
        substitutions = {symbol2: symbol1}

        # Prune duplicates
        provider.prune_duplicate_symbols(substitutions)

        # symbol1 and symbol3 should remain, symbol2 should be gone
        assert provider.get_symbol(symbol1) is not None
        assert provider.get_symbol(symbol3) is not None

        with pytest.raises(DQXError):
            provider.get_symbol(symbol2)

    def test_deduplicate_required_metrics(self, provider: MetricProvider) -> None:
        """Test updating required_metrics after deduplication."""
        # Create base metrics with one being a duplicate
        base1 = provider.average("price", lag=0, dataset="sales")
        base2 = provider.average("price", lag=0, dataset="sales")  # Duplicate
        base3 = provider.sum("quantity", lag=0, dataset="sales")

        # Create an extended metric that depends on these
        def complex_fn(key: ResultKey) -> Success:
            return Success(1.0)

        provider._registry._metrics.append(
            SymbolicMetric(
                name="complex_metric",
                symbol=sp.Symbol("x_99"),
                fn=complex_fn,
                metric_spec=specs.NumRows(),  # Just for testing
                required_metrics=[base1, base2, base3],
            )
        )

        # Create substitution map
        substitutions = {base2: base1}

        # Update required metrics
        provider.deduplicate_required_metrics(substitutions)

        # The complex metric should now depend on base1 (not base2) and base3
        complex_metric = [m for m in provider.metrics if m.name == "complex_metric"][0]
        assert base1 in complex_metric.required_metrics
        assert base2 not in complex_metric.required_metrics
        assert base3 in complex_metric.required_metrics
        assert complex_metric.required_metrics.count(base1) == 2  # base1 appears twice
