"""Tests for SymbolTable implementation."""

import datetime
import pytest
import sympy as sp
from returns.maybe import Some
from returns.result import Failure, Success

from dqx.common import ResultKey, RetrievalFn, Result
from dqx.ops import NumRows, Op
from dqx.provider import SymbolicMetric
from dqx.specs import Average, MetricSpec, NumRows as NumRowsSpec
from dqx.symbol_table import SymbolEntry, SymbolState, SymbolTable


class TestSymbolEntry:
    """Test cases for SymbolEntry."""
    
    @pytest.fixture
    def test_date(self) -> datetime.date:
        """Fixed date for testing."""
        return datetime.date(2024, 1, 1)
    
    def test_symbol_entry_creation(self, test_date: datetime.date) -> None:
        """Test creating a symbol entry."""
        symbol = sp.Symbol("x_1")
        entry = SymbolEntry(
            symbol=symbol,
            name="test_metric",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[NumRows()],
            retrieval_fn=lambda k: Success(42.0),
        )
        
        assert entry.symbol == symbol
        assert entry.name == "test_metric"
        assert entry.dataset == "dataset1"
        assert entry.state == "PENDING"
        assert entry.is_pending()
        assert not entry.is_ready()
        assert not entry.is_error()
    
    def test_state_transitions(self, test_date: datetime.date) -> None:
        """Test state transitions for symbol entry."""
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="test",
            dataset=None,
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        # Test mark_ready
        entry.mark_ready()
        assert entry.state == "READY"
        assert entry.is_ready()
        
        # Test mark_provided
        entry.mark_provided()
        assert entry.state == "PROVIDED"
        assert entry.is_ready()
        
        # Test mark_error
        entry.mark_error("Test error")
        assert entry.state == "ERROR"
        assert entry.is_error()
        assert isinstance(entry.value, Some)
        result = entry.get_value()
        assert isinstance(result, Failure)
        assert result.failure() == "Test error"
        
        # Test mark_success
        entry.mark_success(42.0)
        assert entry.state == "PROVIDED"
        result = entry.get_value()
        assert isinstance(result, Success)
        assert result.unwrap() == 42.0
    
    def test_validate_dataset_no_requirements(self, test_date: datetime.date) -> None:
        """Test dataset validation when no specific dataset required."""
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="test",
            dataset=None,
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        # Should bind to first available dataset
        result = entry.validate_dataset(["dataset1", "dataset2"])
        assert isinstance(result, Success)
        assert entry.dataset == "dataset1"
        
        # Should fail with no datasets
        entry.dataset = None  # Reset
        result = entry.validate_dataset([])
        assert isinstance(result, Failure)
        assert "requires a dataset but none available" in result.failure()
    
    def test_validate_dataset_with_requirements(self, test_date: datetime.date) -> None:
        """Test dataset validation with specific dataset requirement."""
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="test",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        # Should succeed with required dataset
        result = entry.validate_dataset(["dataset1", "dataset2", "dataset3"])
        assert isinstance(result, Success)
        
        # Should fail with missing dataset
        result = entry.validate_dataset(["dataset2", "dataset3"])
        assert isinstance(result, Failure)
        assert "requires dataset 'dataset1'" in result.failure()
    
    def test_get_value_returns_none(self, test_date: datetime.date) -> None:
        """Test get_value returns None when value is Nothing."""
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="test",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        # Value is Nothing by default
        assert entry.get_value() is None


class TestSymbolTable:
    """Test cases for SymbolTable."""
    
    @pytest.fixture
    def test_date(self) -> datetime.date:
        """Fixed date for testing."""
        return datetime.date(2024, 1, 1)
    
    @pytest.fixture
    def symbol_table(self) -> SymbolTable:
        """Create an empty symbol table."""
        return SymbolTable()
    
    @pytest.fixture
    def sample_entry(self, test_date: datetime.date) -> SymbolEntry:
        """Create a sample symbol entry."""
        return SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="num_rows",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[NumRows()],
            retrieval_fn=lambda k: Success(100.0),
        )
    
    def test_register_entry(self, symbol_table: SymbolTable, sample_entry: SymbolEntry) -> None:
        """Test registering a symbol entry."""
        symbol_table.register(sample_entry)
        
        # Check entry is stored
        retrieved = symbol_table.get(sample_entry.symbol)
        assert retrieved == sample_entry
        
        # Check indexes are updated
        dataset_entries = symbol_table.get_by_dataset("dataset1")
        assert len(dataset_entries) == 1
        assert dataset_entries[0] == sample_entry
        
        # Check evaluation order
        assert sample_entry.symbol in symbol_table._evaluation_order
    
    def test_register_duplicate_fails(self, symbol_table: SymbolTable, sample_entry: SymbolEntry) -> None:
        """Test that registering duplicate symbol fails."""
        symbol_table.register(sample_entry)
        
        with pytest.raises(Exception) as exc_info:
            symbol_table.register(sample_entry)
        assert "already registered" in str(exc_info.value)
    
    def test_register_from_provider(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test registering from SymbolicMetric."""
        from dqx.common import ResultKeyProvider
        
        symbolic_metric = SymbolicMetric(
            name="average_value",
            symbol=sp.Symbol("x_2"),
            fn=lambda k: Success(50.0),
            key_provider=ResultKeyProvider(),
            dependencies=[(Average("value"), ResultKeyProvider())],
            datasets=["dataset1"],
        )
        
        key = ResultKey(yyyy_mm_dd=test_date, tags={"test": "value"})
        registered_symbol = symbol_table.register_from_provider(symbolic_metric, key)
        
        assert registered_symbol == symbolic_metric.symbol
        
        entry = symbol_table.get(registered_symbol)
        assert entry is not None
        assert entry.name == "average_value"
        assert entry.dataset == "dataset1"
        assert entry.metric_spec.name == "average(value)"  # type: ignore
    
    def test_get_methods(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test various get methods."""
        # Register multiple entries
        entries = []
        for i in range(3):
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1" if i < 2 else "dataset2",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=NumRowsSpec(),
                ops=[],
                retrieval_fn=lambda k: Success(float(i)),
            )
            entries.append(entry)
            symbol_table.register(entry)
        
        # Mark different states
        entries[0].mark_ready()
        entries[1].mark_success(1.0)  # This sets both state="PROVIDED" and value
        entries[2].state = "PENDING"
        
        # Test get_all
        all_entries = symbol_table.get_all()
        assert len(all_entries) == 3
        
        # Test get_by_dataset
        dataset1_entries = symbol_table.get_by_dataset("dataset1")
        assert len(dataset1_entries) == 2
        
        # Test get_pending
        pending = symbol_table.get_pending()
        assert len(pending) == 1
        assert pending[0] == entries[2]
        
        # Test get_pending with dataset filter
        pending_ds1 = symbol_table.get_pending("dataset1")
        assert len(pending_ds1) == 0
        
        # Test get_ready
        ready = symbol_table.get_ready()
        assert len(ready) == 1
        assert ready[0] == entries[0]
        
        # Test get_successful
        successful = symbol_table.get_successful()
        assert len(successful) == 1
        assert successful[0] == entries[1]
    
    def test_state_management(self, symbol_table: SymbolTable, sample_entry: SymbolEntry) -> None:
        """Test state management methods."""
        symbol_table.register(sample_entry)
        
        # Test update_state
        symbol_table.update_state(sample_entry.symbol, "READY")
        assert sample_entry.state == "READY"
        
        # Test update_state for non-existent symbol
        with pytest.raises(Exception) as exc_info:
            symbol_table.update_state(sp.Symbol("x_999"), "READY")
        assert "not found" in str(exc_info.value)
    
    def test_dataset_state_management(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test dataset-based state management."""
        # Register multiple entries for same dataset
        for i in range(3):
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=NumRowsSpec(),
                ops=[],
                retrieval_fn=lambda k: Success(1.0),
            )
            symbol_table.register(entry)
        
        # Test mark_dataset_failed
        symbol_table.mark_dataset_failed("dataset1", "Test failure")
        
        for entry in symbol_table.get_by_dataset("dataset1"):
            assert entry.is_error()
            result = entry.get_value()
            assert isinstance(result, Failure)
            assert "Dataset dataset1 failed: Test failure" in result.failure()
        
        # Reset and test mark_dataset_ready
        symbol_table.clear()
        for i in range(3):
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=NumRowsSpec(),
                ops=[],
                retrieval_fn=lambda k: Success(1.0),
            )
            symbol_table.register(entry)
        
        symbol_table.mark_dataset_ready("dataset1")
        
        for entry in symbol_table.get_by_dataset("dataset1"):
            assert entry.state == "READY"
    
    def test_evaluate_symbol(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test symbol evaluation."""
        # Test successful evaluation
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="test",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(42.0),
            state="READY",
        )
        symbol_table.register(entry)
        
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        result = symbol_table.evaluate_symbol(entry.symbol, key)
        
        assert isinstance(result, Success)
        assert result.unwrap() == 42.0
        assert entry.state == "PROVIDED"
        
        # Test evaluation of non-existent symbol
        result = symbol_table.evaluate_symbol(sp.Symbol("x_999"), key)
        assert isinstance(result, Failure)
        assert "not found" in result.failure()
        
        # Test evaluation of not-ready symbol
        entry2 = SymbolEntry(
            symbol=sp.Symbol("x_2"),
            name="test2",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
            state="PENDING",
        )
        symbol_table.register(entry2)
        
        result = symbol_table.evaluate_symbol(entry2.symbol, key)
        assert isinstance(result, Failure)
        assert "not ready for evaluation" in result.failure()
        
        # Test evaluation that returns failure
        entry3 = SymbolEntry(
            symbol=sp.Symbol("x_3"),
            name="test3",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Failure("Computation failed"),
            state="READY",
        )
        symbol_table.register(entry3)
        
        result = symbol_table.evaluate_symbol(entry3.symbol, key)
        assert isinstance(result, Failure)
        assert result.failure() == "Computation failed"
        assert entry3.is_error()
    
    def test_evaluate_ready_symbols(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test evaluating all ready symbols."""
        # Register symbols with different states
        for i in range(5):
            state: SymbolState = "READY" if i % 2 == 0 else "PENDING"
            # Create a retrieval function with proper typing
            def make_retrieval_fn(value: float) -> RetrievalFn:
                return lambda k: Success(value)
            
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=NumRowsSpec(),
                ops=[],
                retrieval_fn=make_retrieval_fn(float(i)),
                state=state,
            )
            symbol_table.register(entry)
        
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        results = symbol_table.evaluate_ready_symbols(key)
        
        # Should only evaluate ready symbols (0, 2, 4)
        assert len(results) == 3
        assert sp.Symbol("x_0") in results
        assert sp.Symbol("x_2") in results
        assert sp.Symbol("x_4") in results
        
        # Check all results are successful
        for symbol, result in results.items():
            assert isinstance(result, Success)
    
    def test_get_required_metrics(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test getting required metrics."""
        # Register entries with different specs
        specs: list[MetricSpec] = [NumRowsSpec(), Average("col1"), Average("col2"), NumRowsSpec()]
        
        for i, spec in enumerate(specs):
            state: SymbolState = "PENDING" if i < 3 else "PROVIDED"
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1" if i < 2 else "dataset2",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=spec,
                ops=[],
                retrieval_fn=lambda k: Success(1.0),
                state=state,
            )
            symbol_table.register(entry)
        
        # Get all required metrics
        all_metrics = symbol_table.get_required_metrics()
        assert len(all_metrics) == 3  # NumRows, Average(col1), Average(col2)
        
        # Get required metrics for dataset1
        ds1_metrics = symbol_table.get_required_metrics("dataset1")
        assert len(ds1_metrics) == 2  # NumRows, Average(col1)
        
        # Get required metrics for dataset2
        ds2_metrics = symbol_table.get_required_metrics("dataset2")
        assert len(ds2_metrics) == 1  # Average(col2)
    
    def test_validate_datasets(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test dataset validation."""
        # Register entries with different dataset requirements
        entry1 = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="metric_1",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        entry2 = SymbolEntry(
            symbol=sp.Symbol("x_2"),
            name="metric_2",
            dataset="dataset3",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        symbol_table.register(entry1)
        symbol_table.register(entry2)
        
        # Test with all datasets available
        errors = symbol_table.validate_datasets(["dataset1", "dataset2", "dataset3"])
        assert len(errors) == 0
        
        # Test with missing datasets
        errors = symbol_table.validate_datasets(["dataset2"])
        assert len(errors) == 2
        assert any("dataset1" in error for error in errors)
        assert any("dataset3" in error for error in errors)
    
    def test_clear(self, symbol_table: SymbolTable, sample_entry: SymbolEntry) -> None:
        """Test clearing the symbol table."""
        symbol_table.register(sample_entry)
        
        assert len(symbol_table.get_all()) == 1
        
        symbol_table.clear()
        
        assert len(symbol_table.get_all()) == 0
        assert len(symbol_table._by_dataset) == 0
        assert len(symbol_table._by_metric) == 0
        assert len(symbol_table._evaluation_order) == 0
    
    def test_repr(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test string representation."""
        # Empty table
        assert "total=0" in str(symbol_table)
        
        # Add entries with different states
        states: list[SymbolState] = ["PENDING", "READY", "PROVIDED", "ERROR"]
        for i, state in enumerate(states):
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=NumRowsSpec(),
                ops=[],
                retrieval_fn=lambda k: Success(1.0),
                state=state,
            )
            if state == "PROVIDED":
                entry.value = Some(Success(1.0))
            elif state == "ERROR":
                entry.value = Some(Failure("Error"))
            
            symbol_table.register(entry)
        
        repr_str = str(symbol_table)
        assert "total=4" in repr_str
        assert "pending=1" in repr_str
        assert "ready=1" in repr_str
        assert "successful=1" in repr_str
        assert "error=1" in repr_str
    
    def test_register_with_metric_spec(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test registering entry with metric_spec updates _by_metric index."""
        spec = Average("col1")
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="average_col1",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=spec,
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        symbol_table.register(entry)
        
        # Check that _by_metric index is updated
        assert "average(col1)" in symbol_table._by_metric
        assert entry.symbol in symbol_table._by_metric["average(col1)"]
    
    def test_register_symbol_for_check(self, symbol_table: SymbolTable) -> None:
        """Test tracking which check owns a symbol."""
        symbol1 = sp.Symbol("x_1")
        symbol2 = sp.Symbol("x_2")
        
        symbol_table.register_symbol_for_check(symbol1, "quality_check_1")
        symbol_table.register_symbol_for_check(symbol2, "quality_check_1")
        symbol_table.register_symbol_for_check(symbol1, "quality_check_2")
        
        # Test get_symbols_for_check
        check1_symbols = symbol_table.get_symbols_for_check("quality_check_1")
        assert symbol1 in check1_symbols
        assert symbol2 in check1_symbols
        assert len(check1_symbols) == 2
        
        check2_symbols = symbol_table.get_symbols_for_check("quality_check_2")
        assert symbol1 in check2_symbols
        assert len(check2_symbols) == 1
        
        # Test non-existent check returns empty list
        check3_symbols = symbol_table.get_symbols_for_check("nonexistent_check")
        assert len(check3_symbols) == 0
    
    def test_evaluate_symbol_error_state(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test evaluating a symbol that's already in ERROR state."""
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="test",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(42.0),
            state="ERROR",
        )
        entry.value = Some(Failure("Previous error"))
        symbol_table.register(entry)
        
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        result = symbol_table.evaluate_symbol(entry.symbol, key)
        
        assert isinstance(result, Failure)
        assert result.failure() == "Previous error"
    
    def test_evaluate_symbol_exception_handling(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test exception handling in evaluate_symbol."""
        def failing_retrieval_fn(k: ResultKey) -> Result[float, str]:
            raise ValueError("Unexpected error during retrieval")
        
        entry = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="test",
            dataset="dataset1",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=failing_retrieval_fn,
            state="READY",
        )
        symbol_table.register(entry)
        
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        result = symbol_table.evaluate_symbol(entry.symbol, key)
        
        assert isinstance(result, Failure)
        assert "Failed to evaluate symbol x_1: Unexpected error during retrieval" in result.failure()
        assert entry.is_error()
    
    def test_get_required_analyzers(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test getting required analyzers for pending symbols."""
        # Register entries with different ops
        for i in range(3):
            ops: list[Op] = [NumRows()] if i < 2 else []
            state: SymbolState = "PENDING" if i < 2 else "PROVIDED"
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=NumRowsSpec(),
                ops=ops,
                retrieval_fn=lambda k: Success(1.0),
                state=state,
            )
            symbol_table.register(entry)
        
        # Get all required analyzers (should only include pending symbols)
        analyzers = symbol_table.get_required_analyzers()
        assert len(analyzers) == 2  # Two NumRows from pending symbols
        
        # Get required analyzers for specific dataset
        analyzers_ds1 = symbol_table.get_required_analyzers("dataset1")
        assert len(analyzers_ds1) == 2
    
    def test_build_dependency_graph(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test building dependency graph."""
        # Register multiple entries
        for i in range(3):
            entry = SymbolEntry(
                symbol=sp.Symbol(f"x_{i}"),
                name=f"metric_{i}",
                dataset="dataset1",
                result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
                metric_spec=NumRowsSpec(),
                ops=[],
                retrieval_fn=lambda k: Success(1.0),
            )
            symbol_table.register(entry)
        
        # Build dependency graph
        dep_graph = symbol_table.build_dependency_graph()
        
        # Check that all symbols are in the graph
        assert len(dep_graph) == 3
        for i in range(3):
            symbol = sp.Symbol(f"x_{i}")
            assert symbol in dep_graph
            # Currently all dependencies are empty
            assert dep_graph[symbol] == set()
    
    def test_validate_datasets_with_binding(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test dataset validation that binds None datasets and updates indexes."""
        # Register entry with no dataset requirement
        entry1 = SymbolEntry(
            symbol=sp.Symbol("x_1"),
            name="metric_1",
            dataset=None,  # No specific dataset
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        # Register entry with specific dataset
        entry2 = SymbolEntry(
            symbol=sp.Symbol("x_2"),
            name="metric_2",
            dataset="dataset2",
            result_key=ResultKey(yyyy_mm_dd=test_date, tags={}),
            metric_spec=NumRowsSpec(),
            ops=[],
            retrieval_fn=lambda k: Success(1.0),
        )
        
        symbol_table.register(entry1)
        symbol_table.register(entry2)
        
        # Validate with available datasets
        errors = symbol_table.validate_datasets(["dataset1", "dataset2"])
        assert len(errors) == 0
        
        # Check that entry1 was bound to dataset1
        assert entry1.dataset == "dataset1"
        
        # Check that indexes were updated
        assert entry1.symbol in symbol_table._by_dataset["dataset1"]
        assert entry2.symbol in symbol_table._by_dataset["dataset2"]
    
    def test_register_from_provider_with_different_keys(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test registering from provider when keys differ (provided metric logic)."""
        from dqx.common import ResultKeyProvider
        
        # Create a custom key provider that generates different keys
        class CustomKeyProvider(ResultKeyProvider):
            def create(self, key: ResultKey) -> ResultKey:
                # Return a different key
                return ResultKey(yyyy_mm_dd=test_date, tags={"custom": "value"})
        
        symbolic_metric = SymbolicMetric(
            name="custom_metric",
            symbol=sp.Symbol("x_custom"),
            fn=lambda k: Success(99.0),
            key_provider=CustomKeyProvider(),
            dependencies=[(Average("value"), ResultKeyProvider())],
            datasets=["dataset1"],
        )
        
        original_key = ResultKey(yyyy_mm_dd=test_date, tags={"test": "value"})
        registered_symbol = symbol_table.register_from_provider(symbolic_metric, original_key)
        
        entry = symbol_table.get(registered_symbol)
        assert entry is not None
        # Check that it's marked as PROVIDED since keys differ
        assert entry.state == "PROVIDED"
