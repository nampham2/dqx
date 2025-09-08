"""Tests for SymbolTable implementation."""

import datetime

import pytest
import sympy as sp
from returns.result import Failure, Success

from dqx.common import Result, ResultKey, ResultKeyProvider, RetrievalFn
from dqx.provider import SymbolicMetric
from dqx.specs import Average, MetricSpec, NumRows as NumRowsSpec
from dqx.symbol_table import SymbolEntry, SymbolState, SymbolTable


class TestHelpers:
    """Helper methods for creating test data."""

    @staticmethod
    def create_retrieval_fn(value: float | str, success: bool = True) -> RetrievalFn:
        """Create a retrieval function that returns Success or Failure."""
        if success:
            if isinstance(value, float):
                return lambda k: Success(value)
            else:
                return lambda k: Success(float(value))
        return lambda k: Failure(str(value))

    @staticmethod
    def create_symbol_entry(
        symbol: str = "x_1",
        name: str = "test_metric",
        value: float = 42.0,
        dataset: str | None = "dataset1",
        state: SymbolState = "PENDING",
        dependencies: list[MetricSpec] | None = None,
        success: bool = True,
        retrieval_fn: RetrievalFn | None = None,
    ) -> SymbolEntry:
        """Factory for creating SymbolEntry with sensible defaults."""
        symbol_obj = sp.Symbol(symbol)
        if retrieval_fn is None:
            retrieval_fn = TestHelpers.create_retrieval_fn(value, success)
        deps = dependencies or [NumRowsSpec()]
        
        symbolic_metric = SymbolicMetric(
            symbol=symbol_obj,
            name=name,
            fn=retrieval_fn,
            key_provider=ResultKeyProvider(),
            dependencies=[(dep, ResultKeyProvider()) for dep in deps],
            datasets=[dataset] if dataset else [],
        )
        
        return SymbolEntry(
            symbolic_metric=symbolic_metric,
            dataset=dataset,
            result_key=ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={}),
            state=state,
        )


class TestSymbolEntry:
    """Test cases for SymbolEntry."""

    @pytest.fixture
    def test_date(self) -> datetime.date:
        """Fixed date for testing."""
        return datetime.date(2024, 1, 1)

    @pytest.fixture
    def basic_entry(self, test_date: datetime.date) -> SymbolEntry:
        """Basic symbol entry for testing."""
        return TestHelpers.create_symbol_entry()

    def test_creation_and_initial_state(self, basic_entry: SymbolEntry) -> None:
        """Test symbol entry creation and initial properties."""
        assert basic_entry.symbol == sp.Symbol("x_1")
        assert basic_entry.name == "test_metric"
        assert basic_entry.dataset == "dataset1"
        assert basic_entry.state == "PENDING"
        assert basic_entry.is_pending()
        assert not basic_entry.is_ready()
        assert not basic_entry.is_error()

    @pytest.mark.parametrize("initial_state,method,expected_state,check_method", [
        ("PENDING", "mark_ready", "READY", "is_ready"),
        ("READY", "mark_error", "ERROR", "is_error"),
    ])
    def test_state_transitions(
        self,
        initial_state: SymbolState,
        method: str,
        expected_state: SymbolState,
        check_method: str,
    ) -> None:
        """Test various state transitions."""
        entry = TestHelpers.create_symbol_entry(state=initial_state)
        
        if method == "mark_error":
            getattr(entry, method)("Test error")
            assert entry.value is not None
            result = entry.get_value()
            assert isinstance(result, Failure)
            assert result.failure() == "Test error"
        else:
            getattr(entry, method)()
        
        assert entry.state == expected_state
        assert getattr(entry, check_method)()

    def test_mark_success(self) -> None:
        """Test marking entry as successful."""
        entry = TestHelpers.create_symbol_entry(state="PENDING")
        entry.mark_success(99.0)
        
        assert entry.state == "READY"
        result = entry.get_value()
        assert isinstance(result, Success)
        assert result.unwrap() == 99.0

    @pytest.mark.parametrize("datasets,expected_success,error_contains", [
        (["dataset1", "dataset2"], False, "requires a SINGLE dataset but multiple available"),
        (["dataset1"], True, None),
        ([], False, "requires a dataset but none available"),
    ])
    def test_validate_dataset_no_requirements(
        self,
        datasets: list[str],
        expected_success: bool,
        error_contains: str | None,
    ) -> None:
        """Test dataset validation when no specific dataset required."""
        entry = TestHelpers.create_symbol_entry(dataset=None)
        result = entry.validate_dataset(datasets)
        
        if expected_success:
            assert isinstance(result, Success)
            assert result.unwrap() == datasets[0]
            assert entry.dataset == datasets[0]
        else:
            assert isinstance(result, Failure)
            if error_contains:
                assert error_contains in result.failure()

    @pytest.mark.parametrize("entry_dataset,available_datasets,expected_success", [
        ("dataset1", ["dataset1", "dataset2"], True),
        ("dataset1", ["dataset2", "dataset3"], False),
    ])
    def test_validate_dataset_with_requirements(
        self,
        entry_dataset: str,
        available_datasets: list[str],
        expected_success: bool,
    ) -> None:
        """Test dataset validation with specific dataset requirement."""
        entry = TestHelpers.create_symbol_entry(dataset=entry_dataset)
        result = entry.validate_dataset(available_datasets)
        
        if expected_success:
            assert isinstance(result, Success)
            assert result.unwrap() == entry_dataset
        else:
            assert isinstance(result, Failure)
            assert f"requires dataset '{entry_dataset}'" in result.failure()

    def test_get_value_returns_none(self) -> None:
        """Test get_value returns None when value is None."""
        entry = TestHelpers.create_symbol_entry()
        assert entry.get_value() is None


class TestSymbolTable:
    """Test cases for SymbolTable."""

    @pytest.fixture
    def symbol_table(self) -> SymbolTable:
        """Create an empty symbol table."""
        return SymbolTable()

    @pytest.fixture
    def test_date(self) -> datetime.date:
        """Fixed date for testing."""
        return datetime.date(2024, 1, 1)

    @pytest.fixture
    def populated_table(self, symbol_table: SymbolTable) -> SymbolTable:
        """Create a symbol table with multiple entries in different states."""
        entries_config = [
            ("x_0", "PENDING", "dataset1", 0.0),
            ("x_1", "READY", "dataset1", 1.0),
            ("x_2", "READY", "dataset2", 2.0),
            ("x_3", "ERROR", "dataset2", 3.0),
        ]
        
        for symbol, state, dataset, value in entries_config:
            entry = TestHelpers.create_symbol_entry(
                symbol=symbol,
                name=f"metric_{symbol}",
                value=value,
                dataset=dataset,
                state=state,  # type: ignore[arg-type]
            )
            if state == "READY" and symbol == "x_1":
                entry.value = Success(value)
            symbol_table.register(entry)
        
        return symbol_table

    def test_register_and_retrieve(self, symbol_table: SymbolTable) -> None:
        """Test registering and retrieving entries."""
        entry = TestHelpers.create_symbol_entry()
        symbol_table.register(entry)
        
        # Verify entry is stored and retrievable
        retrieved = symbol_table.get(entry.symbol)
        assert retrieved == entry
        
        # Verify indexes are updated
        dataset_entries = symbol_table.get_by_dataset("dataset1")
        assert len(dataset_entries) == 1
        assert dataset_entries[0] == entry

    def test_register_duplicate_fails(self, symbol_table: SymbolTable) -> None:
        """Test that registering duplicate symbol fails."""
        entry = TestHelpers.create_symbol_entry()
        symbol_table.register(entry)
        
        with pytest.raises(Exception, match="already registered"):
            symbol_table.register(entry)

    def test_register_from_provider(self, symbol_table: SymbolTable, test_date: datetime.date) -> None:
        """Test registering from SymbolicMetric."""
        symbolic_metric = SymbolicMetric(
            name="average_value",
            symbol=sp.Symbol("x_avg"),
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
        assert len(entry.metric_specs) == 1
        assert entry.metric_specs[0].name == "average(value)"

    @pytest.mark.parametrize("method,expected_count,filter_dataset", [
        ("get_all", 4, None),
        ("get_by_dataset", 2, "dataset1"),
        ("get_pending", 1, None),
        ("get_ready", 1, None),
        ("get_successful", 1, None),
    ])
    def test_get_methods(
        self,
        populated_table: SymbolTable,
        method: str,
        expected_count: int,
        filter_dataset: str | None,
    ) -> None:
        """Test various get methods with different filters."""
        get_method = getattr(populated_table, method)
        if filter_dataset and method in ["get_by_dataset", "get_pending"]:
            result = get_method(filter_dataset)
        else:
            result = get_method()
        
        assert len(result) == expected_count

    def test_update_state(self, symbol_table: SymbolTable) -> None:
        """Test updating symbol state."""
        entry = TestHelpers.create_symbol_entry()
        symbol_table.register(entry)
        
        symbol_table.update_state(entry.symbol, "READY")
        assert entry.state == "READY"
        
        # Test non-existent symbol
        with pytest.raises(Exception, match="not found"):
            symbol_table.update_state(sp.Symbol("x_999"), "READY")

    @pytest.mark.parametrize("method,error_msg", [
        ("mark_dataset_failed", "Dataset dataset1 failed: Test failure"),
        ("mark_dataset_ready", None),
    ])
    def test_dataset_state_management(
        self,
        symbol_table: SymbolTable,
        method: str,
        error_msg: str | None,
    ) -> None:
        """Test dataset-based state management."""
        # Register multiple entries for same dataset
        for i in range(3):
            entry = TestHelpers.create_symbol_entry(
                symbol=f"x_{i}",
                name=f"metric_{i}",
                dataset="dataset1",
            )
            symbol_table.register(entry)
        
        # Apply dataset state change
        if method == "mark_dataset_failed":
            getattr(symbol_table, method)("dataset1", "Test failure")
        else:
            getattr(symbol_table, method)("dataset1")
        
        # Verify all entries updated
        for entry in symbol_table.get_by_dataset("dataset1"):
            if error_msg:
                assert entry.is_error()
                result = entry.get_value()
                assert isinstance(result, Failure)
                assert error_msg in result.failure()
            else:
                assert entry.state == "READY"

    @pytest.mark.parametrize("entry_state,retrieval_result,expected_success", [
        ("READY", Success(42.0), True),
        ("PENDING", Success(42.0), False),
        ("READY", Failure("Computation failed"), False),
        ("ERROR", Success(42.0), False),
    ])
    def test_evaluate_symbol(
        self,
        symbol_table: SymbolTable,
        test_date: datetime.date,
        entry_state: SymbolState,
        retrieval_result: Result[float, str],
        expected_success: bool,
    ) -> None:
        """Test symbol evaluation in various states."""
        entry = TestHelpers.create_symbol_entry(
            state=entry_state,
            retrieval_fn=lambda k: retrieval_result
        )
        if entry_state == "ERROR":
            entry.value = Failure("Previous error")
        symbol_table.register(entry)
        
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        result = symbol_table.evaluate_symbol(entry.symbol, key)
        
        if expected_success:
            assert isinstance(result, Success)
            assert result.unwrap() == 42.0
        else:
            assert isinstance(result, Failure)

    def test_evaluate_symbol_exception_handling(
        self,
        symbol_table: SymbolTable,
        test_date: datetime.date,
    ) -> None:
        """Test exception handling during evaluation."""
        def failing_retrieval(k: ResultKey) -> Result[float, str]:
            raise ValueError("Unexpected error")
        
        entry = TestHelpers.create_symbol_entry(
            state="READY",
            retrieval_fn=failing_retrieval
        )
        symbol_table.register(entry)
        
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        result = symbol_table.evaluate_symbol(entry.symbol, key)
        
        assert isinstance(result, Failure)
        assert "Failed to evaluate symbol" in result.failure()
        assert entry.is_error()

    def test_evaluate_ready_symbols(self, populated_table: SymbolTable, test_date: datetime.date) -> None:
        """Test evaluating all ready symbols."""
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        results = populated_table.evaluate_ready_symbols(key)
        
        # Should only evaluate READY symbols (x_2) since x_1 already has a Success value
        assert len(results) == 1
        assert sp.Symbol("x_2") in results

    def test_get_required_metrics(self, symbol_table: SymbolTable) -> None:
        """Test getting required metrics across datasets."""
        # Register entries with different specs and datasets
        specs_config: list[tuple[MetricSpec, str, SymbolState]] = [
            (NumRowsSpec(), "dataset1", "PENDING"),
            (Average("col1"), "dataset1", "PENDING"),
            (Average("col2"), "dataset2", "PENDING"),
            (NumRowsSpec(), "dataset2", "READY"),
        ]
        
        for i, (spec, dataset, state) in enumerate(specs_config):
            entry = TestHelpers.create_symbol_entry(
                symbol=f"x_{i}",
                dataset=dataset,
                state=state,
                dependencies=[spec],
            )
            symbol_table.register(entry)
        
        # Test getting all metrics
        all_metrics = symbol_table.get_required_metrics()
        assert len(all_metrics) == 3  # Unique specs from PENDING entries
        
        # Test dataset-specific metrics
        ds1_metrics = symbol_table.get_required_metrics("dataset1")
        assert len(ds1_metrics) == 2
        
        ds2_metrics = symbol_table.get_required_metrics("dataset2")
        assert len(ds2_metrics) == 1


    def test_clear(self, populated_table: SymbolTable) -> None:
        """Test clearing the symbol table."""
        assert len(populated_table.get_all()) > 0
        
        populated_table.clear()
        
        assert len(populated_table.get_all()) == 0
        assert len(populated_table._by_dataset) == 0

    def test_repr(self, populated_table: SymbolTable) -> None:
        """Test string representation of symbol table."""
        repr_str = str(populated_table)
        
        assert "total=4" in repr_str
        assert "pending=1" in repr_str
        assert "ready=1" in repr_str
        assert "successful=1" in repr_str
        assert "error=1" in repr_str

    def test_bind_symbol_to_dataset(self, symbol_table: SymbolTable) -> None:
        """Test binding symbol to dataset updates indexes."""
        entry = TestHelpers.create_symbol_entry(dataset=None)
        symbol_table.register(entry)
        
        # Initially not in any dataset index
        assert len(symbol_table.get_by_dataset("new_dataset")) == 0
        
        # Bind and validate
        result = entry.validate_dataset(["new_dataset"])
        assert isinstance(result, Success)
        assert entry.dataset == "new_dataset"
        
        # Update table index
        symbol_table.bind_symbol_to_dataset(entry.symbol, "new_dataset")
        
        # Verify index updated
        assert entry.symbol in symbol_table._by_dataset["new_dataset"]

    def test_get_required_analyzers(self, symbol_table: SymbolTable) -> None:
        """Test getting required analyzers for pending symbols."""
        # Create entries with different analyzers
        state_configs: list[tuple[SymbolState, str]] = [
            ("PENDING", "dataset1"),
            ("PENDING", "dataset1"),
            ("READY", "dataset2"),
        ]
        for i, (state, dataset) in enumerate(state_configs):
            entry = TestHelpers.create_symbol_entry(
                symbol=f"x_{i}",
                dataset=dataset,
                state=state,
                dependencies=[NumRowsSpec()] if i < 2 else [Average("col")],
            )
            symbol_table.register(entry)
        
        # Get all required analyzers (only from PENDING)
        analyzers = symbol_table.get_required_analyzers()
        assert len(analyzers) == 2
        
        # Get for specific dataset
        analyzers_ds1 = symbol_table.get_required_analyzers("dataset1")
        assert len(analyzers_ds1) == 2


class TestIntegration:
    """Integration tests for SymbolTable workflows."""

    def test_full_evaluation_workflow(self) -> None:
        """Test complete workflow from registration to evaluation."""
        table = SymbolTable()
        test_date = datetime.date(2024, 1, 1)
        key = ResultKey(yyyy_mm_dd=test_date, tags={})
        
        # Register multiple interdependent entries
        entries = []
        for i in range(3):
            entry = TestHelpers.create_symbol_entry(
                symbol=f"x_{i}",
                name=f"metric_{i}",
                value=float(i * 10),
                dataset="dataset1",
                state="PENDING" if i < 2 else "READY",
            )
            entries.append(entry)
            table.register(entry)
        
        # Mark dataset ready
        table.mark_dataset_ready("dataset1")
        
        # Evaluate ready symbols
        results = table.evaluate_ready_symbols(key)
        
        # Verify results
        assert len(results) == 3  # All should be ready now
        for i, (symbol, result) in enumerate(results.items()):
            assert isinstance(result, Success)
            expected_value = float(int(str(symbol).split("_")[1]) * 10)
            assert result.unwrap() == expected_value

    def test_error_propagation_workflow(self) -> None:
        """Test error propagation through dataset failure."""
        table = SymbolTable()
        
        # Register entries for a dataset
        for i in range(3):
            entry = TestHelpers.create_symbol_entry(
                symbol=f"x_{i}",
                dataset="failing_dataset",
            )
            table.register(entry)
        
        # Mark dataset as failed
        table.mark_dataset_failed("failing_dataset", "Database connection failed")
        
        # All entries should be in error state
        for entry in table.get_by_dataset("failing_dataset"):
            assert entry.is_error()
            result = entry.get_value()
            assert isinstance(result, Failure)
            assert "Database connection failed" in result.failure()
