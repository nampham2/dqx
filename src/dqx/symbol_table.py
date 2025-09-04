"""Symbol table implementation for managing computation symbols and their metadata."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from dqx.common import DQXError, ResultKey, ResultKeyProvider, RetrievalFn
from dqx.ops import Op
from dqx.provider import SymbolicMetric
from dqx.specs import MetricSpec

# Type definitions
SymbolState = Literal["PENDING", "READY", "PROVIDED", "ERROR"]


@dataclass
class SymbolEntry:
    """Complete metadata for a symbol in the computation graph.
    
    Represents a single symbol in the computation system, tracking its state,
    dependencies, and computation requirements throughout its lifecycle.
    
    Attributes:
        symbol: The sympy Symbol instance representing this entry.
        name: Human-readable name for the symbol.
        dataset: The dataset name this symbol is associated with. None means
            the symbol will be bound to a dataset when one becomes available.
        result_key: The ResultKey used to store/retrieve computed values.
        metric_spec: The MetricSpec defining how to compute this symbol.
        ops: List of analyzer operations required for this symbol.
        retrieval_fn: Function to retrieve the computed value.
        dependencies: List of (MetricSpec, ResultKeyProvider) tuples representing
            other metrics this symbol depends on.
        value: Maybe container holding the computed Result (Success/Failure).
        state: Current state in the lifecycle: PENDING → READY → PROVIDED/ERROR.
        tags: Additional metadata tags for categorization.
    
    State Transitions:
        PENDING: Initial state, waiting for dataset availability.
        READY: Dataset available, ready for evaluation.
        PROVIDED: Successfully computed or externally provided.
        ERROR: Computation failed.
    """
    
    # Core identity
    symbol: sp.Symbol
    name: str
    
    # Data source information
    dataset: str | None
    result_key: ResultKey | None
    
    # Computation metadata
    metric_spec: MetricSpec | None
    ops: list[Op]  # Analyzer operations
    retrieval_fn: RetrievalFn
    
    # Dependencies
    dependencies: list[tuple[MetricSpec, ResultKeyProvider]] = field(default_factory=list)
    
    # State tracking
    value: Maybe[Result[float, str]] = field(default_factory=lambda: Nothing)
    state: SymbolState = "PENDING"
    
    # Additional properties
    tags: list[str] = field(default_factory=list)
    
    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied."""
        return self.state in ["READY", "PROVIDED"]
    
    def is_pending(self) -> bool:
        """Check if this symbol is pending computation."""
        return self.state == "PENDING"
    
    def is_error(self) -> bool:
        """Check if this symbol has an error."""
        return self.state == "ERROR"
    
    def mark_provided(self) -> None:
        """Mark this symbol as externally provided."""
        self.state = "PROVIDED"
    
    def mark_ready(self) -> None:
        """Mark this symbol as ready for evaluation."""
        self.state = "READY"
    
    def mark_error(self, error: str) -> None:
        """Mark this symbol as failed."""
        self.state = "ERROR"
        self.value = Some(Failure(error))
    
    def mark_success(self, value: float) -> None:
        """Mark this symbol as successfully computed."""
        self.state = "PROVIDED"
        self.value = Some(Success(value))
    
    def get_value(self) -> Result[float, str] | None:
        """Get the computed value if available."""
        if isinstance(self.value, Some):
            return self.value.unwrap()
        return None
    
    def validate_dataset(self, available_datasets: list[str]) -> Result[None, str]:
        """Validate that required dataset is available."""
        if self.dataset is None:
            # Symbol requires a dataset but none specified - will bind to first available
            if not available_datasets:
                return Failure(
                    f"Symbol {self.symbol} requires a dataset but none available"
                )
            # Bind to the first available dataset
            self.dataset = available_datasets[0]
        elif self.dataset not in available_datasets:
            # Symbol has specific dataset requirement
            return Failure(
                f"Symbol {self.symbol} requires dataset '{self.dataset}' "
                f"but only {available_datasets} available"
            )
        return Success(None)


class SymbolTable:
    """Central registry for all symbols and their computation metadata.
    
    The SymbolTable manages the lifecycle of all symbols in the computation system,
    tracking their states, dependencies, and relationships. It provides efficient
    lookup by symbol, dataset, metric, and state, enabling coordinated evaluation
    of complex symbolic expressions.
    
    The table maintains several indexes for efficient queries:
    - Direct symbol lookup via _entries
    - Dataset-based lookup via _by_dataset
    - Metric-based lookup via _by_metric
    - Check ownership tracking via _check_symbols
    
    Symbol Lifecycle:
        1. Symbol registered with metadata (PENDING state)
        2. Dataset becomes available → mark_dataset_success() (READY state)
        3. Symbol evaluated → evaluate_symbol() (PROVIDED/ERROR state)
    
    Examples:
        >>> table = SymbolTable()
        >>> # Register a symbol entry
        >>> entry = SymbolEntry(
        ...     symbol=sp.Symbol('x'),
        ...     name='metric_x',
        ...     dataset='dataset1',
        ...     result_key=ResultKey('key1'),
        ...     metric_spec=None,
        ...     ops=[],
        ...     retrieval_fn=lambda k: Success(1.0)
        ... )
        >>> table.register(entry)
        >>> 
        >>> # Mark dataset as ready
        >>> table.mark_dataset_ready('dataset1')
        >>> 
        >>> # Evaluate ready symbols
        >>> results = table.evaluate_ready_symbols(ResultKey('key1'))
    """
    
    def __init__(self) -> None:
        self._entries: dict[sp.Symbol, SymbolEntry] = {}
        self._by_dataset: dict[str, list[sp.Symbol]] = defaultdict(list)
        self._by_metric: dict[str, list[sp.Symbol]] = defaultdict(list)
        self._evaluation_order: list[sp.Symbol] = []
        self._available_datasets: set[str] = set()
        self._check_symbols: dict[str, set[sp.Symbol]] = defaultdict(set)
    
    # Registration methods
    def register(self, entry: SymbolEntry) -> None:
        """Register a new symbol entry in the symbol table.
        
        Adds a new symbol to the table and updates all relevant indexes. The symbol
        must not already exist in the table.
        
        Args:
            entry: The SymbolEntry to register containing symbol metadata,
                computation requirements, and initial state.
        
        Raises:
            DQXError: If the symbol is already registered in the table.
        
        Examples:
            >>> table = SymbolTable()
            >>> entry = SymbolEntry(
            ...     symbol=sp.Symbol('x'),
            ...     name='metric_x',
            ...     dataset='dataset1',
            ...     result_key=ResultKey('key1'),
            ...     metric_spec=None,
            ...     ops=[],
            ...     retrieval_fn=lambda k: Success(1.0)
            ... )
            >>> table.register(entry)
        """
        if entry.symbol in self._entries:
            raise DQXError(f"Symbol {entry.symbol} already registered")
        
        self._entries[entry.symbol] = entry
        
        # Update indexes
        if entry.dataset:
            self._by_dataset[entry.dataset].append(entry.symbol)
        
        if entry.metric_spec:
            self._by_metric[entry.metric_spec.name].append(entry.symbol)
        
        # Add to evaluation order
        self._evaluation_order.append(entry.symbol)
    
    def register_symbol_for_check(self, symbol: sp.Symbol, check_name: str) -> None:
        """Track which check owns a specific symbol.
        
        Associates a symbol with a named check for ownership tracking. This allows
        retrieval of all symbols belonging to a specific check.
        
        Args:
            symbol: The sympy Symbol to associate with the check.
            check_name: The name of the check that owns this symbol.
        
        Examples:
            >>> table = SymbolTable()
            >>> symbol = sp.Symbol('x')
            >>> table.register_symbol_for_check(symbol, 'quality_check_1')
            >>> symbols = table.get_symbols_for_check('quality_check_1')
            >>> assert symbol in symbols
        """
        self._check_symbols[check_name].add(symbol)
    
    def register_from_provider(
        self, 
        symbolic_metric: SymbolicMetric, 
        key: ResultKey
    ) -> sp.Symbol:
        """Create and register an entry from an existing SymbolicMetric.
        
        Converts a SymbolicMetric from the provider layer into a SymbolEntry
        and registers it in the table. Automatically determines if the metric
        is externally provided based on key comparison.
        
        Args:
            symbolic_metric: The SymbolicMetric containing symbol definition,
                dependencies, and computation function.
            key: The ResultKey to use for storing/retrieving computed values.
        
        Returns:
            The registered symbol for reference.
        
        Examples:
            >>> from dqx.provider import SymbolicMetric
            >>> table = SymbolTable()
            >>> metric = SymbolicMetric(
            ...     symbol=sp.Symbol('x'),
            ...     name='metric_x',
            ...     datasets=['dataset1'],
            ...     fn=lambda k: Success(1.0),
            ...     key_provider=SomeKeyProvider(),
            ...     dependencies=[]
            ... )
            >>> symbol = table.register_from_provider(metric, ResultKey('key1'))
            >>> entry = table.get(symbol)
            >>> assert entry is not None
        """
        # Extract ops from metric spec
        ops = list(symbolic_metric.dependencies[0][0].analyzers) if symbolic_metric.dependencies else []
        
        # Extract single dataset from provider's datasets list
        dataset = symbolic_metric.datasets[0] if symbolic_metric.datasets else None
        
        entry = SymbolEntry(
            symbol=symbolic_metric.symbol,
            name=symbolic_metric.name,
            dataset=dataset,
            result_key=key,
            metric_spec=symbolic_metric.dependencies[0][0] if symbolic_metric.dependencies else None,  # type: ignore
            ops=ops,
            retrieval_fn=symbolic_metric.fn,
            dependencies=symbolic_metric.dependencies,
        )
        
        # Check if this is a provided metric (different key)
        if symbolic_metric.dependencies and symbolic_metric.key_provider.create(key) != key:
            entry.mark_provided()
        
        self.register(entry)
        return entry.symbol
    
    # Lookup methods
    def get(self, symbol: sp.Symbol) -> SymbolEntry | None:
        """Get entry for a specific symbol.
        
        Args:
            symbol: The sympy Symbol to look up.
        
        Returns:
            The SymbolEntry if found, None otherwise.
        
        Examples:
            >>> table = SymbolTable()
            >>> symbol = sp.Symbol('x')
            >>> entry = table.get(symbol)  # Returns None
            >>> # After registering...
            >>> table.register(SymbolEntry(...))
            >>> entry = table.get(symbol)  # Returns the entry
        """
        return self._entries.get(symbol)
    
    def get_all(self) -> dict[sp.Symbol, SymbolEntry]:
        """Get all symbol entries in the table.
        
        Returns:
            A copy of the internal dictionary mapping symbols to their entries.
            Modifications to the returned dict do not affect the symbol table.
        
        Examples:
            >>> table = SymbolTable()
            >>> # Register some symbols...
            >>> all_entries = table.get_all()
            >>> for symbol, entry in all_entries.items():
            ...     print(f"{symbol}: {entry.state}")
        """
        return self._entries.copy()
    
    def get_symbols_for_check(self, check_name: str) -> list[sp.Symbol]:
        """Get all symbols associated with a specific check.
        
        Args:
            check_name: The name of the check to query.
        
        Returns:
            List of symbols registered for the specified check.
            Returns empty list if check has no symbols.
        
        Examples:
            >>> table = SymbolTable()
            >>> symbol = sp.Symbol('x')
            >>> table.register_symbol_for_check(symbol, 'quality_check')
            >>> symbols = table.get_symbols_for_check('quality_check')
            >>> assert symbol in symbols
        """
        return list(self._check_symbols.get(check_name, set()))
    
    def get_by_dataset(self, dataset: str) -> list[SymbolEntry]:
        """Get all symbols for a specific dataset.
        
        Returns entries for symbols that explicitly require the specified dataset.
        
        Args:
            dataset: The dataset name to filter by.
        
        Returns:
            List of SymbolEntry objects that require the specified dataset.
        
        Examples:
            >>> table = SymbolTable()
            >>> # Register symbols for 'dataset1'
            >>> entries = table.get_by_dataset('dataset1')
            >>> for entry in entries:
            ...     assert entry.dataset == 'dataset1'
        """
        symbols = self._by_dataset.get(dataset, [])
        return [self._entries[sym] for sym in symbols if sym in self._entries]
    
    def get_pending(self, dataset: str | None = None) -> list[SymbolEntry]:
        """Get all pending symbols, optionally filtered by dataset.
        
        When a dataset is specified, returns pending symbols that either:
        1. Explicitly require that dataset
        2. Have no dataset binding yet (dataset is None)
        
        Args:
            dataset: Optional dataset name to filter by. If None, returns
                all pending symbols regardless of dataset requirements.
        
        Returns:
            List of SymbolEntry objects in PENDING state.
        
        Examples:
            >>> table = SymbolTable()
            >>> # Get all pending symbols
            >>> all_pending = table.get_pending()
            >>> 
            >>> # Get pending symbols for 'dataset1'
            >>> dataset_pending = table.get_pending('dataset1')
        """
        if dataset:
            # Include symbols specifically for this dataset or unbound symbols
            return [
                entry for entry in self._entries.values()
                if entry.is_pending() and (entry.dataset == dataset or entry.dataset is None)
            ]
        return [
            entry for entry in self._entries.values()
            if entry.is_pending()
        ]
    
    def get_ready(self) -> list[SymbolEntry]:
        """Get all symbols ready for evaluation.
        
        Returns symbols in READY state, meaning their datasets are available
        but they haven't been evaluated yet.
        
        Returns:
            List of SymbolEntry objects in READY state.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After marking dataset success...
            >>> table.mark_dataset_ready('dataset1')
            >>> ready_symbols = table.get_ready()
            >>> for entry in ready_symbols:
            ...     assert entry.state == "READY"
        """
        return [
            entry for entry in self._entries.values()
            if entry.state == "READY"
        ]
    
    def get_successful(self) -> list[SymbolEntry]:
        """Get all successfully computed symbols.
        
        Returns symbols that have been evaluated successfully, with their
        computed values available.
        
        Returns:
            List of SymbolEntry objects in PROVIDED state with Success values.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After successful evaluation...
            >>> successful = table.get_successful()
            >>> for entry in successful:
            ...     value = entry.get_value()
            ...     assert isinstance(value, Success)
        """
        return [
            entry for entry in self._entries.values()
            if entry.state == "PROVIDED" and isinstance(entry.value, Some)
        ]
    
    # State management
    def update_state(self, symbol: sp.Symbol, state: SymbolState) -> None:
        """Update the state of a symbol.
        
        Directly sets the state of a symbol without any validation or side effects.
        For most use cases, prefer using the specific state transition methods
        on SymbolEntry (mark_ready, mark_error, etc.).
        
        Args:
            symbol: The sympy Symbol to update.
            state: The new state to set (PENDING, READY, PROVIDED, or ERROR).
        
        Raises:
            DQXError: If the symbol is not found in the table.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After registering a symbol...
            >>> table.update_state(symbol, "READY")
        """
        entry = self.get(symbol)
        if not entry:
            raise DQXError(f"Symbol {symbol} not found")
        entry.state = state
    
    def mark_dataset_failed(self, dataset: str, error: str) -> None:
        """Mark all pending symbols for a dataset as failed.
        
        When a dataset fails to load or process, this method transitions all
        pending symbols that depend on it to ERROR state with an appropriate
        error message.
        
        Args:
            dataset: The name of the dataset that failed.
            error: The error message describing the failure.
        
        Examples:
            >>> table = SymbolTable()
            >>> # When dataset loading fails...
            >>> table.mark_dataset_failed('dataset1', 'File not found')
            >>> # All pending symbols for dataset1 are now in ERROR state
        """
        for entry in self.get_pending(dataset):
            entry.mark_error(f"Dataset {dataset} failed: {error}")
    
    def mark_dataset_ready(self, dataset: str) -> None:
        """Mark all pending symbols for a dataset as ready for evaluation.
        
        When a dataset is successfully loaded, this method transitions all
        pending symbols that can use it from PENDING to READY state. This
        includes both symbols that explicitly require the dataset and
        universal symbols with no dataset restrictions.
        
        Args:
            dataset: The name of the dataset that is now available.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After successful dataset loading...
            >>> table.mark_dataset_ready('dataset1')
            >>> # Symbols can now be evaluated
            >>> ready = table.get_ready()
            >>> results = table.evaluate_ready_symbols(key)
        """
        # Mark symbols for this specific dataset as ready
        for entry in self.get_pending(dataset):
            entry.mark_ready()
    
    # Evaluation methods
    def evaluate_symbol(self, symbol: sp.Symbol, key: ResultKey) -> Result[float, str]:
        """Evaluate a single symbol using its retrieval function.
        
        Executes the symbol's retrieval function to compute its value, updating
        the symbol's state based on the result. If successful, the symbol transitions
        to PROVIDED state with the computed value. If failed, it transitions to ERROR
        state with the error message.
        
        Args:
            symbol: The sympy Symbol to evaluate.
            key: The ResultKey to pass to the retrieval function for accessing
                stored computation results.
        
        Returns:
            Result containing either:
            - Success with the computed float value
            - Failure with an error message
        
        Examples:
            >>> table = SymbolTable()
            >>> # After symbol is ready...
            >>> result = table.evaluate_symbol(symbol, ResultKey('key1'))
            >>> if isinstance(result, Success):
            ...     print(f"Value: {result.unwrap()}")
            ... else:
            ...     print(f"Error: {result.failure()}")
        """
        entry = self.get(symbol)
        if not entry:
            return Failure(f"Symbol {symbol} not found")
        
        if entry.state == "ERROR":
            return entry.get_value() or Failure("Unknown error")  # type: ignore
        
        if not entry.is_ready():
            return Failure(f"Symbol {symbol} not ready for evaluation")
        
        try:
            result = entry.retrieval_fn(key)
            if isinstance(result, Success):
                entry.mark_success(result.unwrap())
            else:
                entry.mark_error(result.failure())
            return result
        except Exception as e:
            error_msg = f"Failed to evaluate symbol {symbol}: {str(e)}"
            entry.mark_error(error_msg)
            return Failure(error_msg)
    
    def evaluate_ready_symbols(self, key: ResultKey) -> dict[sp.Symbol, Result[float, str]]:
        """Evaluate all symbols in READY state.
        
        Batch evaluates all symbols that are ready for computation. Each symbol
        is evaluated independently, and failures in one symbol do not affect others.
        
        Args:
            key: The ResultKey to pass to all retrieval functions.
        
        Returns:
            Dictionary mapping each evaluated symbol to its Result:
            - Success with computed value if evaluation succeeded
            - Failure with error message if evaluation failed
        
        Examples:
            >>> table = SymbolTable()
            >>> # After marking datasets as ready...
            >>> results = table.evaluate_ready_symbols(ResultKey('key1'))
            >>> for symbol, result in results.items():
            ...     if isinstance(result, Success):
            ...         print(f"{symbol} = {result.unwrap()}")
            ...     else:
            ...         print(f"{symbol} failed: {result.failure()}")
        """
        results = {}
        for entry in self.get_ready():
            results[entry.symbol] = self.evaluate_symbol(entry.symbol, key)
        return results
    
    # Analysis methods
    def get_required_metrics(self, dataset: str | None = None) -> set[MetricSpec]:
        """Get unique set of metrics required for computation.
        
        Analyzes pending symbols to identify all unique MetricSpec objects
        that need to be computed. This helps determine which analyzers
        need to run.
        
        Args:
            dataset: Optional dataset name to filter analysis. If provided,
                only considers pending symbols for that dataset.
        
        Returns:
            Set of unique MetricSpec objects required by pending symbols.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After registering symbols with metrics...
            >>> metrics = table.get_required_metrics()
            >>> for metric in metrics:
            ...     print(f"Need to compute: {metric.name}")
            >>> 
            >>> # For specific dataset
            >>> dataset_metrics = table.get_required_metrics('dataset1')
        """
        entries = self.get_pending(dataset) if dataset else self._entries.values()
        metrics = set()
        for entry in entries:
            if entry.metric_spec and entry.is_pending():
                metrics.add(entry.metric_spec)
        return metrics
    
    def get_required_analyzers(self, dataset: str | None = None) -> list[Op]:
        """Get all analyzers needed for pending metrics.
        
        Collects all analyzer operations (Op objects) required by pending
        symbols. Note that this may include duplicate analyzers if multiple
        symbols require the same operation.
        
        Args:
            dataset: Optional dataset name to filter analysis. If provided,
                only considers pending symbols for that dataset.
        
        Returns:
            List of Op objects representing required analyzers. May contain
            duplicates if multiple symbols need the same analyzer.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After registering symbols...
            >>> analyzers = table.get_required_analyzers()
            >>> unique_analyzers = list(set(analyzers))
            >>> for analyzer in unique_analyzers:
            ...     print(f"Need to run: {analyzer}")
        """
        analyzers = []
        for entry in self.get_pending(dataset):
            analyzers.extend(entry.ops)
        return analyzers
    
    # Utility methods
    def build_dependency_graph(self) -> dict[sp.Symbol, set[sp.Symbol]]:
        """Build a graph of symbol dependencies.
        
        Creates a dependency graph where each symbol maps to the set of symbols
        it depends on. Currently returns an empty dependency set for each symbol
        as dependencies are implicit through expressions.
        
        Note:
            This method is a placeholder for future dependency tracking.
            Full implementation would parse symbolic expressions to identify
            which symbols depend on others.
        
        Returns:
            Dictionary mapping each symbol to its set of dependencies.
            Currently all sets are empty.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After registering symbols...
            >>> deps = table.build_dependency_graph()
            >>> for symbol, dependencies in deps.items():
            ...     if dependencies:
            ...         print(f"{symbol} depends on: {dependencies}")
        """
        graph: dict[sp.Symbol, set[sp.Symbol]] = defaultdict(set)
        
        for symbol, entry in self._entries.items():
            # For now, dependencies are implicit through expressions
            # This would need to be extended to parse expressions for symbol dependencies
            graph[symbol] = set()
        
        return dict(graph)
    
    def validate_datasets(self, available_datasets: list[str]) -> list[str]:
        """Validate that required datasets are available for all symbols.
        
        Checks each symbol's dataset requirement against the provided list of
        available datasets. For symbols without a dataset binding (dataset is None),
        assigns them to the first available dataset.
        Updates internal indexes when dataset assignments change.
        
        Args:
            available_datasets: List of dataset names that are available
                for computation.
        
        Returns:
            List of error messages for symbols whose dataset requirements
            cannot be satisfied. Empty list if all requirements are met.
        
        Examples:
            >>> table = SymbolTable()
            >>> # Register symbols with dataset requirements...
            >>> errors = table.validate_datasets(['dataset1', 'dataset2'])
            >>> if errors:
            ...     for error in errors:
            ...         print(f"Validation error: {error}")
            ... else:
            ...     print("All dataset requirements satisfied")
        """
        errors = []
        for entry in self._entries.values():
            # Store original dataset to check if it changed
            original_dataset = entry.dataset
            
            result = entry.validate_dataset(available_datasets)
            if isinstance(result, Failure):
                errors.append(result.failure())
            else:
                # If dataset was updated (None -> actual dataset), update indexes
                if original_dataset is None and entry.dataset is not None:
                    if entry.symbol not in self._by_dataset[entry.dataset]:
                        self._by_dataset[entry.dataset].append(entry.symbol)
        return errors
    
    def clear(self) -> None:
        """Clear all entries from the symbol table.
        
        Removes all registered symbols and resets all internal data structures
        to their initial empty state. This includes clearing all indexes and
        ownership tracking.
        
        Examples:
            >>> table = SymbolTable()
            >>> # After using the table...
            >>> table.clear()
            >>> assert len(table.get_all()) == 0
            >>> assert repr(table) == "SymbolTable(total=0, pending=0, ready=0, successful=0, error=0)"
        """
        self._entries.clear()
        self._by_dataset.clear()
        self._by_metric.clear()
        self._evaluation_order.clear()
        self._available_datasets.clear()
        self._check_symbols.clear()
    
    def __repr__(self) -> str:
        """String representation of the symbol table.
        
        Provides a summary of the symbol table's current state, including
        counts of symbols in each state.
        
        Returns:
            String showing total symbols and counts by state (pending, ready,
            successful, error).
        
        Examples:
            >>> table = SymbolTable()
            >>> print(repr(table))
            SymbolTable(total=0, pending=0, ready=0, successful=0, error=0)
            >>> # After registering and processing symbols...
            >>> print(repr(table))
            SymbolTable(total=10, pending=3, ready=2, successful=4, error=1)
        """
        total = len(self._entries)
        pending = len(self.get_pending())
        ready = len(self.get_ready())
        successful = len(self.get_successful())
        error = sum(1 for e in self._entries.values() if e.is_error())
        
        return (
            f"SymbolTable("
            f"total={total}, "
            f"pending={pending}, "
            f"ready={ready}, "
            f"successful={successful}, "
            f"error={error})"
        )
