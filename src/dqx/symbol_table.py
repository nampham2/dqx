"""Symbol table implementation for managing computation symbols and their metadata."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Iterable, Literal, Sequence

import sympy as sp
from returns.result import Failure, Result, Success

from dqx.common import DQXError, ResultKey, ResultKeyProvider, RetrievalFn
from dqx.ops import Op
from dqx.provider import SymbolicMetric
from dqx.specs import MetricSpec

# Type definitions
SymbolState = Literal["PENDING", "READY", "ERROR"]


@dataclass
class SymbolEntry:
    """Complete metadata for a symbol in the computation graph.

    Represents a single symbol in the computation system, tracking its state,
    dependencies, and computation requirements throughout its lifecycle.

    Attributes:
        symbolic_metric: Optional reference to the SymbolicMetric containing core metadata.
        dataset: The dataset name this symbol is associated with. None means
            the symbol will be bound to a dataset when one becomes available.
        result_key: The ResultKey used to store/retrieve computed values.
        value: Optional Result containing either Success(float) or Failure(str).
        state: Current state in the lifecycle: PENDING → READY → ERROR.

    State Transitions:
        PENDING: Initial state, waiting for dataset availability.
        READY: Dataset available, ready for evaluation.
        PROVIDED: Successfully computed or externally provided.
        ERROR: Computation failed.

    Note:
        metric_spec and ops are now computed properties that derive their values
        from symbolic_metric to avoid data redundancy.
    """

    # Core reference to SymbolicMetric
    symbolic_metric: SymbolicMetric

    # Data source information
    dataset: str | None = None
    result_key: ResultKey | None = None

    # State tracking
    value: Result[float, str] | None = None
    state: SymbolState = "PENDING"

    @property
    def symbol(self) -> sp.Symbol:
        """Get the symbol from SymbolicMetric."""
        return self.symbolic_metric.symbol

    @property
    def name(self) -> str:
        """Get the name from SymbolicMetric."""
        return self.symbolic_metric.name

    @property
    def retrieval_fn(self) -> RetrievalFn:
        """Get the retrieval function from SymbolicMetric."""
        return self.symbolic_metric.fn

    @property
    def dependencies(self) -> list[tuple[MetricSpec, ResultKeyProvider]]:
        """Get the dependencies from SymbolicMetric."""
        return self.symbolic_metric.dependencies

    @property
    def metric_specs(self) -> Sequence[MetricSpec]:
        """Get all metric specs from dependencies.

        Returns all MetricSpec objects from the symbol's dependencies.
        This supports metrics that depend on multiple different MetricSpecs.
        """
        return [dep[0] for dep in self.symbolic_metric.dependencies]

    @property
    def ops(self) -> Iterable[Op]:
        """Get aggregated analyzer operations from all metric specs.

        Returns the list of analyzer operations (ops) required for this symbol,
        aggregated from all metric specs' analyzers. Duplicates are preserved
        to maintain the original order and allow the execution layer to handle
        deduplication if needed.
        """
        return chain.from_iterable(spec.analyzers for spec in self.metric_specs)

    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied."""
        return self.state == "READY"

    def is_pending(self) -> bool:
        """Check if this symbol is pending computation."""
        return self.state == "PENDING"

    def is_error(self) -> bool:
        """Check if this symbol has an error."""
        return self.state == "ERROR"

    def mark_ready(self) -> None:
        """Mark this symbol as ready for evaluation and notify observers."""
        self.state = "READY"

    def mark_error(self, error: str) -> None:
        """Mark this symbol as failed and notify observers."""
        self.state = "ERROR"
        self.value = Failure(error)

    def mark_success(self, value: float) -> None:
        """Mark this symbol as successfully computed and notify observers."""
        # Transition to READY state and set success value
        self.state = "READY"
        self.value = Success(value)

    def get_value(self) -> Result[float, str] | None:
        """Get the computed value if available."""
        return self.value

    def validate_dataset(self, available_datasets: list[str]) -> Result[str, str]:
        """Validate that required dataset is available.

        Returns:
            Success with the bound dataset name if a binding occurred (str),
            Success with None if no binding was needed,
            or Failure with error message if validation failed.
        """
        if self.dataset is None:
            # Symbol requires a dataset but none specified - will bind to first available
            if not available_datasets:
                return Failure(f"Symbol {self.symbol} requires a dataset but none available")
            if len(available_datasets) > 1:
                return Failure(f"Symbol {self.symbol} requires a SINGLE dataset but multiple available: {available_datasets}")
            # Bind to the provided dataset
            self.dataset = available_datasets[0]
            # Return the dataset that was bound
        elif available_datasets and self.dataset not in available_datasets:
            # Symbol has specific dataset requirement
            return Failure(
                f"Symbol {self.symbol} requires dataset '{self.dataset}' but only {available_datasets} available"
            )
        # No binding needed - dataset already set and is available
        return Success(self.dataset)


class SymbolTable:
    """Central registry for all symbols and their computation metadata.

    The SymbolTable manages the lifecycle of all symbols in the computation system,
    tracking their states, dependencies, and relationships. It provides efficient
    lookup by symbol, dataset, metric, and state, enabling coordinated evaluation
    of complex symbolic expressions.

    The table maintains several indexes for efficient queries:
    - Direct symbol lookup via _entries
    - Dataset-based lookup via _by_dataset

    Symbol Lifecycle:
        1. Symbol registered with metadata (PENDING state)
        2. Dataset becomes available → mark_dataset_success() (READY state)
        3. Symbol evaluated → evaluate_symbol() (PROVIDED/ERROR state)

    Examples:
        >>> table = SymbolTable()
        >>> # Register a symbol entry using register_from_provider
        >>> from dqx.provider import SymbolicMetric
        >>> symbolic_metric = SymbolicMetric(
        ...     symbol=sp.Symbol('x'),
        ...     name='metric_x',
        ...     datasets=['dataset1'],
        ...     fn=lambda k: Success(1.0),
        ...     key_provider=ResultKeyProvider(),
        ...     dependencies=[]
        ... )
        >>> table.register_from_provider(symbolic_metric, ResultKey('key1'))
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
            >>> # Create entry using SymbolicMetric
            >>> from dqx.provider import SymbolicMetric
            >>> symbolic_metric = SymbolicMetric(
            ...     symbol=sp.Symbol('x'),
            ...     name='metric_x',
            ...     datasets=['dataset1'],
            ...     fn=lambda k: Success(1.0),
            ...     key_provider=ResultKeyProvider(),
            ...     dependencies=[]
            ... )
            >>> entry = SymbolEntry(
            ...     symbolic_metric=symbolic_metric,
            ...     dataset='dataset1',
            ...     result_key=ResultKey('key1'),
            ...     metric_spec=None,
            ...     ops=[]
            ... )
            >>> table.register(entry)
        """
        if entry.symbol in self._entries:
            raise DQXError(f"Symbol {entry.symbol} already registered")

        self._entries[entry.symbol] = entry

        # Update indexes
        if entry.dataset:
            self._by_dataset[entry.dataset].append(entry.symbol)


    def register_from_provider(self, symbolic_metric: SymbolicMetric, key: ResultKey) -> sp.Symbol:
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
        # Extract single dataset from provider's datasets list
        dataset = symbolic_metric.datasets[0] if symbolic_metric.datasets else None

        entry = SymbolEntry(
            symbolic_metric=symbolic_metric,
            dataset=dataset,
            result_key=key,
        )

        # Check if this is a provided metric (different key)
        # In the new design, we no longer need special handling for provided metrics
        # The value field will indicate if a value exists

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
                entry
                for entry in self._entries.values()
                if entry.is_pending() and (entry.dataset == dataset or entry.dataset is None)
            ]
        return [entry for entry in self._entries.values() if entry.is_pending()]

    def get_ready(self) -> list[SymbolEntry]:
        """Get all symbols ready for evaluation.

        Returns symbols in READY state, meaning their datasets are available
        but they haven't been evaluated yet. This excludes symbols that have
        already been successfully evaluated (have a Success value).

        Returns:
            List of SymbolEntry objects in READY state without Success values.

        Examples:
            >>> table = SymbolTable()
            >>> # After marking dataset success...
            >>> table.mark_dataset_ready('dataset1')
            >>> ready_symbols = table.get_ready()
            >>> for entry in ready_symbols:
            ...     assert entry.state == "READY"
        """
        return [
            entry
            for entry in self._entries.values()
            if entry.state == "READY" and not (entry.value and isinstance(entry.value, Success))
        ]

    def get_successful(self) -> list[SymbolEntry]:
        """Get all successfully computed symbols.

        Returns symbols that have been evaluated successfully, with their
        computed values available.

        Returns:
            List of SymbolEntry objects with Success values.

        Examples:
            >>> table = SymbolTable()
            >>> # After successful evaluation...
            >>> successful = table.get_successful()
            >>> for entry in successful:
            ...     value = entry.get_value()
            ...     assert isinstance(value, Success)
        """
        return [
            entry for entry in self._entries.values() if entry.value is not None and isinstance(entry.value, Success)
        ]

    def bind_symbol_to_dataset(self, symbol: sp.Symbol, dataset: str) -> None:
        """Update the dataset binding for a symbol and maintain indexes.

        This method should be called when a symbol that had no dataset (None)
        gets bound to a specific dataset during validation.

        Args:
            symbol: The symbol to bind to a dataset.
            dataset: The dataset name to bind to.

        Examples:
            >>> table = SymbolTable()
            >>> # After a symbol gets bound during validation
            >>> table.bind_symbol_to_dataset(symbol, 'dataset1')
        """
        entry = self.get(symbol)
        if entry and dataset not in self._by_dataset:
            self._by_dataset[dataset] = []

        if entry and symbol not in self._by_dataset[dataset]:
            self._by_dataset[dataset].append(symbol)

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
            return entry.get_value() or Failure("Unknown error")

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
            if entry.is_pending():
                for spec in entry.metric_specs:
                    metrics.add(spec)
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
        analyzers: list[Op] = []
        for entry in self.get_pending(dataset):
            analyzers.extend(entry.ops)
        return analyzers

    def clear(self) -> None:
        """Clear all entries from the symbol table.

        Removes all registered symbols and resets all internal data structures
        to their initial empty state. This includes clearing all indexes.

        Examples:
            >>> table = SymbolTable()
            >>> # After using the table...
            >>> table.clear()
            >>> assert len(table.get_all()) == 0
            >>> assert repr(table) == "SymbolTable(total=0, pending=0, ready=0, successful=0, error=0)"
        """
        self._entries.clear()
        self._by_dataset.clear()

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

        return f"SymbolTable(total={total}, pending={pending}, ready={ready}, successful={successful}, error={error})"
