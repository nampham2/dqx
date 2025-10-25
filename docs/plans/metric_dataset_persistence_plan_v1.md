# Metric Dataset Persistence Implementation Plan v1

## Overview

This plan addresses the issue where dataset information is lost during metric persistence. Currently, `SymbolicMetric` objects have dataset information after imputation, but this information is not persisted to the database, making it impossible to uniquely identify metrics across different datasets.

## Problem Statement

1. The database schema has a commented-out `dataset` field in the `Metric` table
2. The `models.Metric` dataclass lacks a dataset field
3. Dataset information from `SymbolicMetric` is lost when creating `models.Metric` objects
4. This prevents proper metric identification when multiple datasets are used

## Solution Approach

Use a minimal-change strategy that:
- Does NOT modify `MetricSpec` (keeps changes isolated)
- Passes dataset information via a separate mapping dictionary
- Maintains backward compatibility with optional dataset fields
- Ensures E2E tests remain unchanged

## Implementation Phases

### Phase 1: Add Dataset Field to Data Model (TDD)

#### Step 1.1: Write failing tests for Metric.dataset field
```python
# tests/test_models.py
def test_metric_has_dataset_field():
    """Test that Metric dataclass includes dataset field."""
    metric = specs.Average("price")
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
    state = states.Average(10.0, 5.0)

    # Should be able to create Metric with dataset
    m = models.Metric(spec=metric, state=state, key=key, dataset="test_ds")
    assert m.dataset == "test_ds"

def test_metric_build_accepts_dataset():
    """Test that Metric.build() accepts optional dataset parameter."""
    metric = specs.Average("price")
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
    state = states.Average(10.0, 5.0)

    # Without dataset (backward compatibility)
    m1 = models.Metric.build(metric, key, state)
    assert m1.dataset is None

    # With dataset
    m2 = models.Metric.build(metric, key, state, dataset="test_ds")
    assert m2.dataset == "test_ds"
```

#### Step 1.2: Implement dataset field in Metric
```python
# src/dqx/models.py
@dataclass
class Metric:
    """Domain model for a metric computation result."""
    spec: specs.MetricSpec
    state: State
    key: ResultKey
    dataset: str | None = None  # ADD THIS LINE
    metric_id: uuid.UUID | None = None

    @staticmethod
    def build(
        spec: MetricSpec,
        key: ResultKey,
        state: State,
        dataset: str | None = None  # ADD THIS PARAMETER
    ) -> "Metric":
        """Build a Metric instance."""
        return Metric(spec=spec, state=state, key=key, dataset=dataset)
```

#### Step 1.3: Update existing Metric.build() calls in tests
Update ~30 test instances to include dataset parameter where appropriate:
```python
# Before
metric = models.Metric.build(spec, key, state)

# After (for tests that need dataset)
metric = models.Metric.build(spec, key, state, dataset="test_dataset")
```

### Phase 2: Database Schema Updates (TDD)

#### Step 2.1: Write failing tests for database persistence
```python
# tests/orm/test_repositories.py
def test_metric_persists_with_dataset():
    """Test that dataset field is persisted and retrieved."""
    db = InMemoryMetricDB()
    spec = specs.Average("price")
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
    state = states.Average(10.0, 5.0)

    # Create metric with dataset
    metric = models.Metric.build(spec, key, state, dataset="prod_dataset")

    # Persist
    persisted = list(db.persist([metric]))[0]

    # Retrieve
    result = db.get(persisted.metric_id)
    assert result.is_some()
    assert result.unwrap().dataset == "prod_dataset"

def test_metric_search_by_dataset():
    """Test searching metrics by dataset."""
    db = InMemoryMetricDB()
    spec = specs.Average("price")
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})

    # Create metrics for different datasets
    m1 = models.Metric.build(spec, key, states.Average(10.0, 5.0), dataset="ds1")
    m2 = models.Metric.build(spec, key, states.Average(20.0, 5.0), dataset="ds2")

    db.persist([m1, m2])

    # Search by dataset
    results_ds1 = db.search(
        repositories.Metric.spec == spec.to_spec_tuple(),
        repositories.Metric.key == key,
        repositories.Metric.dataset == "ds1"
    )

    assert len(results_ds1) == 1
    assert results_ds1[0].dataset == "ds1"
```

#### Step 2.2: Uncomment and update database schema
```python
# src/dqx/orm/repositories.py
class Metric(DeclarativeBase):
    """SQLAlchemy model for persisting metrics."""

    __tablename__ = "metric"

    metric_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    spec: Mapped[tuple[str, ...]] = mapped_column(nullable=False)
    state: Mapped[str] = mapped_column(nullable=False)
    parameters: Mapped[dict[str, t.Any]] = mapped_column(nullable=False)
    key: Mapped[ResultKey] = mapped_column(nullable=False)
    dataset: Mapped[str | None] = mapped_column(nullable=True)  # UNCOMMENT & make nullable

    # Update from_domain and to_domain methods
    @classmethod
    def from_domain(cls, metric: models.Metric) -> "Metric":
        """Convert domain model to ORM model."""
        # ... existing code ...
        dataset=metric.dataset,  # ADD THIS LINE

    def to_domain(self) -> models.Metric:
        """Convert ORM model to domain model."""
        # ... existing code ...
        dataset=self.dataset,  # ADD THIS LINE
```

### Phase 3: Analyzer Integration (TDD)

#### Step 3.1: Write tests for analyzer dataset mapping
```python
# tests/test_analyzer.py
def test_analyzer_accepts_dataset_mapping():
    """Test that analyzer can accept and use dataset mapping."""
    analyzer = Analyzer()
    spec = specs.Average("price")
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})

    # Create dataset mapping
    dataset_mapping = {spec: "test_dataset"}

    # Mock datasource
    ds = Mock(spec=DataSource)
    ds.name = "test_dataset"
    ds.execute.return_value = Mock(
        analyze=[Mock(spec=spec, state=states.Average(10.0, 5.0))]
    )

    # Analyze with dataset mapping
    report = analyzer.analyze([ds], [spec], key, dataset_mapping=dataset_mapping)

    # Verify dataset was set
    metric = report._report[(spec, key)]
    assert metric.dataset == "test_dataset"
```

#### Step 3.2: Update Analyzer to accept dataset mapping
```python
# src/dqx/analyzer.py
def analyze(
    self,
    dataset: DataSource | list[DataSource],
    metric_specs: Iterable[MetricSpec],
    key: ResultKey,
    dataset_mapping: dict[MetricSpec, str] | None = None,  # ADD THIS PARAMETER
) -> AnalysisReport:
    """Analyze metrics with optional dataset mapping."""
    # ... existing code ...

def _analyze_internal(
    self,
    dataset: DataSource,
    metric_specs: Iterable[MetricSpec],
    key: ResultKey,
    dataset_mapping: dict[MetricSpec, str] | None = None,  # ADD THIS PARAMETER
) -> None:
    """Internal analysis implementation."""
    # ... existing code ...

    # When building metrics:
    for metric_spec, state in zip(metric_specs, states):
        dataset_name = None
        if dataset_mapping and metric_spec in dataset_mapping:
            dataset_name = dataset_mapping[metric_spec]

        metric = models.Metric.build(
            spec=metric_spec,
            key=key,
            state=state,
            dataset=dataset_name  # ADD THIS
        )
        self._report[(metric_spec, key)] = metric
```

### Phase 4: API Integration (TDD)

#### Step 4.1: Write integration test
```python
# tests/test_api_dataset_tracking.py
def test_verification_suite_tracks_dataset_in_metrics():
    """Test that VerificationSuite properly tracks dataset in persisted metrics."""
    @check(name="Test Check", datasets=["ds1"])
    def test_check(mp: MetricProvider, ctx: Context) -> None:
        ctx.assert_that(mp.average("price", dataset="ds1")).is_geq(10.0)

    # Create mock database that captures persisted metrics
    persisted_metrics = []

    class MockDB(MetricDB):
        def persist(self, metrics, overwrite=True):
            persisted_metrics.extend(metrics)
            return metrics
        # ... other required methods ...

    db = MockDB()
    suite = VerificationSuite([test_check], db, "Test Suite")

    # Run suite
    ds = Mock(spec=DataSource, name="ds1")
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 1), tags={})
    suite.run([ds], key)

    # Verify dataset was persisted
    assert len(persisted_metrics) > 0
    assert all(m.dataset == "ds1" for m in persisted_metrics if m.spec.column == "price")
```

#### Step 4.2: Update VerificationSuite to create dataset mapping
```python
# src/dqx/api.py
def run(self, datasources: list[DataSource], key: ResultKey) -> None:
    """Run verification suite with dataset tracking."""
    # ... existing validation code ...

    # Create dataset mapping from pending metrics
    dataset_mapping: dict[MetricSpec, str] = {}
    for ds in self._datasource_map.values():
        for symbol_info in self.provider.pending_metrics(ds.name):
            if symbol_info.dataset:
                dataset_mapping[symbol_info.spec] = symbol_info.dataset

    # Pass dataset mapping to analyzer
    for dataset in self._datasource_map.values():
        metrics_for_dataset = [
            m.spec for m in self.provider.pending_metrics(dataset.name)
        ]
        report = self._analyzer.analyze(
            dataset,
            metrics_for_dataset,
            key,
            dataset_mapping=dataset_mapping  # ADD THIS
        )
        report.persist(self._db)
```

### Phase 5: Update Remaining Tests

Update the ~30 test instances that directly create Metric objects:

1. **tests/test_models.py** - Add dataset parameter to Metric.build calls
2. **tests/test_analyzer.py** - Add dataset to test metrics
3. **tests/orm/test_repositories.py** - Include dataset in persistence tests

Example updates:
```python
# Before
metric = models.Metric.build(spec, key, state)

# After (where dataset context exists)
metric = models.Metric.build(spec, key, state, dataset="test_dataset")

# Or keep as None for backward compatibility tests
metric = models.Metric.build(spec, key, state, dataset=None)
```

### Phase 6: Final Verification

1. Run all tests with coverage:
   ```bash
   uv run pytest tests/ -v --cov=dqx
   ```

2. Run pre-commit checks:
   ```bash
   bin/run-hooks.sh --all
   ```

3. Verify E2E tests pass unchanged:
   ```bash
   uv run pytest tests/e2e/ -v
   ```

4. Run mypy type checking:
   ```bash
   uv run mypy src tests
   ```

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test the flow from API to database
3. **E2E Tests**: Ensure existing E2E tests pass without modification
4. **Database Tests**: Verify dataset field persistence and queries

## Success Criteria

1. Dataset information flows from API → Analyzer → Database
2. Metrics can be uniquely identified by (spec, key, dataset) tuple
3. All existing E2E tests pass without modification
4. 100% test coverage maintained
5. No mypy or ruff errors
6. Backward compatibility maintained (dataset field is optional)

## Rollback Plan

If issues arise, revert the branch:
```bash
git checkout main
git branch -D feat/metric-dataset-persistence
```

## Notes

- This approach minimizes changes by not modifying MetricSpec
- Dataset mapping is handled separately from the spec hierarchy
- Optional fields ensure backward compatibility
- The implementation follows TDD principles throughout
