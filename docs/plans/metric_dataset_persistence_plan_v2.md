# Metric Dataset Persistence Plan v2

## Overview

This plan addresses the issue where dataset information is lost when metrics are persisted to the database. Currently, the `dataset` field in the `metric` table is commented out and not populated, making it impossible to determine which dataset a metric belongs to after persistence.

**Key Changes in v2:**
- Dataset field is **mandatory** everywhere (no Optional types)
- **No backward compatibility** maintained
- Simplified API using list instead of dict
- AST-based automation for test updates

## Problem Statement

1. The `metric` table has a commented-out `dataset` field that should store which dataset each metric belongs to
2. When metrics are built in the analyzer, no dataset information is provided
3. After persistence, we cannot determine which dataset a metric came from
4. This blocks features that need to query metrics by dataset

## Solution Overview

Add dataset tracking throughout the metric lifecycle:
1. Make dataset field mandatory in Metric dataclass
2. Update database schema to include non-nullable dataset column
3. Pass dataset name from analyzer (using `ds.name`)
4. Update all test code using AST automation

## Implementation Steps

### Step 1: Update Metric Model
**File**: `src/dqx/models.py`

#### 1.1 Add mandatory dataset field to Metric dataclass
```python
@dataclass
class Metric:
    spec: specs.MetricSpec
    state: State
    key: ResultKey
    dataset: str  # NEW: Mandatory field
    metric_id: uuid.UUID | None = None
```

#### 1.2 Update Metric.build() to require dataset
```python
@classmethod
def build(
    cls,
    metric: specs.MetricSpec,
    key: ResultKey,
    dataset: str,  # NEW: Required parameter
    state: State | None = None,
    metric_id: uuid.UUID | None = None,
) -> Self:
    return cls(
        metric_id=metric_id,
        spec=metric,
        state=state or metric.state(),
        key=key,
        dataset=dataset,  # NEW
    )
```

#### 1.3 Update test files using AST automation

Create and run `scripts/update_metric_build_calls.py` to automatically update ~30 test instances across the codebase.

**Script Overview:**

The AST-based automation script will parse Python files, identify all `Metric.build()` calls, and add the required `dataset` parameter. This approach ensures consistent updates while preserving code formatting.

**Script Details:**

```python
#!/usr/bin/env python3
"""
AST-based script to update Metric.build() calls to include dataset parameter.

This script will:
1. Find all calls to models.Metric.build() or Metric.build()
2. Add dataset parameter based on context
3. Preserve code formatting as much as possible
"""
```

**Key Features:**

1. **Pattern Detection**:
   - Finds `models.Metric.build()` calls
   - Finds direct `Metric.build()` calls
   - Handles both forms automatically

2. **Parameter Handling**:
   - Detects existing positional arguments
   - Checks for existing `dataset` keyword argument
   - Adds dataset as 3rd positional arg when appropriate
   - Falls back to keyword argument when needed

3. **Dataset Value Determination**:
   - Uses file path context to determine appropriate dataset name
   - Common patterns:
     - `test_analyzer*.py` → `"test_data"`
     - `test_provider*.py` → `"test_dataset"`
     - `test_*.py` (generic) → `"test_ds"`
     - `orm/test_*.py` → `"test_dataset"`
     - Default fallback → `"dataset"`

4. **Target Files**:
   ```python
   test_dirs = [
       Path("tests/test_analyzer.py"),
       Path("tests/test_provider.py"),
       Path("tests/orm/test_repositories.py"),
   ]
   ```

**Usage Instructions:**

1. **Install Dependencies**:
   ```bash
   pip install astor  # Required for AST to source conversion
   ```

2. **Run the Script**:
   ```bash
   cd /path/to/dqx
   python scripts/update_metric_build_calls.py
   ```

3. **Review Output**:
   - Script will show progress for each file
   - Lists all updated files at the end
   - Provides warnings for files not found

4. **Manual Review**:
   - Review the changes with `git diff`
   - Adjust dataset values if needed based on actual test context
   - Run tests to ensure correctness

**Example Transformations:**

Before:
```python
# Positional args only
metric = models.Metric.build(spec, key)

# With state
metric = models.Metric.build(spec, key, state)

# With keyword args
metric = models.Metric.build(spec, key, state=some_state)
```

After:
```python
# Positional args only - adds as 3rd positional
metric = models.Metric.build(spec, key, "test_data")

# With state - adds as keyword
metric = models.Metric.build(spec, key, state, dataset="test_data")

# With keyword args - adds as keyword
metric = models.Metric.build(spec, key, state=some_state, dataset="test_data")
```

**Important Notes:**

- The script uses heuristics for dataset naming based on file patterns
- Manual review is recommended to ensure dataset names match test intent
- The script preserves existing code structure as much as possible
- Run tests after transformation to verify correctness

### Step 2: Update Database Schema
**File**: `src/dqx/orm/schema.py`

#### 2.1 Uncomment and make dataset field non-nullable
```python
sa.Column("dataset", sa.String, nullable=False),  # Changed from commented out
```

#### 2.2 Update from_orm method
```python
@classmethod
def from_orm(cls, orm_metric: "dqx.orm.schema.Metric") -> Self:
    return cls(
        metric_id=orm_metric.metric_id,
        spec=MetricSpec.from_json(orm_metric.spec),
        state=State.from_json(orm_metric.state),
        key=ResultKey.from_json(orm_metric.result_key),
        dataset=orm_metric.dataset,  # NEW
    )
```

### Step 3: Update Analyzer
**File**: `src/dqx/analyzer.py`

#### 3.1 Update analyze method signature (no changes needed)
The analyze method already receives the SqlDataSource which has the dataset name.

#### 3.2 Update _analyze_internal to use dataset from datasource
```python
def _analyze_internal(
    self,
    ds: SqlDataSource,
    metrics_by_key: dict[ResultKey, Sequence[MetricSpec]],
) -> AnalysisReport:
    # ... existing code ...

    # Phase 5: Build report
    report_data = {}
    for key, metrics in metrics_by_key.items():
        for metric in metrics:
            report_data[(metric, key)] = models.Metric.build(
                metric,
                key,
                dataset=ds.name  # NEW: Use datasource name
            )

    return AnalysisReport(data=report_data)
```

### Step 4: Update MetricProvider
**File**: `src/dqx/provider.py`

#### 4.1 Add dataset mapping method
```python
def build_dataset_mapping(self) -> dict[MetricSpec, str]:
    """Build mapping from MetricSpec to dataset name.

    Returns:
        Dict mapping each MetricSpec to its dataset name.
        Only includes metrics that have a dataset assigned.
    """
    mapping = {}
    for sym_metric in self.symbolic_metrics:
        if sym_metric.dataset:
            mapping[sym_metric.metric_spec] = sym_metric.dataset
    return mapping
```

### Step 5: Update ORM Repository
**File**: `src/dqx/orm/repositories.py`

#### 5.1 Update _adapt_to_orm to include dataset
```python
def _adapt_to_orm(self, metric: models.Metric) -> schema.Metric:
    return schema.Metric(
        metric_id=metric.metric_id or self.metric_id(),
        spec=metric.spec.to_json(),
        state=metric.state.to_json(),
        result_key=metric.key.to_json(),
        dataset=metric.dataset,  # NEW
    )
```

#### 5.2 Update _build_from_orm (if it exists)
```python
def _build_from_orm(self, orm_row: Any) -> models.Metric | None:
    # ... existing code ...
    return models.Metric.build(
        spec,
        key,
        dataset=orm_row.dataset,  # NEW
        state=state,
        metric_id=self.metric_id
    )
```

### Step 6: Simplify VerificationSuite API
**File**: `src/dqx/api.py`

#### 6.1 Update _analyze signature to use list
```python
def _analyze(self, datasources: list[SqlDataSource], key: ResultKey) -> None:
    """Analyze metrics for all datasources.

    Args:
        datasources: List of data sources to analyze
        key: Result key for the analysis
    """
    for ds in datasources:
        # Get all symbolic metrics for this dataset
        symbolic_metrics = self._context.pending_metrics(ds.name)

        # Skip if no metrics for this dataset
        if not symbolic_metrics:
            continue

        # Group metrics by their effective date
        metrics_by_date: dict[ResultKey, list[MetricSpec]] = defaultdict(list)
        for sym_metric in symbolic_metrics:
            effective_key = sym_metric.key_provider.create(key)
            metrics_by_date[effective_key].append(sym_metric.metric_spec)

        # Analyze each date group separately
        logger.info(f"Analyzing dataset '{ds.name}'...")
        analyzer = Analyzer()
        analyzer.analyze(ds, metrics_by_date)

        # Persist the combined report
        analyzer.report.persist(self.provider._db)
```

#### 6.2 Update run method to call simplified _analyze
```python
def run(self, datasources: list[SqlDataSource], key: ResultKey, *, enable_plugins: bool = True) -> None:
    # ... existing validation code ...

    # Remove the dict creation, just pass the list
    logger.info(f"Running verification suite '{self._name}' with datasets: {[ds.name for ds in datasources]}")

    # ... rest of the method ...

    # 2. Analyze by datasources
    with self._analyze_ms:
        self._analyze(datasources, key)  # Pass list directly

    # ... rest remains the same ...
```

## Testing Strategy

1. **Unit Tests**: Update using AST automation script
2. **Integration Tests**: Verify dataset is persisted and retrievable
3. **E2E Tests**: Ensure existing functionality remains intact

## Migration Notes

Since we're not maintaining backward compatibility:
1. This is a breaking change - all code creating Metric objects must be updated
2. Existing metrics in the database will need migration to add dataset values
3. The AST automation script will handle test updates

## Automation Script

The `scripts/update_metric_build_calls.py` script will:
1. Parse Python files using AST
2. Find all `Metric.build()` calls
3. Add dataset parameter based on context
4. Handle both positional and keyword arguments
5. Preserve code formatting as much as possible

## Benefits

1. **Complete traceability**: Every metric can be traced back to its source dataset
2. **Enables dataset-specific queries**: Can filter metrics by dataset in database queries
3. **Cleaner API**: Simplified method signatures without redundant parameters
4. **Better separation of concerns**: Dataset mapping in provider, not scattered
5. **Type safety**: Mandatory fields caught at compile time

## Risks and Mitigations

1. **Risk**: Breaking existing code
   - **Mitigation**: AST automation handles test updates systematically

2. **Risk**: Database migration complexity
   - **Mitigation**: Can be handled separately with appropriate migration script

3. **Risk**: Missing dataset information in some code paths
   - **Mitigation**: Mandatory parameter ensures compile-time safety
