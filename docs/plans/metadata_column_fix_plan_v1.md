# Metadata Column Implementation Plan v1

## Overview
This plan addresses the issue where `print_symbols()` fails after `run()` because it tries to retrieve metrics with an `execution_id` tag that doesn't exist in the stored metrics. The solution introduces a metadata column in the database to track execution context without polluting user tags.

**IMPORTANT**: This is a clean implementation with NO backward compatibility or database migration. We will modify the schema directly.

## Problem Summary
- VerificationSuite generates a unique `execution_id` for each run
- This `execution_id` is injected into tags during metric storage
- MetricProvider tries to retrieve metrics using this `execution_id` tag
- Since metrics are stored without the `execution_id` in their tags, retrieval fails
- Additionally, when multiple runs exist for the same key, retrieval is non-deterministic

## Solution Design
1. Add a `metadata` column to the Metric table for runtime context
2. Store `execution_id` and `ttl_hours` in metadata, not in tags
3. Modify retrieval methods to return the latest metric when multiple exist
4. Remove all `execution_id` tag injection from the codebase

## Implementation Tasks

### Task Group 1: Core Infrastructure (TDD)

#### Task 1.1: Add Metadata dataclass
**File**: `src/dqx/common.py`

Add after the Tags type alias:
```python
from dataclasses import dataclass

@dataclass
class Metadata:
    """Metadata for metric lifecycle and execution context."""
    execution_id: str | None = None
    ttl_hours: int = 168  # 7 days default
```

#### Task 1.2: Create tests for MetadataType
**File**: `tests/orm/test_metadata_type.py` (new file)

```python
"""Tests for MetadataType custom type decorator."""

import pytest
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import Session, declarative_base

from dqx.common import Metadata
from dqx.orm.repositories import MetadataType


Base = declarative_base()


class TestModel(Base):
    """Test model with metadata column."""
    __tablename__ = "test_model"

    id = Column(Integer, primary_key=True)
    metadata = Column(MetadataType, nullable=False, default=Metadata)


class TestMetadataType:
    """Test MetadataType serialization and deserialization."""

    @pytest.fixture
    def session(self) -> Session:
        """Create test database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = Session(engine)
        yield session
        session.close()

    def test_serialize_none(self) -> None:
        """Test serializing None metadata."""
        metadata_type = MetadataType()
        result = metadata_type.process_bind_param(None, None)
        assert result == {}

    def test_serialize_metadata(self) -> None:
        """Test serializing Metadata object."""
        metadata = Metadata(execution_id="test-123", ttl_hours=72)
        metadata_type = MetadataType()
        result = metadata_type.process_bind_param(metadata, None)
        assert result == {"execution_id": "test-123", "ttl_hours": 72}

    def test_deserialize_none(self) -> None:
        """Test deserializing None value."""
        metadata_type = MetadataType()
        result = metadata_type.process_result_value(None, None)
        assert result == Metadata()

    def test_deserialize_dict(self) -> None:
        """Test deserializing dictionary to Metadata."""
        metadata_type = MetadataType()
        value = {"execution_id": "test-456", "ttl_hours": 48}
        result = metadata_type.process_result_value(value, None)
        assert result == Metadata(execution_id="test-456", ttl_hours=48)

    def test_roundtrip_with_db(self, session: Session) -> None:
        """Test full roundtrip through database."""
        # Create record with metadata
        metadata = Metadata(execution_id="db-test", ttl_hours=24)
        record = TestModel(metadata=metadata)
        session.add(record)
        session.commit()

        # Retrieve and verify
        retrieved = session.query(TestModel).first()
        assert retrieved is not None
        assert retrieved.metadata == metadata

    def test_default_metadata(self, session: Session) -> None:
        """Test default metadata creation."""
        record = TestModel()
        session.add(record)
        session.commit()

        retrieved = session.query(TestModel).first()
        assert retrieved is not None
        assert retrieved.metadata == Metadata()
```

#### Task 1.3: Implement MetadataType
**File**: `src/dqx/orm/repositories.py`

Add imports:
```python
from dataclasses import asdict
from sqlalchemy.types import TypeDecorator, JSON
```

Add after imports, before the Metric class:
```python
class MetadataType(TypeDecorator):
    """Custom type to handle Metadata dataclass serialization."""
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: Metadata | None, dialect: Any) -> dict[str, Any]:
        """Convert Metadata to JSON-serializable dict."""
        if value is None:
            return {}
        if isinstance(value, Metadata):
            return asdict(value)
        return value

    def process_result_value(self, value: dict[str, Any] | None, dialect: Any) -> Metadata:
        """Convert JSON dict back to Metadata."""
        if value is None:
            return Metadata()
        return Metadata(**value)
```

**Commit after Task Group 1**: Run tests and commit with message "feat: add Metadata dataclass and MetadataType"

### Task Group 2: Database Schema Update (TDD)

#### Task 2.1: Test Metric table with metadata column
**File**: `tests/orm/test_metric_metadata.py` (new file)

```python
"""Tests for Metric table metadata column."""

import datetime
import pytest
from sqlalchemy.orm import Session

from dqx.common import Metadata, ResultKey
from dqx.models import Metric as MetricModel
from dqx.orm.repositories import Metric, MetricDB
from dqx.specs import Average


class TestMetricMetadata:
    """Test Metric table metadata functionality."""

    def test_metric_create_with_metadata(self, db_session: Session) -> None:
        """Test creating metric with metadata."""
        key = ResultKey(datetime.date(2024, 1, 1))
        spec = Average("price")
        metric_model = MetricModel.build(spec, key, "test_dataset")

        metadata = Metadata(execution_id="exec-123", ttl_hours=48)

        # Create ORM object
        metric = Metric(
            yyyy_mm_dd=key.yyyy_mm_dd,
            tags=key.tags,
            spec=spec,
            dataset="test_dataset",
            value=metric_model.value,
            metadata=metadata
        )

        db_session.add(metric)
        db_session.commit()

        # Verify
        retrieved = db_session.query(Metric).first()
        assert retrieved is not None
        assert retrieved.metadata == metadata

    def test_metric_default_metadata(self, db_session: Session) -> None:
        """Test metric creation with default metadata."""
        key = ResultKey(datetime.date(2024, 1, 1))
        spec = Average("price")
        metric_model = MetricModel.build(spec, key, "test_dataset")

        metric = Metric(
            yyyy_mm_dd=key.yyyy_mm_dd,
            tags=key.tags,
            spec=spec,
            dataset="test_dataset",
            value=metric_model.value
        )

        db_session.add(metric)
        db_session.commit()

        retrieved = db_session.query(Metric).first()
        assert retrieved is not None
        assert retrieved.metadata == Metadata()
```

#### Task 2.2: Add metadata column to Metric
**File**: `src/dqx/orm/repositories.py`

In the Metric class, add after the `created` column:
```python
    metadata: Mapped[Metadata] = mapped_column(
        MetadataType,
        nullable=False,
        default=Metadata,
        doc="Runtime metadata for execution context"
    )
```

#### Task 2.3: Test persist with metadata
**File**: `tests/orm/test_persist_with_metadata.py` (new file)

```python
"""Tests for MetricDB persist with metadata."""

import datetime
import pytest

from dqx.common import Metadata, ResultKey
from dqx.models import Metric as MetricModel
from dqx.orm.repositories import MetricDB
from dqx.specs import Average, Maximum


class TestPersistWithMetadata:
    """Test MetricDB persist method with metadata parameter."""

    def test_persist_single_metric_with_metadata(self, metric_db: MetricDB) -> None:
        """Test persisting a single metric with metadata."""
        key = ResultKey(datetime.date(2024, 1, 1))
        spec = Average("price")
        metric = MetricModel.build(spec, key, "test_dataset")
        metadata = Metadata(execution_id="test-exec", ttl_hours=24)

        persisted = list(metric_db.persist([metric], metadata))

        assert len(persisted) == 1
        # Verify in DB
        retrieved = metric_db.get(key, spec)
        assert retrieved is not None
        # Note: get() returns MetricModel, not ORM object, so we can't check metadata here

    def test_persist_multiple_metrics_same_metadata(self, metric_db: MetricDB) -> None:
        """Test persisting multiple metrics with same metadata."""
        key = ResultKey(datetime.date(2024, 1, 1))
        metrics = [
            MetricModel.build(Average("price"), key, "test_dataset"),
            MetricModel.build(Maximum("quantity"), key, "test_dataset")
        ]
        metadata = Metadata(execution_id="batch-exec", ttl_hours=48)

        persisted = list(metric_db.persist(metrics, metadata))

        assert len(persisted) == 2
        # All should be retrievable
        assert metric_db.get(key, Average("price")) is not None
        assert metric_db.get(key, Maximum("quantity")) is not None

    def test_persist_without_metadata(self, metric_db: MetricDB) -> None:
        """Test backward compatibility - persist without metadata."""
        key = ResultKey(datetime.date(2024, 1, 1))
        spec = Average("price")
        metric = MetricModel.build(spec, key, "test_dataset")

        persisted = list(metric_db.persist([metric]))

        assert len(persisted) == 1
        retrieved = metric_db.get(key, spec)
        assert retrieved is not None
```

**Commit after Task Group 2**: Run tests and commit with message "feat: add metadata column to Metric table"

### Task Group 3: Update Persistence Methods

#### Task 3.1: Update persist method signature
**File**: `src/dqx/orm/repositories.py`

Update the persist method:
```python
def persist(self, metrics: Iterable[models.Metric], metadata: Metadata | None = None) -> Iterable[models.Metric]:
    """
    Persist metrics to the database.

    Args:
        metrics: Iterable of Metric models to persist
        metadata: Optional metadata for execution context

    Returns:
        The persisted metrics
    """
    actual_metadata = metadata or Metadata()

    with self.transact() as session:
        for m in metrics:
            metric = Metric(
                yyyy_mm_dd=m.key.yyyy_mm_dd,
                tags=m.key.tags,
                spec=m.spec,
                dataset=m.dataset,
                value=m.value,
                metadata=actual_metadata
            )
            self._persist(session, metric)
            yield m
```

#### Task 3.2: Test retrieval with multiple runs
**File**: `tests/orm/test_retrieval_latest.py` (new file)

```python
"""Tests for retrieving latest metric when multiple runs exist."""

import datetime
import time
import pytest

from dqx.common import Metadata, ResultKey
from dqx.models import Metric as MetricModel
from dqx.orm.repositories import MetricDB
from dqx.specs import Average


class TestRetrievalLatest:
    """Test that retrieval methods return the latest metric."""

    def test_get_returns_latest(self, metric_db: MetricDB) -> None:
        """Test get() returns latest metric when multiple exist."""
        key = ResultKey(datetime.date(2024, 1, 1))
        spec = Average("price")

        # Persist same metric multiple times with different values
        for i in range(3):
            metric = MetricModel(spec=spec, key=key, dataset="test", value=float(i))
            metadata = Metadata(execution_id=f"exec-{i}")
            list(metric_db.persist([metric], metadata))
            time.sleep(0.01)  # Ensure different timestamps

        # Should get the latest (value=2.0)
        retrieved = metric_db.get(key, spec)
        assert retrieved is not None
        assert retrieved.value == 2.0

    def test_get_metric_value_returns_latest(self, metric_db: MetricDB) -> None:
        """Test get_metric_value() returns latest when multiple exist."""
        key = ResultKey(datetime.date(2024, 1, 1))
        spec = Average("quantity")

        # Persist multiple versions
        for i in range(3):
            metric = MetricModel(spec=spec, key=key, dataset="test", value=float(i * 10))
            metadata = Metadata(execution_id=f"run-{i}")
            list(metric_db.persist([metric], metadata))
            time.sleep(0.01)

        # Should get latest value (20.0)
        value = metric_db.get_metric_value(key, spec)
        assert value == 20.0

    def test_get_metric_window_returns_latest_per_date(self, metric_db: MetricDB) -> None:
        """Test get_metric_window() returns latest metric per date."""
        spec = Average("revenue")
        base_date = datetime.date(2024, 1, 1)

        # Create metrics for 3 days, each with 2 runs
        for day_offset in range(3):
            key = ResultKey(base_date + datetime.timedelta(days=day_offset))

            # First run
            metric1 = MetricModel(spec=spec, key=key, dataset="test", value=float(day_offset * 100))
            list(metric_db.persist([metric1], Metadata(execution_id=f"run1-day{day_offset}")))
            time.sleep(0.01)

            # Second run (should be returned)
            metric2 = MetricModel(spec=spec, key=key, dataset="test", value=float(day_offset * 100 + 50))
            list(metric_db.persist([metric2], Metadata(execution_id=f"run2-day{day_offset}")))
            time.sleep(0.01)

        # Get window
        results = metric_db.get_metric_window(
            ResultKey(base_date),
            ResultKey(base_date + datetime.timedelta(days=2)),
            spec
        )

        # Should have 3 results, all from second run
        assert len(results) == 3
        assert results[0].value == 50.0   # day 0, run 2
        assert results[1].value == 150.0  # day 1, run 2
        assert results[2].value == 250.0  # day 2, run 2
```

#### Task 3.3: Update retrieval methods
**File**: `src/dqx/orm/repositories.py`

Update `_get_by_key` to order by created desc:
```python
def _get_by_key(self, session: Session, key: models.ResultKey, spec: MetricSpec) -> Metric | None:
    """Get metric by key, returning latest if multiple exist."""
    spec_id = spec.sql_id()
    return (
        session.query(Metric)
        .filter(
            Metric.yyyy_mm_dd == key.yyyy_mm_dd,
            Metric.tags == key.tags,
            Metric.spec == spec_id,
        )
        .order_by(Metric.created.desc())
        .limit(1)
        .one_or_none()
    )
```

Update `get_metric_value` to order by created desc:
```python
def get_metric_value(self, key: models.ResultKey, spec: MetricSpec) -> float | None:
    """
    Get the metric value for the given key and spec.
    Returns latest value if multiple exist.
    """
    spec_id = spec.sql_id()
    with self.transact() as session:
        result = (
            session.query(Metric.value)
            .filter(
                Metric.yyyy_mm_dd == key.yyyy_mm_dd,
                Metric.tags == key.tags,
                Metric.spec == spec_id,
            )
            .order_by(Metric.created.desc())
            .limit(1)
            .scalar()
        )
        return result
```

Update `get_metric_window` to use a subquery for latest per date:
```python
def get_metric_window(
    self, start_key: models.ResultKey, end_key: models.ResultKey, spec: MetricSpec
) -> list[models.Metric]:
    """
    Get metrics within a date window, returning latest per date if multiple exist.
    """
    spec_id = spec.sql_id()
    with self.transact() as session:
        # Subquery to get max created timestamp per date
        from sqlalchemy import func

        latest_subq = (
            session.query(
                Metric.yyyy_mm_dd,
                func.max(Metric.created).label("max_created")
            )
            .filter(
                Metric.yyyy_mm_dd >= start_key.yyyy_mm_dd,
                Metric.yyyy_mm_dd <= end_key.yyyy_mm_dd,
                Metric.tags == start_key.tags,
                Metric.spec == spec_id,
            )
            .group_by(Metric.yyyy_mm_dd)
            .subquery()
        )

        # Main query joining with subquery
        metrics = (
            session.query(Metric)
            .join(
                latest_subq,
                sa.and_(
                    Metric.yyyy_mm_dd == latest_subq.c.yyyy_mm_dd,
                    Metric.created == latest_subq.c.max_created
                )
            )
            .filter(
                Metric.tags == start_key.tags,
                Metric.spec == spec_id,
            )
            .order_by(Metric.yyyy_mm_dd)
            .all()
        )

        return [m.unwrap() for m in metrics]
```

**Commit after Task Group 3**: Run tests and commit with message "feat: update persistence and retrieval for metadata support"

### Task Group 4: Update Analyzer and AnalysisReport

#### Task 4.1: Test AnalysisReport with metadata
**File**: `tests/test_analyzer_metadata.py` (new file)

```python
"""Tests for Analyzer with metadata support."""

import datetime
import pytest
from unittest.mock import Mock, patch

from dqx.analyzer import AnalysisReport
from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import MetricDB
from dqx.specs import Average


class TestAnalysisReportMetadata:
    """Test AnalysisReport metadata handling."""

    def test_analysis_report_init_with_metadata(self) -> None:
        """Test creating AnalysisReport with metadata."""
        metadata = Metadata(execution_id="test-123", ttl_hours=24)
        report = AnalysisReport(metadata=metadata)

        assert report.metadata == metadata
        assert len(report) == 0

    def test_analysis_report_persist_passes_metadata(self) -> None:
        """Test persist passes metadata to db.persist()."""
        # Create report with metadata
        metadata = Metadata(execution_id="persist-test", ttl_hours=48)
        report = AnalysisReport(metadata=metadata)

        # Add a metric
        key = ResultKey(datetime.date(2024, 1, 1))
        spec = Average("price")
        metric = Metric.build(spec, key, "test_dataset")
        report[(spec, key)] = metric

        # Mock db
        mock_db = Mock(spec=MetricDB)
        mock_db.persist = Mock(return_value=iter([]))

        # Call persist
        report.persist(mock_db)

        # Verify metadata was passed
        mock_db.persist.assert_called_once()
        args = mock_db.persist.call_args
        assert args[0][0] == report.values()  # First arg is metrics
        assert args[1]["metadata"] == metadata  # metadata passed as kwarg

    def test_analysis_report_merge_preserves_metadata(self) -> None:
        """Test merge preserves metadata from first report."""
        metadata1 = Metadata(execution_id="report1", ttl_hours=24)
        metadata2 = Metadata(execution_id="report2", ttl_hours=48)

        report1 = AnalysisReport(metadata=metadata1)
        report2 = AnalysisReport(metadata=metadata2)

        merged = report1.merge(report2)

        # Merged report should have metadata from report1
        assert merged.metadata == metadata1
```

#### Task 4.2: Update AnalysisReport
**File**: `src/dqx/analyzer.py`

Update AnalysisReport class:
```python
class AnalysisReport(UserDict[MetricKey, models.Metric]):
    def __init__(self, data: dict[MetricKey, models.Metric] | None = None, metadata: Metadata | None = None) -> None:
        self.data = data if data is not None else {}
        self.metadata = metadata

    def merge(self, other: AnalysisReport) -> AnalysisReport:
        """Merge two AnalysisReports, using Metric.reduce for conflicts.

        Preserves metadata from self.
        """
        # Start with a copy of self.data for efficiency
        merged_data = dict(self.data)

        # Merge items from other
        for key, metric in other.items():
            if key in merged_data:
                # Key exists in both: use Metric.reduce to merge
                merged_data[key] = models.Metric.reduce([merged_data[key], metric])
            else:
                # Key only in other: just add it
                merged_data[key] = metric

        return AnalysisReport(data=merged_data, metadata=self.metadata)

    def persist(self, db: MetricDB, overwrite: bool = True) -> None:
        """Persist the analysis report to the metric database."""
        if len(self) == 0:
            logger.warning("Try to save an EMPTY analysis report!")
            return

        if overwrite:
            logger.info("Overwriting analysis report ...")
            db.persist(self.values(), self.metadata)
        else:
            logger.info("Merging analysis report ...")
            self._merge_persist(db)

    def _merge_persist(self, db: MetricDB) -> None:
        """Merge with existing metrics in the database before persisting."""
        db_report = AnalysisReport()

        for key, metric in self.items():
            # Find the metric in DB
            db_metric = db.get(metric.key, metric.spec)
            if db_metric is not None:
                db_report[key] = db_metric.unwrap()

        # Merge and persist with metadata
        merged_report = self.merge(db_report)
        db.persist(merged_report.values(), self.metadata)
```

Add import at top:
```python
from dqx.common import (
    DQXError,
    Metadata,
    ResultKey,
    SqlDataSource,
)
```

#### Task 4.3: Update Analyzer to accept metadata
**File**: `src/dqx/analyzer.py`

Update Analyzer._analyze_internal to create report with metadata:
```python
def _analyze_internal(
    self,
    ds: SqlDataSource,
    metrics_by_key: dict[ResultKey, Sequence[MetricSpec]],
    metadata: Metadata | None = None
) -> AnalysisReport:
    """Process a single batch of dates with metadata."""
    from collections import defaultdict

    # ... existing code ...

    # Phase 5: Build report with metadata
    report_data = {}
    for key, metrics in metrics_by_key.items():
        for metric in metrics:
            report_data[(metric, key)] = models.Metric.build(metric, key, dataset=ds.name)

    return AnalysisReport(data=report_data, metadata=metadata)
```

Update Analyzer.analyze to accept and pass metadata:
```python
def analyze(
    self,
    ds: SqlDataSource,
    metrics: Mapping[ResultKey, Sequence[MetricSpec]],
    metadata: Metadata | None = None
) -> AnalysisReport:
    """Analyze multiple dates with metadata support."""
    # ... existing code ...

    # Create final report with metadata at the beginning
    final_report = AnalysisReport(metadata=metadata)

    # Process in batches if needed
    items = list(metrics.items())

    for i in range(0, len(items), DEFAULT_BATCH_SIZE):
        batch_items = items[i : i + DEFAULT_BATCH_SIZE]
        batch = dict(batch_items)

        # ... existing code ...

        report = self._analyze_internal(ds, batch, metadata)
        # Merge directly into final report
        final_report = final_report.merge(report)

    self._report = self._report.merge(final_report)

    return self._report
```

**Commit after Task Group 4**: Run tests and commit with message "feat: add metadata support to Analyzer and AnalysisReport"

### Task Group 5: Update VerificationSuite (Remove execution_id injection)

#### Task 5.1: Test VerificationSuite with ttl_hours
**File**: `tests/test_api_metadata.py` (new file)

```python
"""Tests for VerificationSuite metadata handling."""

import datetime
import pytest
from unittest.mock import Mock, patch

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from dqx.orm.repositories import MetricDB
from dqx.specs import Average


class TestVerificationSuiteMetadata:
    """Test VerificationSuite metadata features."""

    def test_suite_init_with_ttl_hours(self, metric_db: MetricDB) -> None:
        """Test creating suite with custom ttl_hours."""
        @check(name="Test Check")
        def test_check(mp, ctx):
            pass

        suite = VerificationSuite([test_check], metric_db, "Test Suite", ttl_hours=48)

        assert suite._ttl_hours == 48

    def test_suite_default_ttl_hours(self, metric_db: MetricDB) -> None:
        """Test suite uses default ttl_hours."""
        @check(name="Test Check")
        def test_check(mp, ctx):
            pass

        suite = VerificationSuite([test_check], metric_db, "Test Suite")

        assert suite._ttl_hours == 168  # 7 days

    @patch('dqx.analyzer.Analyzer.analyze')
    def test_analyze_passes_metadata(self, mock_analyze, metric_db: MetricDB) -> None:
        """Test _analyze creates metadata and passes to analyzer."""
        @check(name="Test Check")
        def test_check(mp, ctx):
            ctx.assert_that(mp.average("price")).where(name="Positive").is_positive()

        suite = VerificationSuite([test_check], metric_db, "Test Suite", ttl_hours=24)

        # Mock datasource
        mock_ds = Mock()
        mock_ds.name = "test_ds"

        # Run suite
        key = ResultKey(datetime.date(2024, 1, 1))
        suite.run([mock_ds], key)

        # Verify metadata was created and passed
        mock_analyze.assert_called()
        call_args = mock_analyze.call_args
        metadata = call_args[1]["metadata"]

        assert metadata is not None
        assert metadata.execution_id == suite.execution_id
        assert metadata.ttl_hours == 24
```

#### Task 5.2: Update VerificationSuite
**File**: `src/dqx/api.py`

Add ttl_hours parameter to __init__:
```python
def __init__(
    self,
    checks: Sequence[CheckProducer | DecoratedCheck],
    db: MetricDB,
    name: str,
    ttl_hours: int = 168,  # 7 days default
) -> None:
    """
    Initialize the verification suite.

    Args:
        checks: Sequence of check functions to execute
        db: Database for storing and retrieving metrics
        name: Human-readable name for the suite
        ttl_hours: Time-to-live for metrics in hours (default 168 = 7 days)

    Raises:
        DQXError: If no checks provided or name is empty
    """
    # ... existing validation ...

    self._ttl_hours = ttl_hours

    # ... rest of existing code ...
```

Update _analyze to remove execution_id injection and pass metadata:
```python
def _analyze(self, datasources: list[SqlDataSource], key: ResultKey) -> None:
    # Create metadata for this run
    metadata = Metadata(execution_id=self._execution_id, ttl_hours=self._ttl_hours)

    for ds in datasources:
        # Get all symbolic metrics for this dataset
        symbolic_metrics = self._context.pending_metrics(ds.name)

        # Skip if no metrics for this dataset
        if not symbolic_metrics:
            continue

        # Group metrics by their effective date (NO execution_id injection)
        metrics_by_date: dict[ResultKey, list[MetricSpec]] = defaultdict(list)
        for sym_metric in symbolic_metrics:
            # Create the effective key from the provider
            effective_key = sym_metric.key_provider.create(key)
            metrics_by_date[effective_key].append(sym_metric.metric_spec)

        # Analyze each date group separately
        logger.info(f"Analyzing dataset '{ds.name}'...")
        analyzer = Analyzer()
        analyzer.analyze(ds, metrics_by_date, metadata)

        # Persist the combined report
        analyzer.report.persist(self.provider._db)
```

Update run to remove execution_id from evaluator:
```python
def run(self, datasources: list[SqlDataSource], key: ResultKey, *, enable_plugins: bool = True) -> None:
    # ... existing code up to evaluation ...

    # 3. Evaluate assertions (NO execution_id injection)
    evaluator = Evaluator(self.provider, key, self._name)
    self._context._graph.bfs(evaluator)

    # ... rest of existing code ...
```

Add import:
```python
from dqx.common import (
    # ... existing imports ...
    Metadata,
)
```

#### Task 5.3: Test full flow without execution_id pollution
**File**: `tests/test_execution_id_removed.py` (new file)

```python
"""Test that execution_id is no longer in tags."""

import datetime
import pytest
from unittest.mock import patch

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey
from dqx.orm.repositories import MetricDB
from dqx.specs import Average
from dqx.common import DuckRelationDataSource


class TestExecutionIdRemoved:
    """Verify execution_id is not injected into tags."""

    def test_no_execution_id_in_persisted_tags(self, metric_db: MetricDB) -> None:
        """Test that persisted metrics don't have __execution_id in tags."""
        @check(name="Test Check", datasets=["test_ds"])
        def test_check(mp, ctx):
            ctx.assert_that(mp.average("value")).where(name="Positive").is_positive()

        suite = VerificationSuite([test_check], metric_db, "Test Suite")

        # Create data
        import pyarrow as pa
        data = pa.table({"value": [1.0, 2.0, 3.0]})
        ds = DuckRelationDataSource.from_arrow(data, "test_ds")

        # Run with tags
        key = ResultKey(datetime.date(2024, 1, 1), {"env": "prod"})
        suite.run([ds], key)

        # Check persisted metrics
        with metric_db.transact() as session:
            from dqx.orm.repositories import Metric
            metrics = session.query(Metric).all()

            # Verify no __execution_id in tags
            for metric in metrics:
                assert "__execution_id" not in metric.tags
                assert metric.tags == {"env": "prod"}

    @patch('dqx.evaluator.Evaluator')
    def test_evaluator_receives_clean_key(self, mock_evaluator_class, metric_db: MetricDB) -> None:
        """Test that Evaluator receives key without execution_id."""
        @check(name="Test Check")
        def test_check(mp, ctx):
            pass

        suite = VerificationSuite([test_check], metric_db, "Test Suite")

        mock_ds = Mock()
        mock_ds.name = "test_ds"

        key = ResultKey(datetime.date(2024, 1, 1), {"team": "analytics"})
        suite.run([mock_ds], key)

        # Verify Evaluator was called with clean key
        mock_evaluator_class.assert_called_once()
        call_args = mock_evaluator_class.call_args
        evaluator_key = call_args[0][1]  # Second positional arg

        assert evaluator_key == key
        assert "__execution_id" not in evaluator_key.tags
```

**Commit after Task Group 5**: Run tests and commit with message "feat: remove execution_id tag injection"

### Task Group 6: Integration Tests

#### Task 6.1: Test the original issue is fixed
**File**: `tests/e2e/test_print_symbols_after_run.py` (update existing)

The existing test should now pass without modification!

#### Task 6.2: Test multiple runs behavior
**File**: `tests/test_multiple_runs_integration.py` (new file)

```python
"""Integration tests for multiple suite runs."""

import datetime
import time
import pytest

from dqx.api import VerificationSuite, check
from dqx.common import ResultKey, DuckRelationDataSource
from dqx.orm.repositories import MetricDB
import pyarrow as pa


class TestMultipleRuns:
    """Test behavior with multiple suite runs."""

    def test_latest_run_metrics_used(self, metric_db: MetricDB) -> None:
        """Test that latest run's metrics are used."""
        @check(name="Price Check", datasets=["sales"])
        def price_check(mp, ctx):
            ctx.assert_that(mp.average("price")).where(name="Positive").is_positive()

        key = ResultKey(datetime.date(2024, 1, 1))

        # First run with low prices
        data1 = pa.table({"price": [1.0, 2.0, 3.0]})
        ds1 = DuckRelationDataSource.from_arrow(data1, "sales")
        suite1 = VerificationSuite([price_check], metric_db, "Suite 1")
        suite1.run([ds1], key)

        # Check first run metrics
        avg_spec = suite1.provider.average("price")
        first_value = metric_db.get_metric_value(key, avg_spec)
        assert first_value == 2.0

        time.sleep(0.01)  # Ensure different timestamp

        # Second run with high prices
        data2 = pa.table({"price": [10.0, 20.0, 30.0]})
        ds2 = DuckRelationDataSource.from_arrow(data2, "sales")
        suite2 = VerificationSuite([price_check], metric_db, "Suite 2")
        suite2.run([ds2], key)

        # Should get second run's value
        second_value = metric_db.get_metric_value(key, avg_spec)
        assert second_value == 20.0

    def test_print_symbols_uses_latest_run(self, metric_db: MetricDB) -> None:
        """Test print_symbols works with latest run."""
        @check(name="Stats Check", datasets=["data"])
        def stats_check(mp, ctx):
            ctx.assert_that(mp.average("value")).where(name="Avg > 0").is_positive()
            ctx.assert_that(mp.maximum("value")).where(name="Max > 5").is_gt(5)

        key = ResultKey(datetime.date(2024, 1, 1))

        # Run 1
        data1 = pa.table({"value": [1.0, 2.0, 3.0]})
        ds1 = DuckRelationDataSource.from_arrow(data1, "data")
        suite1 = VerificationSuite([stats_check], metric_db, "Suite 1")
        suite1.run([ds1], key)

        time.sleep(0.01)

        # Run 2 with different data
        data2 = pa.table({"value": [5.0, 10.0, 15.0]})
        ds2 = DuckRelationDataSource.from_arrow(data2, "data")
        suite2 = VerificationSuite([stats_check], metric_db, "Suite 2")
        suite2.run([ds2], key)

        # print_symbols should work and show latest values
        suite2.provider.print_symbols(key)

        # Verify latest values are retrieved
        symbols = suite2.provider.collect_symbols(key)

        symbol_values = {s.symbol: s.value for s in symbols}
        assert symbol_values["average('value')"] == 10.0
        assert symbol_values["maximum('value')"] == 15.0
```

**Commit after Task Group 6**: Run all tests and commit with message "test: add integration tests for metadata column"

### Task Group 7: Cleanup and Documentation

#### Task 7.1: Update existing tests
Remove any tests that expect `__execution_id` in tags.

#### Task 7.2: Run full test suite
```bash
uv run pytest tests/ -v
```

#### Task 7.3: Run pre-commit checks
```bash
bin/run-hooks.sh
```

#### Task 7.4: Update documentation
Add a note in the main documentation about the metadata column and its purpose.

**Final commit**: "docs: update documentation for metadata column feature"

## Summary

This implementation:
1. Adds a metadata column to store execution context without polluting user tags
2. Ensures retrieval methods return the latest metric when multiple runs exist
3. Removes all execution_id tag injection from the codebase
4. Fixes the original issue where `print_symbols()` fails after `run()`
5. Provides a clean implementation with no backward compatibility concerns

The solution is transparent to users and maintains the existing API while fixing the underlying issue.
