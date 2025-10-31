"""Test metadata persistence through the Analyzer and MetricDB."""

import datetime

import pyarrow as pa
import pytest

from dqx.analyzer import Analyzer
from dqx.common import Metadata, ResultKey
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB
from dqx.specs import MetricSpec, NullCount, NumRows


class TestMetadataPersistence:
    """Test metadata flows through Analyzer to database."""

    @pytest.fixture
    def db(self) -> InMemoryMetricDB:
        """Create in-memory database."""
        return InMemoryMetricDB()

    @pytest.fixture
    def ds(self) -> DuckRelationDataSource:
        """Create simple data source."""
        data = pa.table(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
            }
        )
        return DuckRelationDataSource.from_arrow(data, "test_data")

    def test_analyzer_with_metadata(self, ds: DuckRelationDataSource, db: InMemoryMetricDB) -> None:
        """Test Analyzer persists metadata correctly."""
        # Create analyzer with execution_id
        from dqx.provider import MetricProvider

        execution_id = "test-run-123"
        provider = MetricProvider(db, execution_id=execution_id)
        key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})

        analyzer = Analyzer(datasources=[ds], provider=provider, key=key, execution_id=execution_id)

        # Define metrics
        metrics: dict[ResultKey, list[MetricSpec]] = {
            key: [
                NumRows(),
                NullCount("name"),
            ]
        }

        # Analyze
        report = analyzer.analyze_simple_metrics(ds, metrics)

        # Persist
        report.persist(db)

        # Verify metadata was persisted
        for metric_spec, result_key, dataset_name in report:
            metric = db.get(result_key, metric_spec)
            assert metric is not None
            persisted = metric.unwrap()
            assert persisted.metadata is not None
            assert persisted.metadata.execution_id == "test-run-123"
            # Default ttl_hours is 168, not 48
            assert persisted.metadata.ttl_hours == 168

    def test_analyzer_without_metadata(self, ds: DuckRelationDataSource, db: InMemoryMetricDB) -> None:
        """Test Analyzer creates metadata with execution_id."""
        # Create analyzer with minimal required parameters
        from dqx.provider import MetricProvider

        # Use empty string instead of None for execution_id
        provider = MetricProvider(db, execution_id="")
        key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
        # When no execution_id is provided, a UUID is generated
        analyzer = Analyzer(datasources=[ds], provider=provider, key=key, execution_id="")

        # Define metrics
        metrics: dict[ResultKey, list[MetricSpec]] = {key: [NumRows()]}

        # Analyze and persist
        report = analyzer.analyze_simple_metrics(ds, metrics)
        report.persist(db)

        # Verify metadata with execution_id was created
        for metric_spec, result_key, dataset_name in report:
            metric = db.get(result_key, metric_spec)
            assert metric is not None
            persisted = metric.unwrap()
            assert persisted.metadata is not None
            # execution_id is empty string when passed as empty string
            assert persisted.metadata.execution_id == ""
            assert persisted.metadata.ttl_hours == 168  # Default is 7 days

    def test_metadata_roundtrip(self, ds: DuckRelationDataSource) -> None:
        """Test metadata survives full roundtrip."""
        # Create two databases to simulate real scenario
        db1 = InMemoryMetricDB()
        db2 = InMemoryMetricDB()

        # Analyze with execution_id and persist to db1
        from dqx.provider import MetricProvider

        execution_id = "roundtrip-test"
        provider = MetricProvider(db1, execution_id=execution_id)
        key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})

        analyzer = Analyzer(datasources=[ds], provider=provider, key=key, execution_id=execution_id)

        metrics: dict[ResultKey, list[MetricSpec]] = {key: [NumRows()]}

        report = analyzer.analyze_simple_metrics(ds, metrics)
        persisted_metrics = list(db1.persist(report.values()))

        # Verify persisted metrics have metadata
        assert len(persisted_metrics) == 1
        expected_metadata = Metadata(execution_id="roundtrip-test", ttl_hours=168)  # Default ttl
        assert persisted_metrics[0].metadata == expected_metadata

        # Simulate retrieving from db1 and persisting to db2
        for metric in persisted_metrics:
            if metric.metric_id is not None:
                retrieved = db1.get(metric.metric_id)
                assert retrieved is not None

                # Persist to second database
                db2.persist([retrieved.unwrap()])

                # Verify in second database
                from_db2 = db2.get(metric.metric_id)
                assert from_db2 is not None
                assert from_db2.unwrap().metadata == expected_metadata
