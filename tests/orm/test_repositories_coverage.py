"""Additional tests for repositories.py to achieve 100% coverage."""

import datetime as dt
from unittest.mock import Mock, patch

import pytest
from returns.maybe import Nothing
from sqlalchemy import create_engine

from dqx.common import Metadata, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB, MetadataType, MetricDB
from dqx.orm.session import db_session_factory
from dqx.specs import Average
from dqx.states import Average as AverageState


def test_metadata_type_process_bind_param_with_non_metadata_dict() -> None:
    """Test MetadataType.process_bind_param with dict value that's not None or Metadata."""
    metadata_type = MetadataType()

    # Test with a plain dict
    test_dict = {"execution_id": "test-123", "ttl_hours": 24}
    # Type ignore because we're specifically testing the edge case where a dict is passed
    result = metadata_type.process_bind_param(test_dict, None)  # type: ignore[arg-type]

    # Should return the dict as-is
    assert result == test_dict


def test_ensure_indexes_failure() -> None:
    """Test _ensure_indexes when index creation fails."""
    from dqx.orm.repositories import Base

    # Create a database with tables
    engine = create_engine("sqlite://", echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = db_session_factory(engine)

    # Create a MetricDB instance but prevent index creation in __init__
    with patch("dqx.orm.repositories.MetricDB._ensure_indexes"):
        db = MetricDB(factory)

    # Now test the _ensure_indexes method with a failing session
    with patch.object(db, "new_session") as mock_session:
        mock_session_instance = Mock()
        mock_session_instance.execute.side_effect = Exception("Index creation failed")
        mock_session.return_value = mock_session_instance

        # Should raise the exception and be caught
        with pytest.raises(Exception, match="Index creation failed"):
            db._ensure_indexes()


def test_get_by_key_with_dataset() -> None:
    """Test _get_by_key method with dataset parameter."""
    db = InMemoryMetricDB()

    # Create metrics with different datasets
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 10), tags={"env": "test"})
    spec = Average("page_views")

    metric1 = Metric.build(
        spec, key, dataset="dataset1", state=AverageState(10.0, 5), metadata=Metadata(execution_id="exec-1")
    )
    metric2 = Metric.build(
        spec, key, dataset="dataset2", state=AverageState(20.0, 5), metadata=Metadata(execution_id="exec-2")
    )

    db.persist([metric1, metric2])

    # Test getting by key with specific dataset
    result1 = db._get_by_key(key, spec, dataset="dataset1")
    assert result1.unwrap().dataset == "dataset1"
    assert result1.unwrap().value == pytest.approx(10.0)

    result2 = db._get_by_key(key, spec, dataset="dataset2")
    assert result2.unwrap().dataset == "dataset2"
    assert result2.unwrap().value == pytest.approx(20.0)

    # Test with non-existent dataset
    result3 = db._get_by_key(key, spec, dataset="non_existent")
    assert result3 == Nothing


def test_get_metric_window_returns_none() -> None:
    """Test get_metric_window when execute returns None (should not happen but handled)."""
    db = InMemoryMetricDB()

    # This test verifies the None check is properly handled
    # In practice, SQLAlchemy execute() shouldn't return None, but the code handles it
    with patch.object(db, "new_session") as mock_session:
        mock_session_instance = Mock()
        # Make execute return None
        mock_session_instance.execute.return_value = None
        mock_session.return_value = mock_session_instance

        spec = Average("test_column")
        key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 10), tags={})

        result = db.get_metric_window(spec, key, lag=0, window=5, dataset="test", execution_id="exec-123")

        # Should return Nothing when execute returns None
        assert result == Nothing
