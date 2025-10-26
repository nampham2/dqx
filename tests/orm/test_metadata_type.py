"""Tests for MetadataType custom type decorator."""

import pytest
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import Session, declarative_base

from dqx.common import Metadata
from dqx.orm.repositories import MetadataType

Base = declarative_base()


class MetadataTestModel(Base):  # type: ignore[misc, valid-type]
    """Test model with metadata column."""

    __tablename__ = "test_model"

    id = Column(Integer, primary_key=True)
    meta = Column(MetadataType, nullable=False, default=Metadata)


class TestMetadataType:
    """Test MetadataType serialization and deserialization."""

    @pytest.fixture
    def session(self) -> Session:  # type: ignore[misc]
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
        record = MetadataTestModel(meta=metadata)
        session.add(record)
        session.commit()

        # Retrieve and verify
        retrieved = session.query(MetadataTestModel).first()
        assert retrieved is not None
        assert retrieved.meta == metadata

    def test_default_metadata(self, session: Session) -> None:
        """Test default metadata creation."""
        record = MetadataTestModel()
        session.add(record)
        session.commit()

        retrieved = session.query(MetadataTestModel).first()
        assert retrieved is not None
        assert retrieved.meta == Metadata()
