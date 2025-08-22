import pytest
from unittest.mock import Mock, patch
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from dqx.orm import session


class TestGetEngine:
    """Test the get_engine function."""
    
    @patch('dqx.orm.session.create_engine')
    @patch('dqx.orm.session.logger')
    def test_get_engine_creates_correct_engine(self, mock_logger: Mock, mock_create_engine: Mock) -> None:
        """Test that get_engine creates an engine with correct parameters."""
        mock_engine = Mock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        result = session.get_engine()
        
        # Verify the engine was created with correct parameters
        mock_create_engine.assert_called_once_with(
            session.DB_URL,
            poolclass=NullPool,
            connect_args={"check_same_thread": False},
        )
        
        # Verify logging was called
        mock_logger.info.assert_called_once_with(
            "Creating SQLAlchemy Engine. Connection URL: %s", session.DB_URL
        )
        
        # Verify correct engine is returned
        assert result == mock_engine

    @patch('dqx.orm.session.create_engine')
    def test_get_engine_uses_sqlite_memory_url(self, mock_create_engine: Mock) -> None:
        """Test that get_engine uses the correct SQLite in-memory URL."""
        mock_create_engine.return_value = Mock(spec=Engine)
        
        session.get_engine()
        
        # Verify the correct URL is used
        call_args = mock_create_engine.call_args[0]
        assert call_args[0] == "sqlite://"


class TestDbSessionFactory:
    """Test the db_session_factory function."""
    
    @patch('dqx.orm.session.sessionmaker')
    @patch('dqx.orm.session.get_engine')
    def test_db_session_factory_with_default_engine(self, mock_get_engine: Mock, mock_sessionmaker: Mock) -> None:
        """Test db_session_factory with default engine."""
        # Setup mocks
        mock_engine = Mock(spec=Engine)
        mock_get_engine.return_value = mock_engine
        
        mock_session = Mock(spec=Session)
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        
        mock_factory = Mock()
        mock_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_factory
        
        # Test the generator
        session_gen = session.db_session_factory()
        
        # Get first session - this will yield the session
        result_session = next(session_gen)
        
        # Verify engine was created
        mock_get_engine.assert_called_once()
        
        # Verify sessionmaker was called with correct engine
        mock_sessionmaker.assert_called_with(bind=mock_engine)
        
        # Verify session is returned
        assert result_session == mock_session
        
        # The context manager should have been entered
        mock_session.__enter__.assert_called_once()
        
        # Commit is not called yet - only after next() is called again
        mock_session.commit.assert_not_called()
        
        # To trigger commit, we need to call next() again (which continues after yield)
        try:
            next(session_gen)
        except StopIteration:
            pass
        
        # Now commit should have been called
        mock_session.commit.assert_called_once()

    @patch('dqx.orm.session.sessionmaker')
    def test_db_session_factory_with_custom_engine(self, mock_sessionmaker: Mock) -> None:
        """Test db_session_factory with custom engine."""
        # Setup mocks
        custom_engine = Mock(spec=Engine)
        
        mock_session = Mock(spec=Session)
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        
        mock_factory = Mock()
        mock_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_factory
        
        # Test the generator with custom engine
        session_gen = session.db_session_factory(engine=custom_engine)
        
        # Get first session
        result_session = next(session_gen)
        
        # Verify sessionmaker was called with custom engine
        mock_sessionmaker.assert_called_with(bind=custom_engine)
        
        # Verify session is returned
        assert result_session == mock_session
        
        # Commit is not called yet
        mock_session.commit.assert_not_called()
        
        # To trigger commit, we need to call next() again
        try:
            next(session_gen)
        except StopIteration:
            pass
        
        # Now commit should have been called
        mock_session.commit.assert_called_once()

    @patch('dqx.orm.session.sessionmaker')
    @patch('dqx.orm.session.get_engine')
    def test_db_session_factory_handles_sqlalchemy_error(self, mock_get_engine: Mock, mock_sessionmaker: Mock) -> None:
        """Test that db_session_factory handles SQLAlchemy errors correctly."""
        # Setup mocks
        mock_engine = Mock(spec=Engine)
        mock_get_engine.return_value = mock_engine
        
        mock_session = Mock(spec=Session)
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.commit.side_effect = SQLAlchemyError("Test error")
        
        mock_factory = Mock()
        mock_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_factory
        
        # Test the generator
        session_gen = session.db_session_factory()
        
        # Get the session first
        result_session = next(session_gen)
        assert result_session == mock_session
        
        # The error should happen when we continue after the yield (call next again)
        with pytest.raises(SQLAlchemyError, match="Test error"):
            next(session_gen)
        
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch('dqx.orm.session.sessionmaker')
    @patch('dqx.orm.session.get_engine')
    def test_db_session_factory_is_generator(self, mock_get_engine: Mock, mock_sessionmaker: Mock) -> None:
        """Test that db_session_factory is a proper generator."""
        # Setup mocks
        mock_engine = Mock(spec=Engine)
        mock_get_engine.return_value = mock_engine
        
        mock_session1 = Mock(spec=Session)
        mock_session1.__enter__ = Mock(return_value=mock_session1)
        mock_session1.__exit__ = Mock(return_value=None)
        
        mock_session2 = Mock(spec=Session)
        mock_session2.__enter__ = Mock(return_value=mock_session2)
        mock_session2.__exit__ = Mock(return_value=None)
        
        mock_factory = Mock()
        # Return different sessions each time
        mock_factory.side_effect = [mock_session1, mock_session2]
        mock_sessionmaker.return_value = mock_factory
        
        # Test the generator
        session_gen = session.db_session_factory()
        
        # Get first session and complete its cycle
        first_session = next(session_gen)
        assert first_session == mock_session1
        
        # Complete first session cycle by calling next() again
        try:
            next(session_gen)
        except StopIteration:
            pass
        
        # Reset mock factory for second session
        mock_factory.side_effect = [mock_session2]
        
        # Get second session
        second_session = next(session_gen)
        assert second_session == mock_session2

    @patch('dqx.orm.session.sessionmaker')
    def test_db_session_factory_none_engine_uses_default(self, mock_sessionmaker: Mock) -> None:
        """Test that passing None engine uses default engine."""
        mock_session = Mock(spec=Session)
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        
        mock_factory = Mock()
        mock_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_factory
        
        with patch('dqx.orm.session.get_engine') as mock_get_engine:
            mock_engine = Mock(spec=Engine)
            mock_get_engine.return_value = mock_engine
            
            # Test with None engine
            session_gen = session.db_session_factory(engine=None)
            next(session_gen)
            
            # Should call get_engine
            mock_get_engine.assert_called_once()
            mock_sessionmaker.assert_called_with(bind=mock_engine)


class TestConstants:
    """Test module constants."""
    
    def test_db_url_constant(self) -> None:
        """Test that DB_URL constant is set correctly."""
        assert session.DB_URL == "sqlite://"
