import logging
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, exc
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)
DB_URL: str = "sqlite://"


def get_engine() -> Engine:
    logger.info("Creating SQLAlchemy Engine. Connection URL: %s", DB_URL)
    return create_engine(
        DB_URL,
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )


def db_session_factory(engine: Engine | None = None) -> Iterator[Session]:  # pragma: no cover
    while True:
        factory = sessionmaker(bind=engine or get_engine())
        with factory() as session:
            try:
                yield session
                session.commit()
            except exc.SQLAlchemyError:
                session.rollback()
                raise
