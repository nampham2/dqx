import datetime as dt
import typing
import uuid
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from threading import Lock
from typing import Any, ClassVar, overload

import sqlalchemy as sa
from returns.maybe import Maybe, Nothing, Some
from sqlalchemy import BinaryExpression, ColumnElement, create_engine, delete, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from dqx import models, specs
from dqx.common import DQXError, ResultKey, Tags, TimeSeries
from dqx.orm.session import db_session_factory
from dqx.specs import MetricSpec, MetricType
from dqx.states import State

Predicate = BinaryExpression | ColumnElement[bool]

METRIC_TABLE = "dq_metric"


class Base(DeclarativeBase):
    type_annotation_map: ClassVar = {
        datetime: sa.TIMESTAMP(timezone=True),
        Tags: sa.JSON,
    }


class Metric(Base):
    __tablename__ = METRIC_TABLE

    metric_id: Mapped[uuid.UUID] = mapped_column(nullable=False, primary_key=True, default=lambda: uuid.uuid4())
    metric_type: Mapped[str] = mapped_column(nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(nullable=False)
    # dataset: Mapped[str] = mapped_column(nullable=False)
    state: Mapped[bytes] = mapped_column(nullable=False)
    value: Mapped[float] = mapped_column(nullable=False)
    yyyy_mm_dd: Mapped[dt.date] = mapped_column(nullable=False)
    tags: Mapped[Tags] = mapped_column(nullable=False)
    created: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    def to_model(self) -> models.Metric:
        _type = typing.cast(MetricType, self.metric_type)
        spec = specs.registry[_type](**self.parameters)
        state: State = spec.deserialize(self.state)
        key = ResultKey(yyyy_mm_dd=self.yyyy_mm_dd, tags=self.tags)

        return models.Metric.build(spec, key, state=state, metric_id=self.metric_id)

    def to_spec(self) -> specs.MetricSpec:
        _type = typing.cast(MetricType, self.metric_type)
        return specs.registry[_type](**self.parameters)


class MetricDB:
    def __init__(self, factory: Iterator[Session]) -> None:
        self._factory = factory
        self._mutex = Lock()

    def new_session(self) -> Session:
        # Create a new session for every request.
        # This simplifies the db access and make it safer in a multi-threaded environment.
        return next(self._factory)

    def exists(self, metric_id: uuid.UUID) -> bool:
        query = select(Metric.metric_id).where(Metric.metric_id == metric_id).limit(1)
        return (self.new_session().execute(query)).first() is not None

    @staticmethod
    def to_db(metric: models.Metric) -> Metric:
        return Metric(
            metric_id=metric.metric_id,
            metric_type=metric.spec.metric_type,
            parameters=metric.spec.parameters,
            state=metric.state.serialize(),
            value=metric.value,
            yyyy_mm_dd=metric.key.yyyy_mm_dd,
            tags=metric.key.tags,
        )

    def persist(self, metrics: Iterable[models.Metric]) -> Iterable[models.Metric]:
        with self._mutex:
            session = self.new_session()
            db_metrics = list(map(MetricDB.to_db, metrics))
            session.add_all(db_metrics)
            session.commit()

            for dbm in db_metrics:
                session.refresh(dbm)

            return [metric.to_model() for metric in db_metrics]

    @overload
    def get(self, key: uuid.UUID) -> Maybe[models.Metric]: ...

    @overload
    def get(self, key: ResultKey, spec: MetricSpec) -> Maybe[models.Metric]: ...

    def get(self, key: uuid.UUID | ResultKey, spec: MetricSpec | None = None) -> Maybe[models.Metric]:
        if isinstance(key, uuid.UUID):
            return self._get_by_uuid(key)

        if isinstance(key, ResultKey):
            if spec is None:
                raise DQXError("MetricSpec must be provided when using ResultKey!")
            return self._get_by_key(key, spec)

        raise DQXError(f"Unsupported key type: {type(key)}")

    def _get_by_uuid(self, metric_id: uuid.UUID | ResultKey) -> Maybe[models.Metric]:
        result = self.new_session().get(Metric, metric_id)
        if result:
            return Maybe.from_value(result.to_model())

        return Maybe.empty

    def _get_by_key(self, key: ResultKey, spec: MetricSpec) -> Maybe[models.Metric]:
        query = select(Metric).where(
            Metric.metric_type == spec.metric_type,
            Metric.parameters == spec.parameters,
            Metric.yyyy_mm_dd == key.yyyy_mm_dd,
            Metric.tags == key.tags,
        )

        result = self.new_session().scalar(query)

        if result:
            return Maybe.from_value(result.to_model())

        return Maybe.empty

    def search(self, *expressions: Predicate) -> Sequence[models.Metric]:
        if len(expressions) == 0:
            raise DQXError("Filter expressions cannot be empty")

        query = select(Metric).where(*expressions)
        return [metric.to_model() for metric in self.new_session().scalars(query)]

    def delete(self, metric_id: uuid.UUID) -> None:
        with self._mutex:
            query = delete(Metric).where(Metric.metric_id == metric_id)
            self.new_session().execute(query)

    def get_metric_value(self, metric: MetricSpec, key: ResultKey) -> Maybe[float]:
        """
        Get a single metric value based on the provided metric and the result key.

        Args:
            metric (MetricCore): The core metric information to search for.
            key (ResultKey): The key containing specific parameters to identify the metric.

        Returns:
            models.Metric: The found metric model.

        Raises:
            DQGuardError: If no metric is found matching the provided criteria.
        """
        query = select(Metric.value).where(
            Metric.metric_type == metric.metric_type,
            Metric.parameters == metric.parameters,
            Metric.yyyy_mm_dd == key.yyyy_mm_dd,
            Metric.tags == key.tags,
        )

        return Maybe.from_optional(self.new_session().scalar(query))

    def get_metric_window(self, metric: MetricSpec, key: ResultKey, lag: int, window: int) -> Maybe[TimeSeries]:
        from_date, until_date = key.range(lag, window)

        query = select(Metric).where(
            Metric.metric_type == metric.metric_type,
            Metric.parameters == metric.parameters,
            Metric.yyyy_mm_dd >= from_date,
            Metric.yyyy_mm_dd <= until_date,
            Metric.tags == key.tags,
        )

        result = self.new_session().scalars(query)
        if result is None:
            return Nothing

        return Some({r.yyyy_mm_dd: r.value for r in result.all()})


class InMemoryMetricDB(MetricDB):
    def __init__(self) -> None:
        engine = create_engine(
            "sqlite://",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        factory = db_session_factory(engine)
        Base.metadata.create_all(bind=engine)

        super().__init__(factory)
