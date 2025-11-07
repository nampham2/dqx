import datetime as dt

# Import logger directly to avoid circular import
import logging
import typing
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, ClassVar

import sqlalchemy as sa
from returns.maybe import Maybe
from sqlalchemy import BinaryExpression, ColumnElement, create_engine, delete, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.types import JSON, TypeDecorator

from dqx import models, specs
from dqx.common import Metadata, Parameters, ResultKey, Tags, TimeSeries
from dqx.orm.session import db_session_factory
from dqx.specs import MetricSpec, MetricType
from dqx.states import State

logger = logging.getLogger(__name__)

Predicate = BinaryExpression | ColumnElement[bool]

METRIC_TABLE = "dq_metric"


@dataclass
class MetricStats:
    """Statistics about metrics in the database."""

    total_metrics: int
    expired_metrics: int


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


class Base(DeclarativeBase):
    type_annotation_map: ClassVar = {
        datetime: sa.TIMESTAMP(timezone=True),
        Parameters: sa.JSON,
        Tags: sa.JSON,
    }


class Metric(Base):
    __tablename__ = METRIC_TABLE

    metric_id: Mapped[uuid.UUID] = mapped_column(nullable=False, primary_key=True, default=uuid.uuid4)
    metric_type: Mapped[str] = mapped_column(nullable=False)
    parameters: Mapped[Parameters] = mapped_column(nullable=False)
    dataset: Mapped[str] = mapped_column(nullable=False)
    state: Mapped[bytes] = mapped_column(nullable=False)
    value: Mapped[float] = mapped_column(nullable=False)
    yyyy_mm_dd: Mapped[dt.date] = mapped_column(nullable=False)
    tags: Mapped[Tags] = mapped_column(nullable=False)
    meta: Mapped[Metadata] = mapped_column(MetadataType, nullable=False, default=Metadata)
    created: Mapped[datetime] = mapped_column(nullable=False, server_default=func.current_timestamp())

    def to_model(self) -> models.Metric:
        _type = typing.cast(MetricType, self.metric_type)
        spec = self._reconstruct_spec(_type, self.parameters)
        state: State = spec.deserialize(self.state)
        key = ResultKey(yyyy_mm_dd=self.yyyy_mm_dd, tags=self.tags)

        return models.Metric.build(
            spec, key, dataset=self.dataset, state=state, metric_id=self.metric_id, metadata=self.meta
        )

    def to_spec(self) -> specs.MetricSpec:
        _type = typing.cast(MetricType, self.metric_type)
        return self._reconstruct_spec(_type, self.parameters)

    def _reconstruct_spec(self, metric_type: MetricType, parameters: Parameters) -> specs.MetricSpec:
        """Reconstruct a spec from stored parameters, handling constructor vs additional parameters."""
        import inspect

        spec_class = specs.registry[metric_type]

        # Extended metrics (DayOverDay, WeekOverWeek, Stddev) have specific constructor signatures
        # and don't accept a 'parameters' argument
        if metric_type in ["DayOverDay", "WeekOverWeek", "Stddev"]:
            # For extended metrics, pass all parameters as constructor args
            return typing.cast(typing.Any, spec_class)(**parameters)
        else:
            # For simple metrics, split into constructor params and additional params
            sig = inspect.signature(spec_class.__init__)

            constructor_params = {}
            additional_params = {}

            for key, value in parameters.items():
                if key in sig.parameters and key != "parameters":
                    constructor_params[key] = value
                else:
                    additional_params[key] = value

            # Create spec with constructor params and additional params
            # Cast to Any to avoid mypy issues with protocol constructors
            return typing.cast(typing.Any, spec_class)(**constructor_params, parameters=additional_params)


class MetricDB:
    def __init__(self, factory: Iterator[Session]) -> None:
        self._factory = factory
        self._mutex = Lock()

        # Create performance indexes
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create performance indexes if they don't exist."""
        with self._mutex:
            session = self.new_session()
            try:
                # Create index for efficient expiration queries
                session.execute(
                    text("""
                    CREATE INDEX IF NOT EXISTS idx_metric_expiration
                    ON dq_metric(created, json_extract(meta, '$.ttl_hours'))
                """)
                )
                # No manual commit needed - session factory handles it
            except Exception as e:
                # Index creation is required for expiration feature
                logger.error(f"Failed to create required metric expiration index: {e}")
                # No manual rollback needed - session factory handles it
                raise  # Re-raise to trigger automatic rollback

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
            dataset=metric.dataset,
            state=metric.state.serialize(),
            value=metric.value,
            yyyy_mm_dd=metric.key.yyyy_mm_dd,
            tags=metric.key.tags,
            meta=metric.metadata or Metadata(),
        )

    def persist(self, metrics: Iterable[models.Metric]) -> Iterable[models.Metric]:
        with self._mutex:
            session = self.new_session()
            db_metrics = list(map(MetricDB.to_db, metrics))

            # Ensure each metric has a unique timestamp by adding microseconds
            for i, dbm in enumerate(db_metrics):
                # Set created timestamp with microsecond precision to ensure ordering
                dbm.created = datetime.now(timezone.utc) + dt.timedelta(microseconds=i)

            session.add_all(db_metrics)
            session.commit()

            for dbm in db_metrics:
                session.refresh(dbm)

            return [metric.to_model() for metric in db_metrics]

    def delete(self, metric_id: uuid.UUID) -> None:
        with self._mutex:
            query = delete(Metric).where(Metric.metric_id == metric_id)
            self.new_session().execute(query)

    def get_metric(self, metric: MetricSpec, key: ResultKey, dataset: str, execution_id: str) -> Maybe[models.Metric]:
        """Get a single metric value for a specific dataset and execution.

        Args:
            metric: The metric specification.
            key: The result key containing date and tags.
            dataset: The dataset name.
            execution_id: The execution ID to filter by.

        Returns:
            Maybe containing the metric value if found, Nothing otherwise.
        """
        query = (
            select(Metric)
            .where(
                Metric.metric_type == metric.metric_type,
                Metric.parameters == metric.parameters,
                Metric.yyyy_mm_dd == key.yyyy_mm_dd,
                Metric.tags == key.tags,
                Metric.dataset == dataset,
                func.json_extract(Metric.meta, "$.execution_id") == execution_id,
            )
            .order_by(Metric.created.desc())
            .limit(1)
        )

        return Maybe.from_optional(self.new_session().scalar(query)).map(Metric.to_model)

    def get_metric_window(
        self, metric: MetricSpec, key: ResultKey, lag: int, window: int, dataset: str, execution_id: str
    ) -> Maybe[TimeSeries]:
        """Get metric values over a time window for a specific dataset and execution.

        Args:
            metric: The metric specification.
            key: The result key for the base date.
            lag: Number of days to lag from the base date.
            window: Number of days to include in the window.
            dataset: The dataset name.
            execution_id: The execution ID to filter by.

        Returns:
            Maybe containing the TimeSeries if found, Nothing otherwise.
        """
        from_date, until_date = key.range(lag, window)

        # Create CTE for finding latest metrics per day within execution
        latest_metrics_cte = (
            select(
                Metric,
                func.row_number().over(partition_by=Metric.yyyy_mm_dd, order_by=Metric.created.desc()).label("rn"),
            ).where(
                Metric.metric_type == metric.metric_type,
                Metric.parameters == metric.parameters,
                Metric.yyyy_mm_dd >= from_date,
                Metric.yyyy_mm_dd <= until_date,
                Metric.tags == key.tags,
                Metric.dataset == dataset,
                func.json_extract(Metric.meta, "$.execution_id") == execution_id,
            )
        ).cte("latest_metrics")

        # Select only the rows with rn=1 (the latest for each date)
        query = select(latest_metrics_cte).where(latest_metrics_cte.c.rn == 1).order_by(latest_metrics_cte.c.yyyy_mm_dd)
        result = self.new_session().execute(query)

        # Convert CTE results to metrics
        time_series: TimeSeries = {}
        for row in result:
            # Reconstruct Metric object from CTE columns
            metric_obj = Metric(
                metric_id=row.metric_id,
                metric_type=row.metric_type,
                parameters=row.parameters,
                dataset=row.dataset,
                state=row.state,
                value=row.value,
                yyyy_mm_dd=row.yyyy_mm_dd,
                tags=row.tags,
                meta=row.meta,
                created=row.created,
            )
            time_series[row.yyyy_mm_dd] = metric_obj.to_model()
        return Maybe.from_value(time_series)

    def get_by_execution_id(self, execution_id: str) -> Sequence[models.Metric]:
        """Retrieve all metrics with the specified execution ID.

        Args:
            execution_id: The execution ID to filter by

        Returns:
            Sequence of Metric models with matching execution_id in their metadata
        """
        # Use SQLAlchemy's JSON functions to filter at DB level
        query = select(Metric).where(func.json_extract(Metric.meta, "$.execution_id") == execution_id)

        return [metric.to_model() for metric in self.new_session().scalars(query)]

    def _build_expiration_filter(self, current_time: datetime) -> ColumnElement[bool]:
        """Build SQLAlchemy filter for expired metrics.

        A metric is expired if created + ttl_hours < current_time.

        Args:
            current_time: The reference time for expiration check

        Returns:
            SQLAlchemy filter expression
        """
        return func.strftime(
            "%Y-%m-%d %H:%M:%S",
            func.datetime(
                Metric.created,
                "+" + func.cast(func.json_extract(Metric.meta, "$.ttl_hours"), sa.String) + " hours",
            ),
        ) < func.strftime("%Y-%m-%d %H:%M:%S", current_time)

    def get_metrics_stats(self) -> MetricStats:
        """Get statistics about expired metrics in the database.

        Returns:
            MetricStats containing:
            - total_metrics: Total number of metrics
            - expired_metrics: Number of expired metrics
        """
        with self._mutex:
            session = self.new_session()

            # Get current UTC time
            current_time = datetime.now(timezone.utc)

            # Total metrics count
            total_metrics = session.query(func.count(Metric.metric_id)).scalar() or 0

            # Query for expired metrics count using the helper
            expired_count = (
                session.query(func.count(Metric.metric_id)).filter(self._build_expiration_filter(current_time)).scalar()
                or 0
            )

            return MetricStats(
                total_metrics=total_metrics,
                expired_metrics=expired_count,
            )

    def delete_expired_metrics(self) -> None:
        """Delete all expired metrics from the database.

        A metric is deleted if created + ttl_hours < current_time
        """
        with self._mutex:
            session = self.new_session()

            # Get current UTC time
            current_time = datetime.now(timezone.utc)

            # Create CTE for expired metrics using the helper
            expired_metrics_cte = (select(Metric.metric_id).where(self._build_expiration_filter(current_time))).cte(
                "expired_metrics"
            )

            # Delete metrics that exist in the CTE
            delete_query = delete(Metric).where(Metric.metric_id.in_(select(expired_metrics_cte.c.metric_id)))

            session.execute(delete_query)
            # No manual commit needed - session factory handles it


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
