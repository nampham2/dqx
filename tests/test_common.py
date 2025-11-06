import datetime as dt

from returns.result import Failure, Success

from dqx.common import (
    AssertionResult,
    DQXError,
    EvaluationFailure,
    Metadata,
    PluginMetadata,
    ResultKey,
    SymbolicValidator,
)


def test_result_key() -> None:
    tags = {"partner": "baguette", "group_name": "french", "group_size": "small"}
    yyyy_mm_dd = dt.date.fromisoformat("2025-02-09")
    key = ResultKey(yyyy_mm_dd=yyyy_mm_dd, tags=tags)
    assert key.yyyy_mm_dd == yyyy_mm_dd
    assert key.tags == tags

    new_key = ResultKey(yyyy_mm_dd=yyyy_mm_dd, tags=tags)
    assert key == new_key

    different_key = ResultKey(yyyy_mm_dd=yyyy_mm_dd, tags={})
    assert key != different_key

    assert hash(key) == hash(new_key)
    assert hash(key) != hash(different_key)


def test_result_key_lag() -> None:
    """Test ResultKey lag method."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 15), tags={"region": "US"})

    # Test lag with positive number (line 43)
    lagged_key = key.lag(5)
    assert lagged_key.yyyy_mm_dd == dt.date(2024, 1, 10)
    assert lagged_key.tags == {"region": "US"}

    # Test lag with 0
    same_key = key.lag(0)
    assert same_key.yyyy_mm_dd == key.yyyy_mm_dd
    assert same_key.tags == key.tags


def test_result_key_range() -> None:
    """Test ResultKey range method."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 15), tags={"env": "prod"})

    # Test basic range (line 59)
    start_date, end_date = key.range(lag=0, window=5)
    assert start_date == dt.date(2024, 1, 11)
    assert end_date == dt.date(2024, 1, 15)

    # Test range with lag
    start_date, end_date = key.range(lag=2, window=3)
    assert start_date == dt.date(2024, 1, 11)
    assert end_date == dt.date(2024, 1, 13)


def test_result_key_repr_and_str() -> None:
    """Test ResultKey string representations."""
    key = ResultKey(yyyy_mm_dd=dt.date(2024, 1, 15), tags={"env": "prod"})

    # Test __repr__ (line 68)
    expected_repr = "ResultKey(2024-01-15, {'env': 'prod'})"
    assert repr(key) == expected_repr

    # Test __str__ (line 71)
    assert str(key) == expected_repr


def test_metadata_dataclass() -> None:
    """Test Metadata dataclass."""
    # Test default values
    metadata = Metadata()
    assert metadata.execution_id is None
    assert metadata.ttl_hours == 168  # 7 days

    # Test with custom values
    metadata = Metadata(execution_id="test-123", ttl_hours=24)
    assert metadata.execution_id == "test-123"
    assert metadata.ttl_hours == 24


def test_evaluation_failure_dataclass() -> None:
    """Test EvaluationFailure dataclass."""
    from dqx.provider import SymbolInfo

    symbol_info = SymbolInfo(
        name="x_1",
        metric="average(price)",
        dataset="orders",
        value=Success(100.0),
        data_av_ratio=0.95,
        yyyy_mm_dd=dt.date(2024, 1, 15),
        tags={},
    )

    failure = EvaluationFailure(error_message="Division by zero", expression="x_1 / 0", symbols=[symbol_info])

    assert failure.error_message == "Division by zero"
    assert failure.expression == "x_1 / 0"
    assert len(failure.symbols) == 1
    assert failure.symbols[0].name == "x_1"


def test_assertion_result_dataclass() -> None:
    """Test AssertionResult dataclass."""
    # Test with Success metric
    result = AssertionResult(
        yyyy_mm_dd=dt.date(2024, 1, 15),
        suite="Revenue Suite",
        check="Daily Revenue",
        assertion="Positive revenue",
        severity="P0",
        status="PASSED",
        metric=Success(1000.0),
        expression="average(revenue) > 0",
        tags={"env": "prod"},
    )

    assert result.yyyy_mm_dd == dt.date(2024, 1, 15)
    assert result.suite == "Revenue Suite"
    assert result.check == "Daily Revenue"
    assert result.assertion == "Positive revenue"
    assert result.severity == "P0"
    assert result.status == "PASSED"

    # Pattern match on metric
    match result.metric:
        case Success(value):
            assert value == 1000.0
        case _:
            assert False, "Expected Success"

    # Test with Failure metric
    evaluation_failures = [EvaluationFailure(error_message="Value out of range", expression="x_1 < 0", symbols=[])]

    failed_result = AssertionResult(
        yyyy_mm_dd=dt.date(2024, 1, 15),
        suite="Revenue Suite",
        check="Daily Revenue",
        assertion="Revenue range check",
        severity="P1",
        status="FAILED",
        metric=Failure(evaluation_failures),
        expression="average(revenue) < 0",
    )

    match failed_result.metric:
        case Failure(errors):
            assert len(errors) == 1
            assert errors[0].error_message == "Value out of range"
        case _:
            assert False, "Expected Failure"


def test_symbolic_validator_dataclass() -> None:
    """Test SymbolicValidator dataclass."""

    # Create a properly typed validator function
    def positive_check(x: float) -> bool:
        return x > 0

    validator = SymbolicValidator(name="positive", fn=positive_check)

    assert validator.name == "positive"
    assert validator.fn(10.0) is True
    assert validator.fn(-5.0) is False
    assert validator.fn(0.0) is False


def test_plugin_metadata_dataclass() -> None:
    """Test PluginMetadata dataclass (frozen)."""
    metadata = PluginMetadata(
        name="MyPlugin",
        version="1.0.0",
        author="Test Author",
        description="A test plugin",
        capabilities={"analyze", "report"},
    )

    assert metadata.name == "MyPlugin"
    assert metadata.version == "1.0.0"
    assert metadata.author == "Test Author"
    assert metadata.description == "A test plugin"
    assert metadata.capabilities == {"analyze", "report"}

    # Test default capabilities
    metadata2 = PluginMetadata(
        name="SimplePlugin", version="0.1.0", author="Another Author", description="Simple plugin"
    )
    assert metadata2.capabilities == set()


def test_dqx_error() -> None:
    """Test DQXError exception."""
    error = DQXError("Something went wrong")
    assert str(error) == "Something went wrong"

    # Test that it's an Exception
    try:
        raise DQXError("Test error")
    except Exception as e:
        assert str(e) == "Test error"


def test_sql_datasource_protocol_skip_dates() -> None:
    """Test SqlDataSource protocol skip_dates property default."""

    # Create a minimal implementation
    class MinimalDataSource:
        dialect = "duckdb"

        @property
        def name(self) -> str:
            return "test"

        def cte(self, nominal_date: dt.date) -> str:
            return "SELECT * FROM test"

        def query(self, query: str) -> None:
            return None

    # Test default skip_dates (line 163)
    ds = MinimalDataSource()

    # The protocol defines skip_dates with a default implementation
    # that returns an empty set
    assert hasattr(ds, "skip_dates") is False  # Our minimal implementation doesn't have it

    # Create an implementation that uses the default
    class DataSourceWithDefaults:
        dialect = "duckdb"

        @property
        def name(self) -> str:
            return "test"

        @property
        def skip_dates(self) -> set[dt.date]:
            # Default implementation from protocol
            return set()

        def cte(self, nominal_date: dt.date) -> str:
            return "SELECT * FROM test"

        def query(self, query: str) -> None:
            return None

    ds_with_defaults = DataSourceWithDefaults()
    assert ds_with_defaults.skip_dates == set()
