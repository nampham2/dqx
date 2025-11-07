"""Test provider API with parameters support."""

import datetime

from dqx import specs
from dqx.common import ExecutionId, ResultKey
from dqx.models import Metric
from dqx.orm.repositories import InMemoryMetricDB
from dqx.provider import MetricProvider


def test_provider_methods_accept_parameters() -> None:
    """Test that all provider methods accept parameters argument."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=ExecutionId(), data_av_threshold=0.0)

    # Test parameters can be passed to each method
    parameters = {"param1": "value1", "param2": 42}

    # Test num_rows
    num_rows_sym = provider.num_rows(parameters=parameters)
    assert provider.get_symbol(num_rows_sym).metric_spec.parameters == parameters

    # Test first
    first_sym = provider.first("col1", parameters=parameters)
    spec_params = provider.get_symbol(first_sym).metric_spec.parameters
    assert spec_params["column"] == "col1"
    assert spec_params["param1"] == "value1"
    assert spec_params["param2"] == 42

    # Test average
    avg_sym = provider.average("col2", parameters=parameters)
    spec_params = provider.get_symbol(avg_sym).metric_spec.parameters
    assert spec_params["column"] == "col2"
    assert spec_params["param1"] == "value1"
    assert spec_params["param2"] == 42

    # Test minimum
    min_sym = provider.minimum("col3", parameters=parameters)
    spec_params = provider.get_symbol(min_sym).metric_spec.parameters
    assert spec_params["column"] == "col3"
    assert spec_params["param1"] == "value1"

    # Test maximum
    max_sym = provider.maximum("col4", parameters=parameters)
    spec_params = provider.get_symbol(max_sym).metric_spec.parameters
    assert spec_params["column"] == "col4"
    assert spec_params["param1"] == "value1"

    # Test sum
    sum_sym = provider.sum("col5", parameters=parameters)
    spec_params = provider.get_symbol(sum_sym).metric_spec.parameters
    assert spec_params["column"] == "col5"
    assert spec_params["param1"] == "value1"

    # Test null_count
    null_sym = provider.null_count("col6", parameters=parameters)
    spec_params = provider.get_symbol(null_sym).metric_spec.parameters
    assert spec_params["column"] == "col6"
    assert spec_params["param1"] == "value1"

    # Test variance
    var_sym = provider.variance("col7", parameters=parameters)
    spec_params = provider.get_symbol(var_sym).metric_spec.parameters
    assert spec_params["column"] == "col7"
    assert spec_params["param1"] == "value1"

    # Test duplicate_count
    dup_sym = provider.duplicate_count(["col8", "col9"], parameters=parameters)
    spec_params = provider.get_symbol(dup_sym).metric_spec.parameters
    assert spec_params["columns"] == ["col8", "col9"]
    assert spec_params["param1"] == "value1"

    # Test count_values
    count_sym = provider.count_values("col10", [1, 2, 3], parameters=parameters)
    spec_params = provider.get_symbol(count_sym).metric_spec.parameters
    assert spec_params["column"] == "col10"
    assert spec_params["values"] == [1, 2, 3]
    assert spec_params["param1"] == "value1"

    # Test unique_count
    unique_sym = provider.unique_count("col11", parameters=parameters)
    spec_params = provider.get_symbol(unique_sym).metric_spec.parameters
    assert spec_params["column"] == "col11"
    assert spec_params["param1"] == "value1"

    # Test custom_sql
    sql_sym = provider.custom_sql("SELECT COUNT(*) FROM table", parameters=parameters)
    spec_params = provider.get_symbol(sql_sym).metric_spec.parameters
    assert spec_params["sql_expression"] == "SELECT COUNT(*) FROM table"
    assert spec_params["param1"] == "value1"


def test_parameters_passed_to_ops() -> None:
    """Test that parameters are correctly passed to underlying ops."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=ExecutionId(), data_av_threshold=0.0)

    parameters = {"filter": "status = 'active'", "group_by": "category"}

    # Create metric with parameters
    avg_sym = provider.average("amount", parameters=parameters)

    # Get the spec and check analyzers have parameters
    symbolic_metric = provider.get_symbol(avg_sym)
    spec = symbolic_metric.metric_spec

    # Check that all analyzers got the parameters
    for analyzer in spec.analyzers:
        # Cast to SqlOp since we know all our analyzers implement it
        from dqx.ops import SqlOp

        assert isinstance(analyzer, SqlOp)
        assert analyzer.parameters == parameters


def test_parameters_in_metric_hash_and_equality() -> None:
    """Test that parameters affect metric hash and equality."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=ExecutionId(), data_av_threshold=0.0)

    # Create two metrics with different parameters
    params1 = {"filter": "status = 'active'"}
    params2 = {"filter": "status = 'inactive'"}

    avg1 = provider.average("amount", parameters=params1)
    avg2 = provider.average("amount", parameters=params2)

    spec1 = provider.get_symbol(avg1).metric_spec
    spec2 = provider.get_symbol(avg2).metric_spec

    # Should be different metrics due to different parameters
    assert spec1 != spec2
    assert hash(spec1) != hash(spec2)

    # Create another with same parameters
    avg3 = provider.average("amount", parameters=params1)
    spec3 = provider.get_symbol(avg3).metric_spec

    # Should be equal to first one
    assert spec1 == spec3
    assert hash(spec1) == hash(spec3)


def test_parameters_preserved_in_clone() -> None:
    """Test that parameters are preserved when cloning specs."""
    params = {"filter": "active = true", "limit": 100}

    # Create specs with parameters
    spec1 = specs.Average("amount", parameters=params)
    spec2 = spec1.clone()

    # Parameters should be copied
    assert spec2.parameters == {"column": "amount", "filter": "active = true", "limit": 100}

    # Modifying cloned parameters shouldn't affect original
    spec2_params = spec2.parameters
    spec2_params["new_param"] = "value"

    assert "new_param" not in spec1.parameters


def test_extended_metrics_inherit_parameters() -> None:
    """Test that extended metrics properly handle base metric parameters."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=ExecutionId(), data_av_threshold=0.0)

    # Create base metric with parameters
    params = {"filter": "region = 'US'"}
    base_metric = provider.average("revenue", parameters=params)

    # Create DoD from base metric
    dod = provider.ext.day_over_day(base_metric)

    # The DoD spec should reference the base spec with parameters
    dod_spec = provider.get_symbol(dod).metric_spec
    assert isinstance(dod_spec, specs.DayOverDay)

    # The base spec should have the parameters
    assert dod_spec.base_spec.parameters == {"column": "revenue", "filter": "region = 'US'"}


def test_parameters_in_metric_persistence() -> None:
    """Test that metrics with parameters are correctly persisted."""
    execution_id = ExecutionId()
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=execution_id, data_av_threshold=0.0)

    # Create metric with parameters
    params = {"filter": "active = true"}
    avg_spec = specs.Average("amount", parameters=params)

    # Create and persist a metric
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    # Create a state for the metric
    from dqx import states
    from dqx.common import Metadata

    state = states.Average(avg=100.0, n=10)
    metric = Metric.build(
        metric=avg_spec,
        key=key,
        dataset="test_data",
        state=state,
        metadata=Metadata(execution_id=execution_id),
    )

    provider.persist([metric])

    # Retrieve the metric
    result = provider.get_metric(avg_spec, key, "test_data", execution_id)

    assert result.unwrap().spec.parameters == {"column": "amount", "filter": "active = true"}


def test_create_metric_with_custom_sql_and_parameters() -> None:
    """Test creating custom SQL metrics with parameters."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=ExecutionId(), data_av_threshold=0.0)

    params = {"table_suffix": "_2024", "region": "US"}
    sql_expr = "SELECT COUNT(*) FROM orders WHERE region = :region"

    # Create custom SQL metric with parameters
    custom_sym = provider.custom_sql(sql_expr, parameters=params)

    # Verify the spec has the right parameters
    spec = provider.get_symbol(custom_sym).metric_spec
    assert isinstance(spec, specs.CustomSQL)
    assert spec.parameters["sql_expression"] == sql_expr
    assert spec.parameters["table_suffix"] == "_2024"
    assert spec.parameters["region"] == "US"


def test_parameter_combinations_unique_metrics() -> None:
    """Test that different parameter combinations create unique metrics."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=ExecutionId(), data_av_threshold=0.0)

    # Create multiple metrics with different parameter combinations
    symbols = []

    # No parameters
    symbols.append(provider.sum("amount"))

    # Single parameter
    symbols.append(provider.sum("amount", parameters={"filter": "active"}))

    # Multiple parameters
    symbols.append(provider.sum("amount", parameters={"filter": "active", "group": "A"}))

    # Different parameter values
    symbols.append(provider.sum("amount", parameters={"filter": "inactive", "group": "A"}))

    # All should be unique metrics
    specs = [provider.get_symbol(sym).metric_spec for sym in symbols]

    # Check all are unique
    for i, spec1 in enumerate(specs):
        for j, spec2 in enumerate(specs):
            if i != j:
                assert spec1 != spec2
                assert hash(spec1) != hash(spec2)


def test_parameters_empty_dict_vs_none() -> None:
    """Test behavior with empty dict vs None for parameters."""
    db = InMemoryMetricDB()
    provider = MetricProvider(db, execution_id=ExecutionId(), data_av_threshold=0.0)

    # Create with None (default)
    sym1 = provider.sum("amount")

    # Create with empty dict
    sym2 = provider.sum("amount", parameters={})

    # Should be treated the same
    spec1 = provider.get_symbol(sym1).metric_spec
    spec2 = provider.get_symbol(sym2).metric_spec

    assert spec1 == spec2
    assert spec1.parameters == {"column": "amount"}
    assert spec2.parameters == {"column": "amount"}
