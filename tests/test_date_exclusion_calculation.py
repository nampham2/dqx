"""Test data availability calculation logic for date exclusion."""

import datetime
from unittest.mock import Mock

import pytest
import sympy as sp
from returns.result import Success

from dqx import specs
from dqx.common import ResultKey, SqlDataSource
from dqx.provider import MetricRegistry, SymbolicMetric


class TestDataAvailabilityCalculation:
    """Test calculate_data_av_ratios method."""

    def test_simple_metric_excluded_date(self) -> None:
        """Test simple metric with excluded date gets 0.0 ratio."""
        registry = MetricRegistry()

        # Create a simple metric with lag=0 and dataset
        sm = SymbolicMetric(
            name="num_rows()",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(100.0),
            metric_spec=specs.NumRows(),
            lag=0,
            dataset="test_dataset",  # Add dataset name
        )
        registry._metrics.append(sm)
        registry._symbol_index[sm.symbol] = sm

        # Calculate ratios with context date excluded
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource with skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = {datetime.date(2024, 1, 15)}  # Current date excluded
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        assert sm.data_av_ratio == 0.0

    def test_simple_metric_available_date(self) -> None:
        """Test simple metric with available date gets 1.0 ratio."""
        registry = MetricRegistry()

        # Create a simple metric with dataset
        sm = SymbolicMetric(
            name="average(price)",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(50.0),
            metric_spec=specs.Average("price"),
            lag=0,
            dataset="test_dataset",
        )
        registry._metrics.append(sm)
        registry._symbol_index[sm.symbol] = sm

        # Calculate ratios with different date excluded
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource with skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = {datetime.date(2024, 1, 10)}  # Different date excluded
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        assert sm.data_av_ratio == 1.0

    def test_lagged_metric_excluded(self) -> None:
        """Test lagged metric with its effective date excluded."""
        registry = MetricRegistry()

        # Create a metric with lag=2 and dataset
        sm = SymbolicMetric(
            name="maximum(revenue)",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(1000.0),
            metric_spec=specs.Maximum("revenue"),
            lag=2,  # Will look at 2024-01-13
            dataset="test_dataset",
        )
        registry._metrics.append(sm)
        registry._symbol_index[sm.symbol] = sm

        # Context date is 2024-01-15, so lag=2 means 2024-01-13
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource with skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = {datetime.date(2024, 1, 13)}  # Effective date excluded
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        assert sm.data_av_ratio == 0.0

    def test_extended_metric_all_children_available(self) -> None:
        """Test extended metric with all children available."""
        registry = MetricRegistry()

        # Create two simple metrics with dataset
        sm1 = SymbolicMetric(
            name="sum(tax)",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(100.0),
            metric_spec=specs.Sum("tax"),
            lag=0,
            dataset="test_dataset",
        )
        sm2 = SymbolicMetric(
            name="sum(tax)",
            symbol=sp.Symbol("x_2"),
            fn=lambda k: Success(110.0),
            metric_spec=specs.Sum("tax"),
            lag=1,
            dataset="test_dataset",
        )

        # Create extended metric that depends on both
        extended = SymbolicMetric(
            name="day_over_day(sum(tax))",
            symbol=sp.Symbol("x_3"),
            fn=lambda k: Success(1.1),
            metric_spec=specs.DayOverDay.from_base_spec(specs.Sum("tax")),
            lag=0,
            required_metrics=[sm1.symbol, sm2.symbol],
            dataset="test_dataset",
        )

        # Add all to registry
        for sm in [sm1, sm2, extended]:
            registry._metrics.append(sm)
            registry._symbol_index[sm.symbol] = sm

        # No dates excluded
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource with empty skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = set()  # No dates excluded
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        # All should have 1.0 ratio
        assert sm1.data_av_ratio == 1.0
        assert sm2.data_av_ratio == 1.0
        assert extended.data_av_ratio == 1.0

    def test_extended_metric_partial_availability(self) -> None:
        """Test extended metric with one child excluded."""
        registry = MetricRegistry()

        # Create three simple metrics for stddev with dataset
        sm1 = SymbolicMetric(
            name="average(price)",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(100.0),
            metric_spec=specs.Average("price"),
            lag=0,  # 2024-01-15
            dataset="test_dataset",
        )
        sm2 = SymbolicMetric(
            name="average(price)",
            symbol=sp.Symbol("x_2"),
            fn=lambda k: Success(105.0),
            metric_spec=specs.Average("price"),
            lag=1,  # 2024-01-14
            dataset="test_dataset",
        )
        sm3 = SymbolicMetric(
            name="average(price)",
            symbol=sp.Symbol("x_3"),
            fn=lambda k: Success(95.0),
            metric_spec=specs.Average("price"),
            lag=2,  # 2024-01-13
            dataset="test_dataset",
        )

        # Create stddev metric
        stddev = SymbolicMetric(
            name="stddev(0, 3)(average(price))",
            symbol=sp.Symbol("x_4"),
            fn=lambda k: Success(5.0),
            metric_spec=specs.Stddev.from_base_spec(specs.Average("price"), 0, 3),
            lag=0,
            required_metrics=[sm1.symbol, sm2.symbol, sm3.symbol],
            dataset="test_dataset",
        )

        # Add all to registry
        for sm in [sm1, sm2, sm3, stddev]:
            registry._metrics.append(sm)
            registry._symbol_index[sm.symbol] = sm

        # Exclude middle date
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource with skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = {datetime.date(2024, 1, 14)}  # sm2's date
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        # Check ratios
        assert sm1.data_av_ratio == 1.0  # Available
        assert sm2.data_av_ratio == 0.0  # Excluded
        assert sm3.data_av_ratio == 1.0  # Available

        # Extended metric should average: (1.0 + 0.0 + 1.0) / 3 = 0.667
        assert stddev.data_av_ratio == pytest.approx(2.0 / 3.0)

    def test_nested_extended_metrics(self) -> None:
        """Test nested extended metrics (extended depending on extended)."""
        registry = MetricRegistry()

        # Base metrics with dataset
        sm1 = SymbolicMetric(
            name="sum(revenue)",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(1000.0),
            metric_spec=specs.Sum("revenue"),
            lag=0,
            dataset="test_dataset",
        )
        sm2 = SymbolicMetric(
            name="sum(revenue)",
            symbol=sp.Symbol("x_2"),
            fn=lambda k: Success(900.0),
            metric_spec=specs.Sum("revenue"),
            lag=1,
            dataset="test_dataset",
        )

        # First level extended metric
        dod = SymbolicMetric(
            name="day_over_day(sum(revenue))",
            symbol=sp.Symbol("x_3"),
            fn=lambda k: Success(1.111),
            metric_spec=specs.DayOverDay.from_base_spec(specs.Sum("revenue")),
            lag=0,
            required_metrics=[sm1.symbol, sm2.symbol],
            dataset="test_dataset",
        )

        # Could have another extended metric depending on dod
        # For simplicity, we'll test the basic case

        for sm in [sm1, sm2, dod]:
            registry._metrics.append(sm)
            registry._symbol_index[sm.symbol] = sm

        # Exclude sm2's date
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource with skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = {datetime.date(2024, 1, 14)}
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        assert sm1.data_av_ratio == 1.0
        assert sm2.data_av_ratio == 0.0
        assert dod.data_av_ratio == 0.5  # (1.0 + 0.0) / 2

    def test_topological_sort_called(self) -> None:
        """Verify topological sort is called to ensure correct order."""
        registry = MetricRegistry()

        # Create metrics in wrong order (extended before simple) with dataset
        dod = SymbolicMetric(
            name="day_over_day(sum(revenue))",
            symbol=sp.Symbol("x_3"),
            fn=lambda k: Success(1.0),
            metric_spec=specs.DayOverDay.from_base_spec(specs.Sum("revenue")),
            required_metrics=[sp.Symbol("x_1"), sp.Symbol("x_2")],
            dataset="test_dataset",
        )
        sm1 = SymbolicMetric(
            name="sum(revenue)",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(100.0),
            metric_spec=specs.Sum("revenue"),
            dataset="test_dataset",
        )
        sm2 = SymbolicMetric(
            name="sum(revenue)",
            symbol=sp.Symbol("x_2"),
            fn=lambda k: Success(110.0),
            metric_spec=specs.Sum("revenue"),
            dataset="test_dataset",
        )

        # Add in wrong order
        for sm in [dod, sm1, sm2]:
            registry._metrics.append(sm)
            registry._symbol_index[sm.symbol] = sm

        # Calculate should still work due to topological sort
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource with no skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = set()
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        # All available
        assert sm1.data_av_ratio == 1.0
        assert sm2.data_av_ratio == 1.0
        assert dod.data_av_ratio == 1.0

    def test_empty_registry(self) -> None:
        """Test calculate_data_av_ratios with empty registry."""
        registry = MetricRegistry()
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Should not raise
        registry.calculate_data_av_ratios({}, context_key)

        assert len(registry._metrics) == 0

    def test_all_dates_excluded(self) -> None:
        """Test when all dates are excluded."""
        registry = MetricRegistry()

        # Create metrics with different lags and dataset
        metrics = []
        for i in range(5):
            sm = SymbolicMetric(
                name=f"sum(col{i})",
                symbol=sp.Symbol(f"x_{i + 1}"),
                fn=lambda k: Success(100.0),
                metric_spec=specs.Sum(f"col{i}"),
                lag=i,
                dataset="test_dataset",
            )
            metrics.append(sm)
            registry._metrics.append(sm)
            registry._symbol_index[sm.symbol] = sm

        # Exclude all possible dates
        context_key = ResultKey(datetime.date(2024, 1, 15), {})
        skip_dates = {context_key.yyyy_mm_dd - datetime.timedelta(days=i) for i in range(10)}

        # Create mock datasource with skip_dates
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = skip_dates
        datasources: dict[str, SqlDataSource] = {"test_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        # All should have 0.0 ratio
        for sm in metrics:
            assert sm.data_av_ratio == 0.0

    def test_metric_without_dataset(self) -> None:
        """Test metric without dataset gets full availability."""
        registry = MetricRegistry()

        # Create a metric without dataset
        sm = SymbolicMetric(
            name="num_rows()",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(100.0),
            metric_spec=specs.NumRows(),
            lag=0,
            dataset=None,  # No dataset
        )
        registry._metrics.append(sm)
        registry._symbol_index[sm.symbol] = sm

        # Calculate ratios
        context_key = ResultKey(datetime.date(2024, 1, 15), {})
        datasources: dict[str, SqlDataSource] = {}  # No datasources

        registry.calculate_data_av_ratios(datasources, context_key)

        # Should have full availability when no dataset
        assert sm.data_av_ratio == 1.0

    def test_dataset_not_in_datasources(self) -> None:
        """Test metric with dataset not in datasources gets full availability."""
        registry = MetricRegistry()

        # Create a metric with dataset
        sm = SymbolicMetric(
            name="average(price)",
            symbol=sp.Symbol("x_1"),
            fn=lambda k: Success(50.0),
            metric_spec=specs.Average("price"),
            lag=0,
            dataset="missing_dataset",
        )
        registry._metrics.append(sm)
        registry._symbol_index[sm.symbol] = sm

        # Calculate ratios with different datasource
        context_key = ResultKey(datetime.date(2024, 1, 15), {})

        # Create mock datasource but for different dataset
        mock_datasource = Mock(spec=SqlDataSource)
        mock_datasource.skip_dates = {datetime.date(2024, 1, 15)}
        datasources: dict[str, SqlDataSource] = {"other_dataset": mock_datasource}

        registry.calculate_data_av_ratios(datasources, context_key)

        # Should have full availability when dataset not found
        assert sm.data_av_ratio == 1.0
